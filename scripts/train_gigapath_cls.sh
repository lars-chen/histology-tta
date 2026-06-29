#!/bin/bash
#SBATCH --job-name=histo_gigapath_cls_seeds
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00

# ---------------------------------------------------------------------------
# Train GigaPath (1.1B params, frozen backbone / linear probe) on all
# available datasets, then run TTA evaluation.
# Seeds 0-3 (seed 42 already complete). Total N=5.
#
# Weights downloaded from HF Hub. Requires HF_TOKEN for gated access.
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
LOG_DIR=$PROJECT/logs
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}

MODEL=gigapath
BATCH_SIZE=8
EVAL_BATCH_SIZE=4
EPOCHS=20
DATASETS=(tcga-ut nct-crc-100k nct-crc-nonorm)
SEEDS=(0 1 2 3)

mkdir -p "$CKPT_DIR" "$LOG_DIR"
cd "$PROJECT" || exit 1

echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL  | Batch: $BATCH_SIZE  | Epochs: $EPOCHS"
echo "Seeds: ${SEEDS[*]}"
echo "Started: $(date)"
echo "========================================"

for SEED in "${SEEDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo ""
        echo "-------- Training: $MODEL on $DATASET (seed $SEED) --------"

        $PYTHON train.py \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --freeze_backbone \
            --patience 5 \
            --num_workers 4 \
            --amp \
            --seed "$SEED" \
            --cache_dir "$CACHE_DIR" \
            --checkpoint_dir "$CKPT_DIR"

        CKPT="$CKPT_DIR/${DATASET}_${MODEL}_frozen_aug_seed${SEED}_best.pt"
        echo ""
        echo "-------- TTA Evaluation: $MODEL on $DATASET (seed $SEED) --------"
        $PYTHON evaluate_tta.py \
            --model "$MODEL" \
            --checkpoint "$CKPT" \
            --dataset "$DATASET" \
            --tta_strategies none flips d4 d4_color \
            --aggregations mean vote confidence \
            --batch_size "$EVAL_BATCH_SIZE" \
            --amp \
            --seed "$SEED" \
            --cache_dir "$CACHE_DIR"
    done
done

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "========================================"
