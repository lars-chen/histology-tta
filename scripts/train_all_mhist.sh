#!/bin/bash
#SBATCH --job-name=histo_mhist_all
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

# ---------------------------------------------------------------------------
# Train ALL models (frozen backbone / linear probe) on MHIST (2-class,
# HP vs SSA), then run TTA evaluation. 5 seeds per model.
#
# MHIST is small (2,175 train / 977 test) so this runs fast.
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
LOG_DIR=$PROJECT/logs
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}

DATASET=mhist
EPOCHS=20
SEEDS=(42 0 1 2 3)

# Model configs: name batch_size eval_batch_size
MODELS=(
    "phikon      32 32"
    "phikon2     32 32"
    "uni         32 32"
    "uni2        16 16"
    "virchow     16 16"
    "virchow2    16 16"
    "gigapath     8  4"
    "hoptimus     8  4"
    "convnextv2_tiny 64 64"
    "dinov2_s    64 64"
)

mkdir -p "$CKPT_DIR" "$LOG_DIR"
cd "$PROJECT" || exit 1

echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Dataset: $DATASET  | Epochs: $EPOCHS"
echo "Seeds: ${SEEDS[*]}"
echo "Started: $(date)"
echo "========================================"

for MODEL_CFG in "${MODELS[@]}"; do
    read -r MODEL BATCH_SIZE EVAL_BATCH_SIZE <<< "$MODEL_CFG"

    for SEED in "${SEEDS[@]}"; do
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
