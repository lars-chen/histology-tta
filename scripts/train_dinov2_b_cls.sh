#!/bin/bash
#SBATCH --job-name=histo_dinov2_b_cls_seeds
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

# ---------------------------------------------------------------------------
# Train DINOv2-Base (frozen backbone / linear probe) on all available
# datasets, then run TTA evaluation on datasets with a dedicated test split.
# 5 seeds for variance estimation.
#
# DINOv2 weights are downloaded from facebookresearch/dinov2 on first run.
# Set TORCH_HUB_DIR if you want to cache them somewhere specific.
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
LOG_DIR=$PROJECT/logs
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}
export TORCH_HOME=/gpfs/scratch/lpc8816/.cache/torch

MODEL=dinov2_b
BATCH_SIZE=32
EVAL_BATCH_SIZE=32
EPOCHS=20
PATIENCE=5
DATASETS=(tcga-ut nct-crc-100k nct-crc-nonorm)
SEEDS=(0 1 2 3 42)

mkdir -p "$CKPT_DIR" "$LOG_DIR"
cd "$PROJECT" || exit 1

echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL (frozen)  | Batch: $BATCH_SIZE  | Epochs: $EPOCHS"
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
            --patience "$PATIENCE" \
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
