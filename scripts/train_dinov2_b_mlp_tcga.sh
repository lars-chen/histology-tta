#!/bin/bash
#SBATCH --job-name=histo_dinov2_b_mlp_tcga
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

# ---------------------------------------------------------------------------
# MLP probe experiment: DINOv2-Base on TCGA-UT.
# MLP head: Linear(768→768) → BN → ReLU → Dropout → Linear(768→31)
# Paired with the linear probe runs (train_dinov2_b_cls.sh) to measure
# how much extra head capacity changes TTA gain (z_spur reliance).
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
LOG_DIR=$PROJECT/logs
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}
export TORCH_HOME=/gpfs/scratch/lpc8816/.cache/torch

MODEL=dinov2_b
MLP_HIDDEN=768   # = feature_dim; no bottleneck
DATASET=tcga-ut
BATCH_SIZE=32
EVAL_BATCH_SIZE=32
EPOCHS=20
PATIENCE=5
SEEDS=(0 1 2 3 42)

mkdir -p "$CKPT_DIR" "$LOG_DIR"
cd "$PROJECT" || exit 1

echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL (frozen, MLP hidden=$MLP_HIDDEN) | Dataset: $DATASET"
echo "Seeds: ${SEEDS[*]}"
echo "Started: $(date)"
echo "========================================"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "-------- Training: $MODEL MLP on $DATASET (seed $SEED) --------"

    $PYTHON train.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --freeze_backbone \
        --mlp_hidden "$MLP_HIDDEN" \
        --patience "$PATIENCE" \
        --num_workers 4 \
        --amp \
        --seed "$SEED" \
        --cache_dir "$CACHE_DIR" \
        --checkpoint_dir "$CKPT_DIR"

    CKPT="$CKPT_DIR/${DATASET}_${MODEL}_frozen_aug_mlp${MLP_HIDDEN}_seed${SEED}_best.pt"
    echo ""
    echo "-------- TTA Evaluation: $MODEL MLP on $DATASET (seed $SEED) --------"
    $PYTHON evaluate_tta.py \
        --model "$MODEL" \
        --checkpoint "$CKPT" \
        --dataset "$DATASET" \
        --tta_strategies none flips d4 \
        --aggregations mean vote confidence \
        --batch_size "$EVAL_BATCH_SIZE" \
        --amp \
        --seed "$SEED" \
        --cache_dir "$CACHE_DIR"
done

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "========================================"
