#!/bin/bash
#SBATCH --job-name=histo_mhist_missing
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# ---------------------------------------------------------------------------
# Fill missing mhist runs: convnextv2_base and dinov2_b (frozen, 5 seeds).
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
LOG_DIR=$PROJECT/logs
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}
export TORCH_HOME=/gpfs/scratch/lpc8816/.cache/torch

DATASET=mhist
EPOCHS=20
PATIENCE=5
SEEDS=(0 1 2 3 42)

mkdir -p "$CKPT_DIR" "$LOG_DIR"
cd "$PROJECT" || exit 1

echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Dataset: $DATASET  | Epochs: $EPOCHS"
echo "Seeds: ${SEEDS[*]}"
echo "Started: $(date)"
echo "========================================"

run_seed() {
    local MODEL=$1
    local BATCH_SIZE=$2
    local EVAL_BATCH_SIZE=$3
    local SEED=$4

    CKPT="$CKPT_DIR/${DATASET}_${MODEL}_frozen_aug_seed${SEED}_best.pt"

    if [ -f "$CKPT" ]; then
        echo "---- SKIP: $CKPT already exists ----"
        return
    fi

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
}

for SEED in "${SEEDS[@]}"; do
    run_seed convnextv2_base 32 32 "$SEED"
done

for SEED in "${SEEDS[@]}"; do
    run_seed dinov2_b 32 32 "$SEED"
done

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "========================================"
