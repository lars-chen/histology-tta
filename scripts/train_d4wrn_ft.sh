#!/bin/bash
#SBATCH --job-name=histo_d4wrn_ft_fill
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00

# ---------------------------------------------------------------------------
# Fill missing D4WRN finetuned runs:
#   tcga-ut:        seeds 2, 3
#   nct-crc-100k:   seeds 1, 2, 3
#   nct-crc-nonorm: seeds 1, 2, 3
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
LOG_DIR=$PROJECT/logs
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}

MODEL=d4wrn
BATCH_SIZE=64
EVAL_BATCH_SIZE=32
EPOCHS=30
PATIENCE=5

mkdir -p "$CKPT_DIR" "$LOG_DIR"
cd "$PROJECT" || exit 1

echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL (fill missing seeds)"
echo "Started: $(date)"
echo "========================================"

run_seed() {
    local DATASET=$1
    local SEED=$2

    CKPT="$CKPT_DIR/${DATASET}_${MODEL}_finetuned_aug_seed${SEED}_best.pt"

    # Skip if checkpoint already exists
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
        --lr 1e-3 \
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

# tcga-ut: missing seeds 2, 3
run_seed tcga-ut 2
run_seed tcga-ut 3

# nct-crc-100k: missing seeds 1, 2, 3
run_seed nct-crc-100k 1
run_seed nct-crc-100k 2
run_seed nct-crc-100k 3

# nct-crc-nonorm: missing seeds 1, 2, 3
run_seed nct-crc-nonorm 1
run_seed nct-crc-nonorm 2
run_seed nct-crc-nonorm 3

# mhist: missing seeds 1, 2, 3
run_seed mhist 1
run_seed mhist 2
run_seed mhist 3

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "========================================"
