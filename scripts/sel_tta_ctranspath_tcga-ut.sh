#!/bin/bash
#SBATCH --job-name=sel_tta_ctran_tcga-ut
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}
MODEL=ctranspath
DATASET=tcga-ut
BATCH=16

cd "$PROJECT" || exit 1

run_timed() {
    local label="$1"; shift
    echo ">>> $label"
    { time $PYTHON evaluate_tta.py "$@"; } 2>&1
    echo ""
}

for SEED in 0 1 2 3 42; do
    CKPT="$CKPT_DIR/${DATASET}_${MODEL}_frozen_aug_seed${SEED}_best.pt"
    [[ -f "$CKPT" ]] || { echo "SKIP seed$SEED (no checkpoint)"; continue; }

    # Skip if selective results already exist
    OUT="results/raw/selective_tta_${DATASET}_${MODEL}_frozen_aug_seed${SEED}.json"
    [[ -f "$PROJECT/$OUT" ]] && { echo "SKIP seed$SEED (already done)"; continue; }

    echo "---- seed=$SEED ----"

    COMMON="--model $MODEL --checkpoint $CKPT --dataset $DATASET
            --aggregations mean --batch_size $BATCH --amp --seed $SEED
            --cache_dir $CACHE_DIR"

    run_timed "No TTA + Full d4" $COMMON --tta_strategies none d4

    for T in 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
        run_timed "Selective t=$T" $COMMON \
            --tta_strategies none d4 \
            --selective_tta \
            --selective_threshold "$T"
    done

    echo ""
done

echo "Done: $(date)"
