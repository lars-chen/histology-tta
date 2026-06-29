#!/bin/bash
#SBATCH --job-name=selective_tta_phikon2
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}
MODEL=phikon2
BATCH=16

cd "$PROJECT" || exit 1

run_timed() {
    # Usage: run_timed <label> <python args...>
    local label="$1"; shift
    echo ">>> $label"
    { time $PYTHON evaluate_tta.py "$@"; } 2>&1
    echo ""
}

for DATASET in tcga-ut mhist nct-crc-100k nct-crc-nonorm; do
    echo "========================================================"
    echo "DATASET: $DATASET"
    echo "========================================================"

    for SEED in 0 1 2 3 42; do
        CKPT="$CKPT_DIR/${DATASET}_${MODEL}_frozen_aug_seed${SEED}_best.pt"
        [[ -f "$CKPT" ]] || { echo "SKIP $DATASET seed$SEED (no checkpoint)"; continue; }

        echo "---- seed=$SEED ----"

        COMMON="--model $MODEL --checkpoint $CKPT --dataset $DATASET
                --aggregations mean --batch_size $BATCH --amp --seed $SEED
                --cache_dir $CACHE_DIR"

        run_timed "No TTA" $COMMON --tta_strategies none

        run_timed "Full d4 TTA" $COMMON --tta_strategies none d4

        for T in 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
            run_timed "Selective TTA t=$T" $COMMON \
                --tta_strategies none d4 \
                --selective_tta \
                --selective_threshold "$T"
        done

        echo ""
    done
done

echo "Done: $(date)"
