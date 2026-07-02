#!/bin/bash
#SBATCH --job-name=selective_tta
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00

# ---------------------------------------------------------------------------
# Sweep the selective-TTA entropy threshold across seeds/datasets for a model.
#
# Usage: sbatch scripts/selective_tta.sh <model> \
#            [--datasets "tcga-ut mhist nct-crc-100k nct-crc-nonorm"] \
#            [--thresholds "0.1 0.2 0.3 0.4 0.5 0.6 0.7"]
# ---------------------------------------------------------------------------

set -euo pipefail

MODEL=$1; shift
DATASETS_OVERRIDE=""
THRESHOLDS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) DATASETS_OVERRIDE=$2; shift 2 ;;
        --thresholds) THRESHOLDS_OVERRIDE=$2; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

case "$MODEL" in
    hoptimus) BATCH=4 ;;
    ctranspath|phikon2) BATCH=16 ;;
    *) echo "No batch-size default for model '$MODEL', using 16" >&2; BATCH=16 ;;
esac

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}

DATASETS=(${DATASETS_OVERRIDE:-tcga-ut mhist nct-crc-100k nct-crc-nonorm})
THRESHOLDS=(${THRESHOLDS_OVERRIDE:-0.1 0.2 0.3 0.4 0.5 0.6 0.7})

cd "$PROJECT"

run_timed() {
    local label="$1"; shift
    echo ">>> $label"
    { time $PYTHON evaluate_tta.py "$@"; } 2>&1
    echo ""
}

for DATASET in "${DATASETS[@]}"; do
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

        for T in "${THRESHOLDS[@]}"; do
            run_timed "Selective TTA t=$T" $COMMON \
                --tta_strategies none d4 \
                --selective_tta \
                --selective_threshold "$T"
        done

        echo ""
    done
done

echo "Done: $(date)"
