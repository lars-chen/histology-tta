#!/bin/bash
#SBATCH --job-name=selective_tta_hoptimus
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}
MODEL=hoptimus
DATASET=tcga-ut

cd "$PROJECT" || exit 1

for SEED in 0 1 2 3 42; do
    CKPT="$CKPT_DIR/${DATASET}_${MODEL}_frozen_aug_seed${SEED}_best.pt"
    [[ -f "$CKPT" ]] || { echo "SKIP seed$SEED (no checkpoint)"; continue; }

    echo "==== seed=$SEED ===="

    echo "-- Full TTA (baseline comparison) --"
    $PYTHON evaluate_tta.py \
        --model "$MODEL" \
        --checkpoint "$CKPT" \
        --dataset "$DATASET" \
        --tta_strategies none d4 \
        --aggregations mean \
        --batch_size 4 \
        --amp \
        --seed "$SEED" \
        --cache_dir "$CACHE_DIR"

    echo ""
    echo "-- Selective TTA (threshold=0.3) --"
    $PYTHON evaluate_tta.py \
        --model "$MODEL" \
        --checkpoint "$CKPT" \
        --dataset "$DATASET" \
        --tta_strategies none d4 \
        --aggregations mean \
        --selective_tta \
        --selective_threshold 0.3 \
        --batch_size 4 \
        --amp \
        --seed "$SEED" \
        --cache_dir "$CACHE_DIR"

    echo ""
done

echo "Done: $(date)"
