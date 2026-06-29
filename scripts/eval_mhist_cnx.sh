#!/bin/bash
#SBATCH --job-name=histo_mhist_cnx_eval
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints

cd "$PROJECT" || exit 1

for SEED in 42 0 1 2 3; do
    CKPT="$CKPT_DIR/mhist_convnextv2_tiny_frozen_aug_seed${SEED}_best.pt"
    echo "--- Eval convnextv2_tiny mhist seed $SEED ---"
    $PYTHON evaluate_tta.py \
        --model convnextv2_tiny \
        --checkpoint "$CKPT" \
        --dataset mhist \
        --tta_strategies none flips d4 d4_color \
        --aggregations mean vote confidence \
        --batch_size 16 \
        --amp \
        --seed "$SEED"
done
