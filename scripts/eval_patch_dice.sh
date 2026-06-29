#!/bin/bash
#SBATCH --job-name=eval_patch_dice
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
cd "$PROJECT" || exit 1
echo "Started: $(date)"

$PROJECT/.venv/bin/python -u eval_patch_dice.py \
    --dataset both \
    --out_dir "$PROJECT/figures" \
    --results_dir "$PROJECT/results"

echo "Finished: $(date)"
