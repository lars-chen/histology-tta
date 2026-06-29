#!/bin/bash
#SBATCH --job-name=panda_heatmap_all
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
cd "$PROJECT" || exit 1
echo "Started: $(date)"

$PROJECT/.venv/bin/python -u -m panda.wsi_tta_heatmap \
    --all \
    --out_dir "$PROJECT/figures/panda_wsi_heatmaps"

echo "Finished: $(date)"
