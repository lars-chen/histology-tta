#!/bin/bash
#SBATCH --job-name=panda_heatmap
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
cd "$PROJECT" || exit 1
echo "Started: $(date)"

# One representative slide per ISUP grade (0-5)
$PROJECT/.venv/bin/python -u -m panda.wsi_tta_heatmap \
    --n_per_grade 1 \
    --out_dir figures/panda_wsi_heatmaps

echo "Finished: $(date)"
