#!/bin/bash
#SBATCH --job-name=wsi_heatmap
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=1:00:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
cd "$PROJECT" || exit 1

echo "Started: $(date)"
$PYTHON -u -m camelyon17.wsi_tta_heatmap \
    --all \
    --checkpoint "$PROJECT/checkpoints/camelyon17/patch_probe_uni_seed42/best.pt" \
    --out_dir    "$PROJECT/figures/wsi_heatmaps" \
    --wsi_level  4
echo "Finished: $(date)"
