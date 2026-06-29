#!/bin/bash
#SBATCH --job-name=patch_stats
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
cd "$PROJECT" || exit 1

echo "Started: $(date)"
$PYTHON -m camelyon17.aggregate_patch_stats \
    --checkpoint checkpoints/camelyon17/patch_probe_uni_seed42/best.pt \
    --features_dir camelyon17_features/uni/d4_all
echo "Finished: $(date)"
