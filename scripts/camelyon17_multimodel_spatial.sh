#!/bin/bash
#SBATCH --job-name=c17_mm_spatial
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00

# ---------------------------------------------------------------------------
# Multi-model patch-level spatial TTA analysis on the 50 pixel-annotated
# Camelyon17 slides. Trains patch probes for the 5 backbones that lack one
# (UNI already trained), then runs the boundary-distance + patch-F1 driver
# across all 6 backbones. Encoder-robustness check on a fixed slide set.
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
cd "$PROJECT" || exit 1

echo "Started: $(date)"

for M in phikon2 gigapath virchow2 convnextv2 dinov2-s; do
    CKPT="checkpoints/camelyon17/patch_probe_${M}_seed42/best.pt"
    if [ -f "$CKPT" ]; then
        echo "== probe exists, skipping train: $M =="
    else
        echo "== training patch probe: $M =="
        $PYTHON -m camelyon17.train_patch_probe --model "$M" --seed 42
    fi
done

echo "== running multi-model spatial driver =="
$PYTHON -m camelyon17.multimodel_spatial

echo "Finished: $(date)"
