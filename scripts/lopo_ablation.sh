#!/bin/bash
#SBATCH --job-name=lopo_ablation
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# ---------------------------------------------------------------------------
# Data-ablation: leave-one-patient-out patch probes trained on decreasing
# fractions of the (stratified) training patches. As the baseline weakens,
# does the TTA gain grow, and do corrections concentrate spatially (margin)?
# One histology FM (UNI) and one general-purpose model (ConvNeXtV2).
# Writes:
#   results/patch_dice_camelyon17_ablation.csv   (per-slide, per train_frac)
#   results/camelyon17_margin_ablation.csv       (binned margin histogram)
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
cd "$PROJECT" || exit 1

echo "==== Job $SLURM_JOB_ID on $SLURMD_NODENAME  $(date) ===="

$PYTHON -u -m camelyon17.lopo_patch_probe \
    --ablation --model uni convnextv2 --device cuda \
    --train_frac 0.02 0.05 0.1 0.25 0.5 1.0

echo "==== Finished $(date) ===="
