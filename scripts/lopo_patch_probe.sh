#!/bin/bash
#SBATCH --job-name=lopo_patch_probe
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# ---------------------------------------------------------------------------
# Leave-one-patient-out patch-probe evaluation on the 50 annotated Camelyon17
# slides, across all 6 backbones. Every slide is scored out-of-sample by a
# probe trained on the other patients. Single class-balancing (sampler only).
# Writes:
#   results/patch_dice_camelyon17_lopo.csv
#   results/camelyon17_boundary_dist_lopo.csv
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
cd "$PROJECT" || exit 1
mkdir -p "$PROJECT/logs"

echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================"

$PYTHON -u -m camelyon17.lopo_patch_probe \
    --model uni phikon2 virchow2 gigapath convnextv2 dinov2-s \
    --device cuda

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
