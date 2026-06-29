#!/bin/bash
#SBATCH --job-name=ga_variance
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00

# Polished single-panel D4 view-variance heatmap for the graphical abstract.
# Full-strength out-of-sample UNI probe on patient_096_node_0 (held out).

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
cd "$PROJECT" || exit 1

echo "==== Job $SLURM_JOB_ID on $SLURMD_NODENAME  $(date) ===="

$PYTHON -u -m camelyon17.ga_variance_heatmap \
    --slide_id patient_096_node_0 \
    --checkpoint checkpoints/camelyon17/lopo_uni/heldout_patient_096.pt \
    --features_dir camelyon17_features/uni/d4_all \
    --cmap inferno

echo "==== Finished $(date) ===="
