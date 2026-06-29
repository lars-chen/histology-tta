#!/bin/bash
#SBATCH --job-name=export_wsi_panels
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00

# Graphical-abstract panels for patient_096_node_0, honest LOPO UNI probe:
#   figures/panels/p096_node0_outcome.pdf/.png  (corrected/corrupted/both_wrong on H&E)
#   figures/panels/p096_node0_delta_p.pdf/.png  (Δ P(tumor) on H&E)
# Dark-blue tumor outline, cropped to the two tissue masses.

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
cd "$PROJECT" || exit 1

export PANEL_MODEL="${1:-uni}"   # backbone: uni (default) / convnextv2 / ...

echo "==== Job $SLURM_JOB_ID on $SLURMD_NODENAME  $(date)  model=$PANEL_MODEL ===="
$PYTHON -u -m utils.export_wsi_panels
echo "==== Finished $(date) ===="
