#!/bin/bash
#SBATCH --job-name=lopo_wsi_heatmap
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00

# ---------------------------------------------------------------------------
# 1. Re-run UNI leave-one-patient-out, saving each fold's probe keyed by
#    held-out patient (throwaway CSVs so the real 6-model results are untouched).
# 2. Render the out-of-sample WSI TTA heatmap for patient_096_node_0 using the
#    probe that never saw patient_096.
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
cd "$PROJECT" || exit 1
mkdir -p "$PROJECT/logs"

echo "==== Job $SLURM_JOB_ID on $SLURMD_NODENAME  $(date) ===="

echo "---- Step 1: LOPO UNI with --save_probes ----"
$PYTHON -u -m camelyon17.lopo_patch_probe \
    --model uni --save_probes --device cuda \
    --dice_csv /tmp/lopo_uni_saveprobes_dice.csv \
    --bd_csv   /tmp/lopo_uni_saveprobes_bd.csv

echo "---- Step 2: out-of-sample heatmap for patient_096_node_0 ----"
$PYTHON -u -m camelyon17.wsi_tta_heatmap \
    --slide_id patient_096_node_0 \
    --checkpoint checkpoints/camelyon17/lopo_uni/heldout_patient_096.pt \
    --out_dir figures/wsi_heatmaps_lopo

echo "==== Finished $(date) ===="
