#!/bin/bash
#SBATCH --job-name=lopo_wsi_general
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00

# ---------------------------------------------------------------------------
# Save leave-one-patient-out probes for the general-purpose backbones, then
# render out-of-sample WSI heatmaps for their biggest-effect slides:
#   convnextv2 patient_022_node_4  (+6.9 pp, 203/51, net +152 — most corrections)
#   dinov2-s   patient_044_node_4  (+7.1 pp, 51/7)
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
cd "$PROJECT" || exit 1

echo "==== Job $SLURM_JOB_ID on $SLURMD_NODENAME  $(date) ===="

echo "---- Step 1: save LOPO probes for convnextv2 + dinov2-s ----"
$PYTHON -u -m camelyon17.lopo_patch_probe \
    --model convnextv2 dinov2-s --save_probes --device cuda \
    --dice_csv /tmp/lopo_general_saveprobes_dice.csv \
    --bd_csv   /tmp/lopo_general_saveprobes_bd.csv

echo "---- Step 2: heatmap convnextv2 patient_022_node_4 ----"
$PYTHON -u -m camelyon17.wsi_tta_heatmap \
    --slide_id patient_022_node_4 \
    --checkpoint checkpoints/camelyon17/lopo_convnextv2/heldout_patient_022.pt \
    --features_dir camelyon17_features/convnextv2/d4_all \
    --out_dir figures/wsi_heatmaps_lopo

echo "---- Step 3: heatmap dinov2-s patient_044_node_4 ----"
$PYTHON -u -m camelyon17.wsi_tta_heatmap \
    --slide_id patient_044_node_4 \
    --checkpoint checkpoints/camelyon17/lopo_dinov2-s/heldout_patient_044.pt \
    --features_dir camelyon17_features/dinov2-s/d4_all \
    --out_dir figures/wsi_heatmaps_lopo

echo "==== Finished $(date) ===="
