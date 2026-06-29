#!/bin/bash
#SBATCH --job-name=lopo_wsi_heatmap2
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00

# ---------------------------------------------------------------------------
# Render out-of-sample WSI TTA heatmaps for the striking candidate slides,
# using the saved leave-one-patient-out UNI probes (each slide scored by a
# probe that never saw that patient). Probes were saved by scripts/lopo_wsi_heatmap.sh
#   patient_010_node_4  → heldout_patient_010.pt   (+14.3 pp, 40/3)
#   patient_046_node_3  → heldout_patient_046.pt   (+7.3 pp, 18/0)
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
cd "$PROJECT" || exit 1

echo "==== Job $SLURM_JOB_ID on $SLURMD_NODENAME  $(date) ===="

for ITEM in "patient_010_node_4:patient_010" "patient_046_node_3:patient_046"; do
    SLIDE="${ITEM%%:*}"
    PAT="${ITEM##*:}"
    echo "---- heatmap $SLIDE (held-out probe $PAT) ----"
    $PYTHON -u -m camelyon17.wsi_tta_heatmap \
        --slide_id "$SLIDE" \
        --checkpoint "checkpoints/camelyon17/lopo_uni/heldout_${PAT}.pt" \
        --out_dir figures/wsi_heatmaps_lopo
done

echo "==== Finished $(date) ===="
