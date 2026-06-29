#!/bin/bash
#SBATCH --job-name=lopo_wsi_cand
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00

# ---------------------------------------------------------------------------
# Render additional out-of-sample (full-strength LOPO) WSI heatmaps for
# graphical-abstract candidate selection. Uses the already-saved LOPO probe
# checkpoints — no retraining. Candidates picked for legible tumor size
# (n_pos>=100) + clean correction signal:
#   convnextv2 patient_092_node_1  (9.6:1 corrected:corrupted, dDice +0.064)
#   dinov2-s   patient_020_node_4  (largest legible dDice +0.081)
#   convnextv2 patient_075_node_4  (large tumor, n_pos=809, net +68)
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
cd "$PROJECT" || exit 1
OUT=figures/wsi_heatmaps_candidates
mkdir -p "$OUT"

echo "==== Job $SLURM_JOB_ID on $SLURMD_NODENAME  $(date) ===="

for ITEM in \
  "convnextv2:patient_092_node_1:patient_092:lopo_convnextv2" \
  "dinov2-s:patient_020_node_4:patient_020:lopo_dinov2-s" \
  "convnextv2:patient_075_node_4:patient_075:lopo_convnextv2" ; do
    M="${ITEM%%:*}"; REST="${ITEM#*:}"
    SLIDE="${REST%%:*}"; REST="${REST#*:}"
    PAT="${REST%%:*}"; DIR="${REST##*:}"
    echo "---- heatmap $M $SLIDE (probe $DIR/heldout_$PAT) ----"
    $PYTHON -u -m camelyon17.wsi_tta_heatmap \
        --slide_id "$SLIDE" \
        --checkpoint "checkpoints/camelyon17/${DIR}/heldout_${PAT}.pt" \
        --features_dir "camelyon17_features/${M}/d4_all" \
        --out_dir "$OUT"
    mv "$OUT/wsi_tta_heatmap_${SLIDE}.png" "$OUT/${M}_${SLIDE}.png" 2>/dev/null
done

echo "==== Finished $(date) ===="
