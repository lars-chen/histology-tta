#!/bin/bash
#SBATCH --job-name=lopo_wsi_ga
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00

# ---------------------------------------------------------------------------
# Render out-of-sample WSI heatmaps from DELIBERATELY DATA-LIMITED probes
# (10% of training patches) for the graphical abstract. A weaker baseline has
# many correctable boundary errors, so the correction pattern is dense and
# margin-concentrated (honest: out-of-sample + explicitly data-limited).
#   ConvNeXtV2 p022 / p096   (general — densest corrections)
#   UNI        p096          (histology FM — on-narrative, but robust → sparser)
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
cd "$PROJECT" || exit 1
OUT=figures/wsi_heatmaps_ga

echo "==== Job $SLURM_JOB_ID on $SLURMD_NODENAME  $(date) ===="

echo "---- save 10%-data probes: convnextv2 ----"
$PYTHON -u -m camelyon17.lopo_patch_probe \
    --model convnextv2 --save_probes --probe_frac 0.1 --device cuda \
    --dice_csv /tmp/ga_convnextv2_dice.csv --bd_csv /tmp/ga_convnextv2_bd.csv

echo "---- save 5%-data probes: uni ----"
$PYTHON -u -m camelyon17.lopo_patch_probe \
    --model uni --save_probes --probe_frac 0.05 --device cuda \
    --dice_csv /tmp/ga_uni_dice.csv --bd_csv /tmp/ga_uni_bd.csv

for ITEM in \
  "convnextv2:patient_022_node_4:patient_022:lopo_convnextv2_frac0.1" \
  "convnextv2:patient_096_node_0:patient_096:lopo_convnextv2_frac0.1" \
  "uni:patient_096_node_0:patient_096:lopo_uni_frac0.05" ; do
    M="${ITEM%%:*}"; REST="${ITEM#*:}"
    SLIDE="${REST%%:*}"; REST="${REST#*:}"
    PAT="${REST%%:*}"; DIR="${REST##*:}"
    echo "---- heatmap $M $SLIDE (probe $DIR/heldout_$PAT) ----"
    $PYTHON -u -m camelyon17.wsi_tta_heatmap \
        --slide_id "$SLIDE" \
        --checkpoint "checkpoints/camelyon17/${DIR}/heldout_${PAT}.pt" \
        --features_dir "camelyon17_features/${M}/d4_all" \
        --out_dir "$OUT"
    mv "$OUT/wsi_tta_heatmap_${SLIDE}.png" "$OUT/${M}_${SLIDE}_datalimited.png" 2>/dev/null
done

echo "==== Finished $(date) ===="
