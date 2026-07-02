#!/bin/bash
#SBATCH --job-name=cam17_extract
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7

# ---------------------------------------------------------------------------
# Extract D4 patch features from Camelyon17 WSIs for a given model.
#
# Usage: sbatch scripts/extract_camelyon17.sh <model> [slide_list.txt]
#
# Gated models (gigapath, virchow2) need HF_TOKEN set in the submission
# environment. For a100 nodes: `sbatch --partition=a100_short
# --gres=gpu:a100:1 --time=1-00:00:00 scripts/extract_camelyon17.sh <model>`.
# Larger batch sizes (512) are safe on a100; halve on other GPUs if you OOM.
# ---------------------------------------------------------------------------

set -euo pipefail

MODEL=$1
SLIDE_LIST=${2:-}

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CACHE_DIR=${HF_HOME:-/gpfs/scratch/lpc8816/.cache/huggingface}

case "$MODEL" in
    gigapath|virchow2)
        if [ -z "${HF_TOKEN:-}" ]; then
            echo "ERROR: $MODEL is gated. Submit with: HF_TOKEN=<token> sbatch $0 $MODEL" >&2
            exit 1
        fi
        BATCH=256 ;;
    dinov2-s|dinov2-b|resnet50|convnextv2|phikon2) BATCH=512 ;;
    uni) BATCH=512 ;;
    *) echo "No batch-size default for model '$MODEL', using 256" >&2; BATCH=256 ;;
esac

cd "$PROJECT"

echo "========================================"
echo "Model: $MODEL | Task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} | Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================"

EXTRA_ARGS=()
[[ -n "$SLIDE_LIST" ]] && EXTRA_ARGS+=(--slide_list "$SLIDE_LIST")

HF_HOME=$CACHE_DIR $PYTHON -m camelyon17.extract_features \
    --model       "$MODEL" \
    --tta_modes   d4_all \
    --batch_size  "$BATCH" \
    --patch_level 1 \
    --patch_size  256 \
    --device      cuda \
    --task_id     "$SLURM_ARRAY_TASK_ID" \
    --n_tasks     "$SLURM_ARRAY_TASK_COUNT" \
    "${EXTRA_ARGS[@]}"

echo "Finished: $(date)"
