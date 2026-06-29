#!/bin/bash
#SBATCH --job-name=cam17_extract
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4       # 1 GPU worker thread + 1 prefetch thread + headroom
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=0-7             # 8 workers × ~55 slides each = 441 total

# ---------------------------------------------------------------------------
# Extract UNI patch features from Camelyon17 WSIs.
#
# Each array task handles 1/8 of the slides independently on its own GPU.
# Within each task, patch reads are prefetched in a background thread while
# the GPU encodes the previous batch (hides OpenSlide I/O latency).
# AMP (fp16) is used for inference.
#
# Produces per-slide .pt files:
#   camelyon17_features/uni/none/<slide_id>.pt    (N, 1024)
#   camelyon17_features/uni/d4_all/<slide_id>.pt  (N, 8, 1024)
#
# Wall-time estimate: ~55 slides × ~3 min/slide = ~2.75 h per task
# (d4_all is 9× more compute than none; none alone is ~20 min per task)
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CACHE_DIR=${HF_HOME:-/gpfs/scratch/lpc8816/.cache/huggingface}

PATCHES_DIR=/gpfs/data/mankowskilab/chen/camelyon17_patched/patches
SLIDES_DIR=/gpfs/data/mankowskilab/chen/camelyon17_patched/slides_symlinks
OUT_DIR=$PROJECT/camelyon17_features

mkdir -p "$PROJECT/logs"
cd "$PROJECT" || exit 1

echo "========================================"
echo "Array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
echo "Node: $SLURMD_NODENAME  |  GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "========================================"

HF_HOME=$CACHE_DIR $PYTHON -m camelyon17.extract_features \
    --patches_dir  "$PATCHES_DIR" \
    --slides_dir   "$SLIDES_DIR" \
    --out_dir      "$OUT_DIR" \
    --model        uni \
    --tta_modes    none d4_all \
    --batch_size   512 \
    --patch_level  1 \
    --patch_size   256 \
    --device       cuda \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --n_tasks      "$SLURM_ARRAY_TASK_COUNT"

echo "========================================"
echo "Task ${SLURM_ARRAY_TASK_ID} finished: $(date)"
echo "========================================"
