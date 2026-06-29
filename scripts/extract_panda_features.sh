#!/bin/bash
#SBATCH --job-name=panda_feat
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=0-31

# Run AFTER patch_panda.sh completes.
# Extracts UNI D4 features for all PANDA slides.
# Features saved to: panda_features/uni/d4_all/<slide_id>.pt  (N, 8, 1024)

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CACHE_DIR=${HF_HOME:-/gpfs/scratch/lpc8816/hf_cache}

PANDA=/gpfs/scratch/lpc8816/panda

cd "$PROJECT" || exit 1

echo "========================================"
echo "Task: $SLURM_ARRAY_TASK_ID/32  Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================"

HF_HOME=$CACHE_DIR $PYTHON -m camelyon17.extract_features \
    --model        uni \
    --tta_modes    d4_all \
    --patches_dir  "$PANDA/clam_patches/patches" \
    --slides_dir   "$PANDA/train_images" \
    --out_dir      "$PROJECT/panda_features" \
    --patch_level  0 \
    --patch_size   256 \
    --batch_size   256 \
    --device       cuda \
    --task_id      $SLURM_ARRAY_TASK_ID \
    --n_tasks      32

echo "Finished: $(date)"
