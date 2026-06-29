#!/bin/bash
#SBATCH --job-name=cam17_phikon2
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=0-7

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CACHE_DIR=${HF_HOME:-/gpfs/scratch/lpc8816/.cache/huggingface}

cd "$PROJECT" || exit 1

echo "========================================"
echo "Task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} | Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================"

HF_HOME=$CACHE_DIR $PYTHON -m camelyon17.extract_features \
    --model       phikon2 \
    --tta_modes   d4_all \
    --batch_size  512 \
    --patch_level 1 \
    --patch_size  256 \
    --device      cuda \
    --task_id     "$SLURM_ARRAY_TASK_ID" \
    --n_tasks     "$SLURM_ARRAY_TASK_COUNT"

echo "Finished: $(date)"
