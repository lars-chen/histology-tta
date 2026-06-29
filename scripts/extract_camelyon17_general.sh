#!/bin/bash
#SBATCH --job-name=cam17_general
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=0-15

# Array layout: tasks 0-7 → resnet50, tasks 8-15 → convnextv2
# Submit with: sbatch scripts/extract_camelyon17_general.sh

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CACHE_DIR=${HF_HOME:-/gpfs/scratch/lpc8816/.cache/huggingface}

cd "$PROJECT" || exit 1

if [ "$SLURM_ARRAY_TASK_ID" -lt 8 ]; then
    MODEL=resnet50
    LOCAL_TASK=$SLURM_ARRAY_TASK_ID
else
    MODEL=convnextv2
    LOCAL_TASK=$((SLURM_ARRAY_TASK_ID - 8))
fi

echo "========================================"
echo "Model: $MODEL | Task ${LOCAL_TASK}/8 | Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================"

HF_HOME=$CACHE_DIR $PYTHON -m camelyon17.extract_features \
    --model       "$MODEL" \
    --tta_modes   d4_all \
    --batch_size  512 \
    --patch_level 1 \
    --patch_size  256 \
    --device      cuda \
    --task_id     "$LOCAL_TASK" \
    --n_tasks     8

echo "Finished: $(date)"
