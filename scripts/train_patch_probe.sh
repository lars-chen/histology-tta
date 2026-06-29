#!/bin/bash
#SBATCH --job-name=patch_probe
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python

MODEL=${MODEL:-uni}

mkdir -p "$PROJECT/logs" "$PROJECT/checkpoints/camelyon17"
cd "$PROJECT" || exit 1

echo "========================================"
echo "Model: $MODEL | Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================"

$PYTHON -m camelyon17.train_patch_probe \
    --model   "$MODEL" \
    --epochs  20 \
    --seed    42

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
