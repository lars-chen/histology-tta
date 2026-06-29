#!/bin/bash
#SBATCH --job-name=panda_probe
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
cd "$PROJECT" || exit 1

echo "Started: $(date)"
$PYTHON -u -m panda.train_patch_probe \
    --model   uni \
    --epochs  20 \
    --seed    42
echo "Finished: $(date)"
