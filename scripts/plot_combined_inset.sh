#!/bin/bash
#SBATCH --job-name=plot_combined_inset
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
export TORCH_HOME=/gpfs/scratch/lpc8816/.cache/torch

cd "$PROJECT" || exit 1

$PYTHON utils/plot_combined_inset.py
