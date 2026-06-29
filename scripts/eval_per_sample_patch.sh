#!/bin/bash
#SBATCH --job-name=eval_per_sample_patch
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
cd "$PROJECT" || exit 1
echo "Started: $(date)"

echo "=== MHIST phikon ==="
$PROJECT/.venv/bin/python -u eval_per_sample.py --dataset mhist --model phikon --seed 42

echo "=== NCT-CRC-100K phikon ==="
$PROJECT/.venv/bin/python -u eval_per_sample.py --dataset nct-crc-100k --model phikon --seed 42

echo "Finished: $(date)"
