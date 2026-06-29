#!/bin/bash
#SBATCH --job-name=per_sample_tta
#SBATCH --output=logs/per_sample_%j.out
#SBATCH --error=logs/per_sample_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /gpfs/data/mankowskilab/chen/histology-tta
source .venv/bin/activate

mkdir -p results/per_sample logs

# MHIST — phikon, all 5 seeds
for SEED in 0 1 2 3 42; do
    python eval_per_sample.py --model phikon --dataset mhist --seed $SEED --device cuda
done

# NCT-CRC-100K — phikon, all 5 seeds (loads from HuggingFace; needs internet or cache)
for SEED in 0 1 2 3 42; do
    python eval_per_sample.py --model phikon --dataset nct-crc-100k --seed $SEED --device cuda
done

echo "Done."
