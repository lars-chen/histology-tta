#!/bin/bash
#SBATCH --job-name=plot_d4_orbit
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

echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

$PYTHON plot_d4_orbit.py \
    --checkpoint checkpoints/mhist_dinov2_b_frozen_aug_seed0_best.pt \
    --data_dir data/mhist \
    --n_candidates 977 \
    --out_dir figures \
    --device cuda

echo "Finished: $(date)"
