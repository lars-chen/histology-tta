#!/bin/bash
#SBATCH --job-name=calib_test
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
cd /gpfs/data/mankowskilab/chen/histology-tta || exit 1
echo "==== $SLURM_JOB_ID $(date) ===="
.venv/bin/python -u -m camelyon17.calibration_test
echo "==== done $(date) ===="
