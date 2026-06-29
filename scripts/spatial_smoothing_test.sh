#!/bin/bash
#SBATCH --job-name=spatial_smooth
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
cd "$PROJECT" || exit 1
echo "==== Job $SLURM_JOB_ID on $SLURMD_NODENAME  $(date) ===="
$PROJECT/.venv/bin/python -u -m camelyon17.spatial_smoothing_test
echo "==== Finished $(date) ===="
