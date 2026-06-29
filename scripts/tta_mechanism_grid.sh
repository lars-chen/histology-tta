#!/bin/bash
#SBATCH --job-name=tta_mech_grid
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=cpu_medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=0:40:00
cd /gpfs/data/mankowskilab/chen/histology-tta || exit 1
echo "==== $SLURM_JOB_ID $(date) ===="
.venv/bin/python -u -m utils.analysis.plot_tta_mechanism_grid
echo "==== done $(date) ===="
