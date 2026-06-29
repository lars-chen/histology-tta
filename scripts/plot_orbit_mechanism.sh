#!/bin/bash
#SBATCH --job-name=plot_orbit
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=cpu_medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
cd /gpfs/data/mankowskilab/chen/histology-tta || exit 1
echo "start $(date)"
.venv/bin/python plot_orbit_mechanism.py
echo "end $(date)"
