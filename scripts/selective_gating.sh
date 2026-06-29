#!/bin/bash
#SBATCH --job-name=sel_gate
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=cpu_medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
cd /gpfs/data/mankowskilab/chen/histology-tta || exit 1
echo "Job $SLURM_JOB_ID start $(date)"
.venv/bin/python analysis_selective_gating.py --seed 0
echo "end $(date)"
