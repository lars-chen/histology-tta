#!/bin/bash
#SBATCH --job-name=flip_disc
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=cpu_medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
cd "$PROJECT" || exit 1
echo "Job: $SLURM_JOB_ID  Started: $(date)"
$PROJECT/.venv/bin/python analysis_flip_discrimination.py --seed 0
echo "Finished: $(date)"
