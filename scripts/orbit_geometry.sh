#!/bin/bash
#SBATCH --job-name=orbit_geom
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=cpu_medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
cd "$PROJECT" || exit 1
mkdir -p logs

echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

$PYTHON analysis_orbit_geometry.py --seed 0 --save_persample

echo "Finished: $(date)"
