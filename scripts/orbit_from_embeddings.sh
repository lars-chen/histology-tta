#!/bin/bash
#SBATCH --job-name=orbit_emb
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=cpu_medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python

mkdir -p "$PROJECT/logs"
cd "$PROJECT" || exit 1

echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

$PYTHON analysis_orbit_from_embeddings.py --per_class

echo "Finished: $(date)"
