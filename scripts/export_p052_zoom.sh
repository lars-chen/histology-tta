#!/bin/bash
#SBATCH --job-name=export_p052_zoom
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/export_p052_zoom_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/export_p052_zoom_%j.err
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python

source ~/.bashrc 2>/dev/null || true
cd "$PROJECT" || exit 1

$PYTHON utils/export_wsi_panels.py
