#!/bin/bash
#SBATCH --job-name=panda_download
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=8:00:00

DEST=/gpfs/scratch/lpc8816/panda
mkdir -p "$DEST"

echo "Started: $(date)"
echo "Downloading PANDA to $DEST"

kaggle competitions download prostate-cancer-grade-assessment -p "$DEST"

echo "Download finished: $(date)"
echo "Unzipping..."
cd "$DEST" && unzip -q prostate-cancer-grade-assessment.zip
echo "Done: $(date)"
ls -lh "$DEST"
