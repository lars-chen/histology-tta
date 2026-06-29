#!/bin/bash
#SBATCH --job-name=convert_ckpt
#SBATCH --output=logs/convert_ckpt_%j.out
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu_short

cd /gpfs/data/mankowskilab/chen/histology-tta
source .venv/bin/activate
python convert_checkpoints.py
