#!/bin/bash
#SBATCH --job-name=mil_attn
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
cd /gpfs/data/mankowskilab/chen/histology-tta || exit 1
export PYTHONPATH=$PWD:$PYTHONPATH
echo "start $(date)"
.venv/bin/python analysis_mil_attention.py
echo "end $(date)"
