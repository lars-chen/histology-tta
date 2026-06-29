#!/bin/bash
#SBATCH --job-name=relin_full
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/relin_%A_%a.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/relin_%A_%a.err
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=0-59

# Retrain ONLY the linear head on FULL training data (no 50k cap) for the
# headline table — so reported FM baselines match published values. CV C is
# still selected on a 25k subsample; only the final refit uses all data.
PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
export TORCH_HOME=/gpfs/scratch/lpc8816/.cache/torch
export HF_HOME=/gpfs/scratch/lpc8816/.cache/huggingface
cd "$PROJECT" || exit 1

MODELS=(
    phikon phikon2 uni uni2
    virchow virchow2 gigapath hoptimus
    ctranspath
    dinov2_s dinov2_b
    convnextv2_tiny convnextv2_base
    resnet18 resnet50
)
DATASETS=(tcga-ut nct-crc-100k nct-crc-nonorm mhist)
N_DS=${#DATASETS[@]}

MODEL=${MODELS[$(( SLURM_ARRAY_TASK_ID / N_DS ))]}
DATASET=${DATASETS[$(( SLURM_ARRAY_TASK_ID % N_DS ))]}

echo "Task ${SLURM_ARRAY_TASK_ID}: linear full-data  model=$MODEL  dataset=$DATASET"
echo "Started: $(date)"

$PYTHON train_probe.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --head_type linear \
    --seed 0 \
    --train_cap 100000000

echo "Finished: $(date)"
