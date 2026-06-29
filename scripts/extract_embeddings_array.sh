#!/bin/bash
#SBATCH --job-name=emb_extract
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/emb_extract_%A_%a.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/emb_extract_%A_%a.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --array=0-59

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CACHE_DIR=/gpfs/scratch/lpc8816/.cache/huggingface

export TORCH_HOME=/gpfs/scratch/lpc8816/.cache/torch
export HF_HOME=/gpfs/scratch/lpc8816/.cache/huggingface
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
source ~/.bashrc 2>/dev/null || true

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

N_DATASETS=${#DATASETS[@]}
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / N_DATASETS ))
DATASET_IDX=$(( SLURM_ARRAY_TASK_ID % N_DATASETS ))

MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: model=$MODEL  dataset=$DATASET"
echo "Node: $SLURMD_NODENAME  Started: $(date)"

$PYTHON extract_embeddings.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --amp \
    --num_workers 4 \
    --cache_dir "$CACHE_DIR"

echo "Finished: $(date)"
