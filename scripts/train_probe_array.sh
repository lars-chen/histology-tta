#!/bin/bash
#SBATCH --job-name=train_probe
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/train_probe_%A_%a.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/train_probe_%A_%a.err
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-299

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python

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
SEEDS=(0 1 2 3 42)

N_DS=${#DATASETS[@]}
N_SEEDS=${#SEEDS[@]}

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / (N_DS * N_SEEDS) ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % (N_DS * N_SEEDS) ))
DS_IDX=$(( REMAINDER / N_SEEDS ))
SEED_IDX=$(( REMAINDER % N_SEEDS ))

MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DS_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: model=$MODEL  dataset=$DATASET  seed=$SEED"
echo "Node: $SLURMD_NODENAME  Started: $(date)"

# Torch heads (linear / mlp_1h / mlp_2h) are stochastic (random init + shuffle)
# → run all seeds for error bars. kNN is deterministic, so run it once (seed 0)
# to avoid wasted compute and fake zero-variance error bars.
STOCH_HEADS="linear mlp_1h mlp_2h"
DET_HEADS="knn_5 knn_10"

if [ "$SEED" -eq 0 ]; then
    HEADS="$STOCH_HEADS $DET_HEADS"
else
    HEADS="$STOCH_HEADS"
fi

for HEAD in $HEADS; do
    echo "--- head=$HEAD ---"
    $PYTHON train_probe.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --head_type "$HEAD" \
        --seed "$SEED"
done

echo "Finished: $(date)"
