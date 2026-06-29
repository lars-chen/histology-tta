#!/bin/bash
#SBATCH --job-name=d4wrn_fill
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-10

# ---------------------------------------------------------------------------
# Fill missing D4WRN finetuned runs — one (dataset, seed) per array task so all
# 11 run concurrently instead of sequentially.
#   task 0-1  : tcga-ut        seeds 2, 3
#   task 2-4  : nct-crc-100k   seeds 1, 2, 3
#   task 5-7  : nct-crc-nonorm seeds 1, 2, 3
#   task 8-10 : mhist          seeds 1, 2, 3
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}

MODEL=d4wrn
BATCH_SIZE=64
EVAL_BATCH_SIZE=32
EPOCHS=30
PATIENCE=5

mkdir -p "$CKPT_DIR" "$PROJECT/logs"
cd "$PROJECT" || exit 1

# (dataset, seed) per array index
DATASETS=(tcga-ut tcga-ut nct-crc-100k nct-crc-100k nct-crc-100k \
          nct-crc-nonorm nct-crc-nonorm nct-crc-nonorm mhist mhist mhist)
SEEDS=(2 3 1 2 3 1 2 3 1 2 3)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
CKPT="$CKPT_DIR/${DATASET}_${MODEL}_finetuned_aug_seed${SEED}_best.pt"

echo "========================================"
echo "Job: $SLURM_JOB_ID  Task: $SLURM_ARRAY_TASK_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL | Dataset: $DATASET | Seed: $SEED"
echo "Started: $(date)"
echo "========================================"

if [ -f "$CKPT" ]; then
    echo "---- SKIP: $CKPT already exists ----"
    exit 0
fi

echo "-------- Training: $MODEL on $DATASET (seed $SEED) --------"
$PYTHON train.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr 1e-3 \
    --patience "$PATIENCE" \
    --num_workers 4 \
    --amp \
    --seed "$SEED" \
    --cache_dir "$CACHE_DIR" \
    --checkpoint_dir "$CKPT_DIR"

echo "-------- TTA Evaluation: $MODEL on $DATASET (seed $SEED) --------"
$PYTHON evaluate_tta.py \
    --model "$MODEL" \
    --checkpoint "$CKPT" \
    --dataset "$DATASET" \
    --tta_strategies none flips d4 d4_color \
    --aggregations mean vote confidence \
    --batch_size "$EVAL_BATCH_SIZE" \
    --amp \
    --seed "$SEED" \
    --cache_dir "$CACHE_DIR"

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
