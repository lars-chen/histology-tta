#!/bin/bash
#SBATCH --job-name=histo_ft
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00

# ---------------------------------------------------------------------------
# Full fine-tune a model on all datasets (seeds; skips training when a
# matching checkpoint already exists), then TTA-evaluate.
#
# Usage: sbatch scripts/train_ft.sh <model> [--scratch] [--noaug] \
#            [--seeds "0 1 2 3 42"] [--datasets "tcga-ut nct-crc-100k ..."]
#
# d4wrn wants `sbatch --partition=a100_short --gres=gpu:a100:1 --mem=128G`.
# ---------------------------------------------------------------------------

set -euo pipefail

MODEL=$1; shift
SCRATCH=0
NO_AUGMENT=0
SEEDS_OVERRIDE=""
DATASETS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scratch) SCRATCH=1; shift ;;
        --noaug) NO_AUGMENT=1; shift ;;
        --seeds) SEEDS_OVERRIDE=$2; shift 2 ;;
        --datasets) DATASETS_OVERRIDE=$2; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
LOG_DIR=$PROJECT/logs
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}

# Per-model defaults: batch, eval_batch, epochs, workers, patience, lr.
case "$MODEL" in
    resnet18)        BATCH=64 EVAL_BATCH=128 EPOCHS=30 WORKERS=4 PATIENCE=5 LR=1e-4 ;;
    resnet50)        BATCH=64 EVAL_BATCH=128 EPOCHS=30 WORKERS=4 PATIENCE=5 LR=1e-4 ;;
    convnextv2_tiny) BATCH=64 EVAL_BATCH=32  EPOCHS=30 WORKERS=4 PATIENCE=5 LR=5e-5 ;;
    convnextv2_base) BATCH=16 EVAL_BATCH=32  EPOCHS=20 WORKERS=2 PATIENCE=2 LR=1e-5 ;;
    dinov2_s)        BATCH=64 EVAL_BATCH=64  EPOCHS=30 WORKERS=4 PATIENCE=5 LR=1e-5
                      export TORCH_HOME=/gpfs/scratch/lpc8816/.cache/torch ;;
    dinov2_b)        BATCH=16 EVAL_BATCH=32  EPOCHS=30 WORKERS=4 PATIENCE=5 LR=1e-5
                      export TORCH_HOME=/gpfs/scratch/lpc8816/.cache/torch ;;
    d4wrn)           BATCH=64 EVAL_BATCH=32  EPOCHS=30 WORKERS=4 PATIENCE=5 LR=1e-3 ;;
    *) echo "No config for model '$MODEL' — add one to scripts/train_ft.sh" >&2; exit 1 ;;
esac

if [[ $SCRATCH -eq 1 ]]; then
    [[ "$MODEL" != resnet18 && "$MODEL" != resnet50 ]] && echo "Note: --scratch is untested outside resnet18/50" >&2
    LR=1e-3
fi

SEEDS=(${SEEDS_OVERRIDE:-0 1 2 3 42})
DATASETS=(${DATASETS_OVERRIDE:-tcga-ut nct-crc-100k nct-crc-nonorm mhist})

EXTRA_ARGS=()
TAG="finetuned_aug"
if [[ $SCRATCH -eq 1 ]]; then
    EXTRA_ARGS+=(--no_pretrained)
    TAG="finetuned_scratch_aug"
fi
if [[ $NO_AUGMENT -eq 1 ]]; then
    EXTRA_ARGS+=(--no_augment)
    TAG="${TAG/_aug/_noaug}"
fi

mkdir -p "$CKPT_DIR" "$LOG_DIR"
cd "$PROJECT"

echo "========================================"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Model: $MODEL (fine-tune)  | Batch: $BATCH  | Epochs: $EPOCHS  | LR: $LR"
echo "Seeds: ${SEEDS[*]}  | Datasets: ${DATASETS[*]}"
echo "Started: $(date)"
echo "========================================"

for SEED in "${SEEDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        CKPT="$CKPT_DIR/${DATASET}_${MODEL}_${TAG}_seed${SEED}_best.pt"

        if [[ -f "$CKPT" ]]; then
            echo "-------- SKIP training: $MODEL on $DATASET (seed $SEED, checkpoint exists) --------"
        else
            echo ""
            echo "-------- Training: $MODEL on $DATASET (seed $SEED) --------"
            $PYTHON train.py \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH" \
                --lr "$LR" \
                --patience "$PATIENCE" \
                --num_workers "$WORKERS" \
                --amp \
                --seed "$SEED" \
                --cache_dir "$CACHE_DIR" \
                --checkpoint_dir "$CKPT_DIR" \
                "${EXTRA_ARGS[@]}"
        fi

        echo ""
        echo "-------- TTA Evaluation: $MODEL on $DATASET (seed $SEED) --------"
        $PYTHON evaluate_tta.py \
            --model "$MODEL" \
            --checkpoint "$CKPT" \
            --dataset "$DATASET" \
            --tta_strategies none flips d4 d4_color \
            --aggregations mean vote confidence \
            --batch_size "$EVAL_BATCH" \
            --amp \
            --seed "$SEED" \
            --cache_dir "$CACHE_DIR"
    done
done

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "========================================"
