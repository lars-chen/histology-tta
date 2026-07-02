#!/bin/bash
#SBATCH --job-name=histo_cls
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

# ---------------------------------------------------------------------------
# Train a frozen-backbone linear/MLP probe on all datasets (seeds and skip
# training when a matching checkpoint already exists), then TTA-evaluate.
#
# Usage: sbatch scripts/train_cls.sh <model> [--mlp] [--noaug] \
#            [--seeds "0 1 2 3 42"] [--datasets "tcga-ut nct-crc-100k ..."]
#
# Gated models (uni, uni2, virchow, virchow2, gigapath, hoptimus) need
# HF_TOKEN set in the submission environment. Large FMs generally want
# `sbatch --partition=a100_short --mem=128G scripts/train_cls.sh <model>`.
# ---------------------------------------------------------------------------

set -euo pipefail

MODEL=$1; shift
USE_MLP=0
NO_AUGMENT=0
SEEDS_OVERRIDE=""
DATASETS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mlp) USE_MLP=1; shift ;;
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

# Per-model defaults: batch, eval_batch, workers, patience, mlp_hidden, num_workers.
# mlp_hidden of 0 means "no MLP head defined for this model".
case "$MODEL" in
    resnet18)         BATCH=64 EVAL_BATCH=128 WORKERS=0 PATIENCE=2 MLP_HIDDEN=0 ;;
    resnet50)         BATCH=64 EVAL_BATCH=64  WORKERS=0 PATIENCE=2 MLP_HIDDEN=0 ;;
    convnextv2_tiny)  BATCH=64 EVAL_BATCH=32  WORKERS=4 PATIENCE=5 MLP_HIDDEN=768 ;;
    convnextv2_base)  BATCH=32 EVAL_BATCH=32  WORKERS=4 PATIENCE=2 MLP_HIDDEN=1024
                       [[ $USE_MLP -eq 1 ]] && { BATCH=16; EVAL_BATCH=16; } ;;
    dinov2_s)         BATCH=64 EVAL_BATCH=64  WORKERS=2 PATIENCE=5 MLP_HIDDEN=384
                       export TORCH_HOME=/gpfs/scratch/lpc8816/.cache/torch ;;
    dinov2_b)         BATCH=32 EVAL_BATCH=32  WORKERS=4 PATIENCE=5 MLP_HIDDEN=768
                       export TORCH_HOME=/gpfs/scratch/lpc8816/.cache/torch ;;
    ctranspath)       BATCH=64 EVAL_BATCH=64  WORKERS=4 PATIENCE=5 MLP_HIDDEN=0 ;;
    phikon)           BATCH=32 EVAL_BATCH=32  WORKERS=4 PATIENCE=5 MLP_HIDDEN=0 ;;
    phikon2)          BATCH=32 EVAL_BATCH=16  WORKERS=2 PATIENCE=5 MLP_HIDDEN=0 ;;
    uni)              BATCH=16 EVAL_BATCH=16  WORKERS=4 PATIENCE=5 MLP_HIDDEN=1024 ;;  # gated
    uni2)             BATCH=8  EVAL_BATCH=8   WORKERS=4 PATIENCE=5 MLP_HIDDEN=0 ;;      # gated
    virchow)          BATCH=8  EVAL_BATCH=8   WORKERS=4 PATIENCE=5 MLP_HIDDEN=0 ;;      # gated
    virchow2)         BATCH=8  EVAL_BATCH=8   WORKERS=4 PATIENCE=5 MLP_HIDDEN=0 ;;      # gated
    gigapath)         BATCH=8  EVAL_BATCH=4   WORKERS=4 PATIENCE=5 MLP_HIDDEN=0 ;;      # gated
    hoptimus)         BATCH=8  EVAL_BATCH=4   WORKERS=4 PATIENCE=5 MLP_HIDDEN=0 ;;      # gated
    *) echo "No config for model '$MODEL' — add one to scripts/train_cls.sh" >&2; exit 1 ;;
esac

SEEDS=(${SEEDS_OVERRIDE:-0 1 2 3 42})
DATASETS=(${DATASETS_OVERRIDE:-tcga-ut nct-crc-100k nct-crc-nonorm mhist})
EPOCHS=20

EXTRA_ARGS=()
TAG="frozen"
if [[ $USE_MLP -eq 1 ]]; then
    [[ $MLP_HIDDEN -eq 0 ]] && { echo "Model '$MODEL' has no MLP config" >&2; exit 1; }
    EXTRA_ARGS+=(--mlp_hidden "$MLP_HIDDEN")
    TAG="frozen_aug_mlp${MLP_HIDDEN}"
fi
AUG_TAG="aug"
if [[ $NO_AUGMENT -eq 1 ]]; then
    EXTRA_ARGS+=(--no_augment)
    AUG_TAG="noaug"
fi
[[ $USE_MLP -eq 0 ]] && TAG="frozen_${AUG_TAG}"
[[ $USE_MLP -eq 1 && $NO_AUGMENT -eq 1 ]] && TAG="frozen_noaug_mlp${MLP_HIDDEN}"

mkdir -p "$CKPT_DIR" "$LOG_DIR"
cd "$PROJECT"

echo "========================================"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Model: $MODEL  | Batch: $BATCH  | Epochs: $EPOCHS  | MLP: $USE_MLP  | NoAug: $NO_AUGMENT"
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
                --freeze_backbone \
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
