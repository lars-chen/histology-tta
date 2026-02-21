#!/bin/bash
#SBATCH --job-name=histo_convnextv2_l_ft
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00

# ---------------------------------------------------------------------------
# Full fine-tuning of ConvNeXt V2-Large (198M params) on all available
# datasets, then TTA evaluation on datasets with a dedicated test split.
#
# Weights are downloaded from timm (Hugging Face) on first run.
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
LOG_DIR=$PROJECT/logs
CACHE_DIR=${HF_CACHE_DIR:-$HOME/.cache/huggingface}

MODEL=convnextv2_large
BATCH_SIZE=8
EPOCHS=20
DATASETS=(tcga-ut nct-crc-100k nct-crc-nonorm)

mkdir -p "$CKPT_DIR" "$LOG_DIR"
cd "$PROJECT" || exit 1

echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL (fine-tune)  | Batch: $BATCH_SIZE  | Epochs: $EPOCHS"
echo "Started: $(date)"
echo "========================================"

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "-------- Training: $MODEL on $DATASET --------"

    $PYTHON train.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr 1e-5 \
        --patience 2 \
        --num_workers 2 \
        --amp \
        --cache_dir "$CACHE_DIR" \
        --checkpoint_dir "$CKPT_DIR"

    CKPT="$CKPT_DIR/${DATASET}_${MODEL}_finetuned_best.pt"
    echo ""
    echo "-------- TTA Evaluation: $MODEL on $DATASET --------"
    $PYTHON evaluate_tta.py \
        --model "$MODEL" \
        --checkpoint "$CKPT" \
        --dataset "$DATASET" \
        --tta_strategies none flips d4 d4_color \
        --aggregations mean vote confidence \
        --batch_size "$BATCH_SIZE" \
        --cache_dir "$CACHE_DIR"
done

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "========================================"
