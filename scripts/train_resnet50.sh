#!/bin/bash
#SBATCH --job-name=histo_resnet50
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=9:00:00

# ---------------------------------------------------------------------------
# Train ResNet-50 (frozen backbone / linear probe) on all available datasets,
# then run TTA evaluation on datasets that have a dedicated test split.
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
LOG_DIR=$PROJECT/logs
CACHE_DIR=${HF_CACHE_DIR:-$HOME/.cache/huggingface}

MODEL=resnet50
BATCH_SIZE=64
EPOCHS=20
DATASETS=(tcga-ut nct-crc-100k nct-crc-7k nct-crc-nonorm)

mkdir -p "$CKPT_DIR" "$LOG_DIR"
cd "$PROJECT" || exit 1

echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL  | Batch: $BATCH_SIZE  | Epochs: $EPOCHS"
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
        --freeze_backbone \
        --patience 1 \
        --num_workers 0 \
        --cache_dir "$CACHE_DIR" \
        --checkpoint_dir "$CKPT_DIR"

    if [ "$DATASET" = "tcga-ut" ]; then
        CKPT="$CKPT_DIR/${DATASET}_${MODEL}_frozen_best.pt"
        echo ""
        echo "-------- TTA Evaluation: $MODEL on $DATASET --------"
        $PYTHON evaluate_tta.py \
            --model "$MODEL" \
            --checkpoint "$CKPT" \
            --tta_strategies none flips d4 d4_color \
            --aggregations mean vote confidence \
            --batch_size "$BATCH_SIZE" \
            --cache_dir "$CACHE_DIR"
    fi
done

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "========================================"
