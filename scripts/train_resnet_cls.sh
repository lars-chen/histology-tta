#!/bin/bash
#SBATCH --job-name=resnet_cls
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-9

# Array layout: 5 seeds × 2 models
# index 0-4  → resnet18, seeds 0 1 2 3 42
# index 5-9  → resnet50, seeds 0 1 2 3 42

SEEDS=(0 1 2 3 42)
MODELS=(resnet18 resnet50)

MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 5))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 5))]}

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}
DATASETS=(mhist tcga-ut nct-crc-100k nct-crc-nonorm)

cd "$PROJECT" || exit 1
echo "Started: $(date)  model=$MODEL  seed=$SEED"

for DATASET in "${DATASETS[@]}"; do
    CKPT="$CKPT_DIR/${DATASET}_${MODEL}_frozen_aug_seed${SEED}_best.pt"
    if [ -f "$CKPT" ]; then
        echo "  skip training $MODEL $DATASET seed$SEED (checkpoint exists)"
    else
        echo "  train $MODEL on $DATASET seed$SEED"
        $PYTHON train.py \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --epochs 20 \
            --batch_size 64 \
            --freeze_backbone \
            --seed "$SEED" \
            --patience 2 \
            --num_workers 0 \
            --cache_dir "$CACHE_DIR" \
            --checkpoint_dir "$CKPT_DIR"
    fi

    echo "  eval $MODEL on $DATASET seed$SEED"
    $PYTHON evaluate_tta.py \
        --model "$MODEL" \
        --checkpoint "$CKPT" \
        --dataset "$DATASET" \
        --tta_strategies none flips d4 d4_color \
        --aggregations mean vote confidence \
        --batch_size 128 \
        --seed "$SEED" \
        --amp \
        --cache_dir "$CACHE_DIR"
done

echo "Finished: $(date)"
