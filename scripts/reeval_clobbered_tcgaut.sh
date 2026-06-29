#!/bin/bash
#SBATCH --job-name=reeval_tcgaut
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}
DATASET=tcga-ut

cd "$PROJECT" || exit 1

declare -A BATCH_FOR
BATCH_FOR[virchow]=16
BATCH_FOR[virchow2]=16
BATCH_FOR[uni]=32
BATCH_FOR[resnet18]=128
BATCH_FOR[resnet50]=128

for MODEL in virchow virchow2 uni resnet18 resnet50; do
    BATCH=${BATCH_FOR[$MODEL]:-32}
    for SEED in 0 1 2 3 42; do
        CKPT="$CKPT_DIR/${DATASET}_${MODEL}_frozen_aug_seed${SEED}_best.pt"
        [[ -f "$CKPT" ]] || { echo "SKIP $MODEL seed$SEED (no checkpoint)"; continue; }
        echo "---- $MODEL seed$SEED (batch=$BATCH) ----"
        $PYTHON evaluate_tta.py \
            --model $MODEL \
            --checkpoint "$CKPT" \
            --dataset $DATASET \
            --tta_strategies none flips d4 \
            --aggregations mean logit_mean \
            --batch_size $BATCH \
            --amp \
            --seed $SEED \
            --cache_dir "$CACHE_DIR"
        echo ""
    done
done

echo "Done: $(date)"
