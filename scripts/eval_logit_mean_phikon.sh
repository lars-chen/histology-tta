#!/bin/bash
#SBATCH --job-name=logit_mean_phikon
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}

cd "$PROJECT" || exit 1

for SEED in 0 1 2 3 42; do
    for MODEL in phikon phikon2; do
        echo "---- $MODEL frozen tcga-ut seed$SEED ----"
        $PYTHON evaluate_tta.py \
            --model $MODEL \
            --checkpoint "$CKPT_DIR/tcga-ut_${MODEL}_frozen_aug_seed${SEED}_best.pt" \
            --dataset tcga-ut \
            --tta_strategies none flips d4 \
            --aggregations mean logit_mean \
            --batch_size 64 \
            --amp \
            --seed $SEED \
            --cache_dir "$CACHE_DIR"
        echo ""
    done
done

echo "Done: $(date)"