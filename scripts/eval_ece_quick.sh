#!/bin/bash
#SBATCH --job-name=histo_ece_quick
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}

cd "$PROJECT" || exit 1

echo "---- phikon2 frozen tcga-ut seed42 ----"
$PYTHON evaluate_tta.py \
    --model phikon2 \
    --checkpoint "$CKPT_DIR/tcga-ut_phikon2_frozen_aug_seed42_best.pt" \
    --dataset tcga-ut \
    --tta_strategies none flips d4 \
    --aggregations mean \
    --batch_size 64 \
    --amp \
    --seed 42 \
    --cache_dir "$CACHE_DIR"

echo ""
echo "---- dinov2_s finetuned tcga-ut seed42 ----"
$PYTHON evaluate_tta.py \
    --model dinov2_s \
    --checkpoint "$CKPT_DIR/tcga-ut_dinov2_s_finetuned_aug_seed42_best.pt" \
    --dataset tcga-ut \
    --tta_strategies none flips d4 \
    --aggregations mean \
    --batch_size 64 \
    --amp \
    --seed 42 \
    --cache_dir "$CACHE_DIR"

echo "Done: $(date)"
