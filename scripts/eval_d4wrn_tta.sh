#!/bin/bash
#SBATCH --job-name=eval_d4wrn_tta
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# ---------------------------------------------------------------------------
# TTA evaluation for D4-equivariant Wide ResNet checkpoint (tcga-ut, aug)
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}

MODEL=d4wrn
EVAL_BATCH_SIZE=32
DATASET=tcga-ut
CKPT="$CKPT_DIR/${DATASET}_${MODEL}_finetuned_aug_seed42_best.pt"

cd "$PROJECT" || exit 1

echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "TTA Eval: $MODEL on $DATASET"
echo "Checkpoint: $CKPT"
echo "Started: $(date)"
echo "========================================"

$PYTHON evaluate_tta.py \
    --model "$MODEL" \
    --checkpoint "$CKPT" \
    --dataset "$DATASET" \
    --tta_strategies none flips d4 d4_color \
    --aggregations mean vote confidence \
    --batch_size "$EVAL_BATCH_SIZE" \
    --amp \
    --cache_dir "$CACHE_DIR"

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "========================================"
