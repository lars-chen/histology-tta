#!/bin/bash
#SBATCH --job-name=histo_eval_ece
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# ---------------------------------------------------------------------------
# Re-run TTA eval on all existing frozen checkpoints to capture ECE metric.
# Overwrites existing JSON results with ECE-enriched versions.
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}

cd "$PROJECT" || exit 1

echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Re-eval all frozen checkpoints for ECE"
echo "Started: $(date)"
echo "========================================"

MODELS=(gigapath hoptimus phikon phikon2 uni uni2 virchow virchow2 convnextv2_tiny dinov2_s)
DATASETS=(tcga-ut nct-crc-100k nct-crc-nonorm mhist)
SEEDS=(42 0 1 2 3)

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            CKPT="$CKPT_DIR/${DATASET}_${MODEL}_frozen_aug_seed${SEED}_best.pt"
            [ -f "$CKPT" ] || continue

            echo ""
            echo "---- $MODEL / $DATASET / seed$SEED ----"
            $PYTHON evaluate_tta.py \
                --model "$MODEL" \
                --checkpoint "$CKPT" \
                --dataset "$DATASET" \
                --tta_strategies none flips d4 \
                --aggregations mean vote confidence \
                --batch_size 64 \
                --amp \
                --seed "$SEED" \
                --cache_dir "$CACHE_DIR"
        done
    done
done

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "========================================"
