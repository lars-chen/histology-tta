#!/bin/bash
#SBATCH --job-name=histo_mhist_eval
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

# ---------------------------------------------------------------------------
# TTA evaluation only for MHIST (checkpoints already exist from training).
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}

DATASET=mhist
SEEDS=(42 0 1 2 3)

MODELS=(
    "phikon      32"
    "phikon2     32"
    "uni         32"
    "uni2        16"
    "virchow     16"
    "virchow2    16"
    "gigapath     4"
    "hoptimus     4"
    "convnextv2_tiny 64"
    "dinov2_s    64"
)

cd "$PROJECT" || exit 1

echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "MHIST TTA Evaluation Only"
echo "Seeds: ${SEEDS[*]}"
echo "Started: $(date)"
echo "========================================"

for MODEL_CFG in "${MODELS[@]}"; do
    read -r MODEL EVAL_BATCH_SIZE <<< "$MODEL_CFG"

    for SEED in "${SEEDS[@]}"; do
        CKPT="$CKPT_DIR/${DATASET}_${MODEL}_frozen_aug_seed${SEED}_best.pt"

        if [ ! -f "$CKPT" ]; then
            echo "SKIP: $CKPT not found"
            continue
        fi

        echo ""
        echo "-------- TTA Evaluation: $MODEL on $DATASET (seed $SEED) --------"
        $PYTHON evaluate_tta.py \
            --model "$MODEL" \
            --checkpoint "$CKPT" \
            --dataset "$DATASET" \
            --tta_strategies none flips d4 d4_color \
            --aggregations mean vote confidence \
            --batch_size "$EVAL_BATCH_SIZE" \
            --amp \
            --seed "$SEED" \
            --cache_dir "$CACHE_DIR"
    done
done

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "========================================"
