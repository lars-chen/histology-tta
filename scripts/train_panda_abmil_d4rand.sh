#!/bin/bash
#SBATCH --job-name=panda_mil_d4rand
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-9

# Array layout:
#   tasks 0-4  → mean_pool, folds 0-4
#   tasks 5-9  → abmil,     folds 0-4

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints/panda
RESULTS=$PROJECT/results/panda_tta.csv

mkdir -p "$CKPT_DIR" "$PROJECT/results" "$PROJECT/logs"
cd "$PROJECT" || exit 1

MODEL=${MODEL:-uni}
TASK=$SLURM_ARRAY_TASK_ID

if [ "$TASK" -lt 5 ]; then
    MIL=mean_pool
    FOLD=$TASK
else
    MIL=abmil
    FOLD=$((TASK - 5))
fi

RUN="${MIL}_${MODEL}_d4_rand_fold${FOLD}"

echo "========================================"
echo "Model: $MODEL | MIL: $MIL | train_tta: d4_rand | Fold: $FOLD | Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================"

echo "──── Train: $RUN ────"
$PYTHON -m panda.train_abmil \
    --model     "$MODEL" \
    --mil_type  "$MIL" \
    --fold      "$FOLD" \
    --train_tta d4_rand \
    --out_dir   "$CKPT_DIR" \
    --epochs    50 \
    --lr        1e-4

echo "──── Eval: $RUN ────"
$PYTHON -m panda.evaluate_abmil \
    --checkpoint "$CKPT_DIR/$RUN/best.pt" \
    --tta_eval   none d4_mean d4_bag d4_both \
    --out_csv    "$RESULTS"

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
