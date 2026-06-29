#!/bin/bash
#SBATCH --job-name=cam17_mil_d4rand
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-4

# ---------------------------------------------------------------------------
# MIL experiment: MeanPool and ABMIL trained with d4_rand view augmentation.
#
# Parameterize by model via env var (defaults to uni):
#   MODEL=phikon2 sbatch scripts/train_camelyon17_abmil_d4rand.sh
#
# Features expected at: camelyon17_features/<MODEL>/d4_all/
# Results appended to:  results/camelyon17_tta.csv
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python

MODEL=${MODEL:-uni}
FEATURES=$PROJECT/camelyon17_features/$MODEL/d4_all
CKPT_DIR=$PROJECT/checkpoints/camelyon17
RESULTS=$PROJECT/results/camelyon17_tta.csv

mkdir -p "$CKPT_DIR" "$PROJECT/results" "$PROJECT/logs"
cd "$PROJECT" || exit 1

SEED=$SLURM_ARRAY_TASK_ID

echo "========================================"
echo "Model: $MODEL | train_tta: d4_rand | Job: $SLURM_ARRAY_JOB_ID  Task: $SEED  Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================"

run_condition() {
    local MIL=$1
    local RUN="${MIL}_${MODEL}_d4_rand_seed${SEED}"

    echo ""
    echo "──── Train: $RUN ────"
    $PYTHON -m camelyon17.train_abmil \
        --features_dir  "$FEATURES" \
        --model         "$MODEL" \
        --mil_type      "$MIL" \
        --train_tta     d4_rand \
        --out_dir       "$CKPT_DIR" \
        --epochs        50 \
        --lr            1e-4 \
        --seed          "$SEED"

    echo ""
    echo "──── Eval: $RUN ────"
    $PYTHON -m camelyon17.evaluate_abmil \
        --checkpoint    "$CKPT_DIR/$RUN/best.pt" \
        --features_dir  "$FEATURES" \
        --tta_eval      none d4_mean d4_bag d4_both \
        --out_csv       "$RESULTS" \
        --seed          "$SEED"
}

run_condition mean_pool
run_condition abmil

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "Results: $RESULTS"
echo "========================================"
