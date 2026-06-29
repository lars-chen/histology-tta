#!/bin/bash
#SBATCH --job-name=logit_mean_pilot
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2:00:00

# ---------------------------------------------------------------------------
# Pilot: prob-space mean vs logit-space mean aggregation.
# Models: hoptimus (frozen), dinov2_b (frozen + finetuned where available).
# All available seeds per model/dataset combo — skips missing checkpoints.
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}

cd "$PROJECT" || exit 1

echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Pilot: mean vs logit_mean aggregation"
echo "Started: $(date)"
echo "========================================"

DATASETS=(tcga-ut)
SEEDS=(0 1 2 3 42)

run_eval() {
    local model=$1 dataset=$2 mode=$3 seed=$4
    local ckpt="$CKPT_DIR/${dataset}_${model}_${mode}_aug_seed${seed}_best.pt"
    if [[ ! -f "$ckpt" ]]; then
        echo "  [skip] missing: $ckpt"
        return
    fi
    echo ""
    echo "---- $model / $dataset / $mode / seed$seed ----"
    $PYTHON evaluate_tta.py \
        --model "$model" \
        --checkpoint "$ckpt" \
        --dataset "$dataset" \
        --tta_strategies none d4 d4_color \
        --aggregations mean logit_mean vote confidence \
        --batch_size 128 --amp --seed "$seed" \
        --cache_dir "$CACHE_DIR"
}

# hoptimus — frozen only (1.1B params, batch=4 as per train_hoptimus_cls.sh)
for DS in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        run_eval hoptimus "$DS" frozen "$SEED" 4
    done
done

# dinov2_b — frozen and finetuned
for DS in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        run_eval dinov2_b "$DS" frozen "$SEED" 128
        run_eval dinov2_b "$DS" finetuned "$SEED" 128
    done
done

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "========================================"
