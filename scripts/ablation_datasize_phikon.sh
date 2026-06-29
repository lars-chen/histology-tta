#!/bin/bash
#SBATCH --job-name=histo_ablation_datasize
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

# ---------------------------------------------------------------------------
# Dataset size ablation: Phikon (frozen) on NCT-CRC-100K
# Training subsets: 1K, 5K, 10K, 50K, full (~90K after val split)
# 3 seeds each, TTA eval after each training run
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}

MODEL=phikon
DATASET=tcga-ut
BATCH_SIZE=64
EPOCHS=20
PATIENCE=5
SEEDS=(42 0 1)
SUBSETS=(1000 5000 10000 50000)

mkdir -p "$CKPT_DIR" "$PROJECT/logs"
cd "$PROJECT" || exit 1

echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Ablation: $MODEL on $DATASET | Subsets: ${SUBSETS[*]} + full"
echo "Seeds: ${SEEDS[*]}"
echo "Started: $(date)"
echo "========================================"

run_one() {
    local SUBSET=$1
    local SEED=$2
    local SUBSET_ARG=""
    local SUBSET_TAG=""

    if [ "$SUBSET" != "full" ]; then
        SUBSET_ARG="--train_subset $SUBSET"
        SUBSET_TAG="_sub${SUBSET}"
    fi

    CKPT="$CKPT_DIR/${DATASET}_${MODEL}_frozen_aug${SUBSET_TAG}_seed${SEED}_best.pt"

    # Skip if checkpoint exists
    if [ -f "$CKPT" ]; then
        echo "---- SKIP training (exists): $(basename $CKPT) ----"
    else
        echo ""
        echo "-------- Training: $MODEL on $DATASET subset=$SUBSET seed=$SEED --------"
        $PYTHON train.py \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --freeze_backbone \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr 1e-3 \
            --patience "$PATIENCE" \
            --num_workers 4 \
            --amp \
            --seed "$SEED" \
            --cache_dir "$CACHE_DIR" \
            --checkpoint_dir "$CKPT_DIR" \
            $SUBSET_ARG
    fi

    echo ""
    echo "-------- TTA Eval: $MODEL subset=$SUBSET seed=$SEED --------"
    $PYTHON evaluate_tta.py \
        --model "$MODEL" \
        --checkpoint "$CKPT" \
        --dataset "$DATASET" \
        --tta_strategies none flips d4 \
        --aggregations mean vote confidence \
        --batch_size "$BATCH_SIZE" \
        --amp \
        --seed "$SEED" \
        --cache_dir "$CACHE_DIR"
}

# Run subsets
for SUBSET in "${SUBSETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        run_one "$SUBSET" "$SEED"
    done
done

# Run full (no subset flag) — skip training since seed42 already exists
for SEED in "${SEEDS[@]}"; do
    run_one "full" "$SEED"
done

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "========================================"
