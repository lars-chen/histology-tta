#!/bin/bash
#SBATCH --job-name=histo_fix_ablation
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# ---------------------------------------------------------------------------
# Priority order:
# 1. Restore corrupted full-dataset phikon results (seeds 0, 1, 42)
# 2. Re-eval existing ablation checkpoints with fixed _sub{N} naming
# 3. Train + eval remaining ablation subsets
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

cd "$PROJECT" || exit 1

echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Fix ablation naming + restore corrupted results"
echo "Started: $(date)"
echo "========================================"

# --- Part 1: Restore corrupted full-dataset results (HIGHEST PRIORITY) ---
echo ""
echo "=== Part 1: Restore full-dataset phikon tcga-ut (seeds 0, 1, 42) ==="

for SEED in 42 0 1; do
    CKPT="$CKPT_DIR/${DATASET}_${MODEL}_frozen_aug_seed${SEED}_best.pt"
    [ -f "$CKPT" ] || { echo "MISSING: $CKPT"; continue; }

    echo ""
    echo "---- Full-dataset restore: seed${SEED} ----"
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
done

# --- Part 2: Re-eval existing ablation checkpoints with fixed naming ---
echo ""
echo "=== Part 2: Re-eval existing ablation checkpoints ==="

for SUBSET in "${SUBSETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="$CKPT_DIR/${DATASET}_${MODEL}_frozen_aug_sub${SUBSET}_seed${SEED}_best.pt"
        [ -f "$CKPT" ] || continue

        echo ""
        echo "---- Ablation eval: sub${SUBSET} seed${SEED} ----"
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
    done
done

# --- Part 3: Train + eval remaining ablation subsets ---
echo ""
echo "=== Part 3: Train + eval remaining ablation subsets ==="

for SUBSET in "${SUBSETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="$CKPT_DIR/${DATASET}_${MODEL}_frozen_aug_sub${SUBSET}_seed${SEED}_best.pt"

        if [ -f "$CKPT" ]; then
            echo "---- SKIP training (exists): $(basename $CKPT) ----"
        else
            echo ""
            echo "---- Training: sub${SUBSET} seed${SEED} ----"
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
                --train_subset "$SUBSET"

            echo ""
            echo "---- Ablation eval: sub${SUBSET} seed${SEED} ----"
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
        fi
    done
done

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "========================================"
