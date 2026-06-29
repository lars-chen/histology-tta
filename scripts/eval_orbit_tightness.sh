#!/bin/bash
#SBATCH --job-name=orbit_tightness
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

# ---------------------------------------------------------------------------
# Evaluate embedding-space orbit tightness for all models × datasets × seeds.
#
# Produces: results/orbit_tightness.csv
# Optionally saves subsampled embeddings for UMAP (for 3 representative models
# on TCGA-UT, seed 42).
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}
OUTPUT=$PROJECT/results/orbit_tightness.csv
EMB_DIR=$PROJECT/embeddings

mkdir -p "$PROJECT/logs" "$EMB_DIR"
cd "$PROJECT" || exit 1

# Remove stale output to start fresh (re-run appends; this ensures clean start)
rm -f "$OUTPUT"

echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================"

SEEDS_FINETUNED=(0 1 2 3 42)
SEEDS_FROZEN=(42)
BATCH=64

# ---------------------------------------------------------------------------
# Helper: run orbit eval for one (dataset, model, backbone_mode) combo
# ---------------------------------------------------------------------------
run_orbit() {
    local DATASET=$1
    local MODEL=$2
    local BACKBONE=$3   # frozen | finetuned
    local SAVE_EMB=$4   # "" or path prefix for saving embeddings (UMAP)

    # Frozen backbone: weights are seed-independent, one seed suffices
    local SEEDS
    if [ "$BACKBONE" = "frozen" ]; then
        SEEDS=("${SEEDS_FROZEN[@]}")
    else
        SEEDS=("${SEEDS_FINETUNED[@]}")
    fi

    for SEED in "${SEEDS[@]}"; do
        CKPT="$CKPT_DIR/${DATASET}_${MODEL}_${BACKBONE}_aug_seed${SEED}_best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "  [SKIP] missing checkpoint: $CKPT"
            continue
        fi
        EMB_ARG=""
        if [ -n "$SAVE_EMB" ] && [ "$SEED" -eq 42 ]; then
            # Save embeddings only for seed 42 (for UMAP)
            EMB_ARG="--save_embeddings ${EMB_DIR}/${SAVE_EMB}"
        fi
        echo "  $MODEL / $DATASET / $BACKBONE / seed $SEED"
        $PYTHON eval_orbit_tightness.py \
            --model "$MODEL" \
            --checkpoint "$CKPT" \
            --dataset "$DATASET" \
            --backbone_mode "$BACKBONE" \
            --seed "$SEED" \
            --batch_size "$BATCH" \
            --num_workers 4 \
            --cache_dir "$CACHE_DIR" \
            --output "$OUTPUT" \
            --append \
            $EMB_ARG
    done
}

# ---------------------------------------------------------------------------
# Histology foundation models — frozen backbone only
# ---------------------------------------------------------------------------
HISTO_MODELS=(phikon phikon2 uni uni2 virchow virchow2 gigapath hoptimus)
MAIN_DATASETS=(tcga-ut nct-crc-100k nct-crc-nonorm mhist)

echo ""
echo "--- Histology FMs (frozen) ---"
for MODEL in "${HISTO_MODELS[@]}"; do
    for DATASET in "${MAIN_DATASETS[@]}"; do
        # Save UMAP embeddings for all histology FMs on tcga-ut seed 42
        SAVE_EMB=""
        if [ "$DATASET" = "tcga-ut" ]; then
            SAVE_EMB="tcga-ut_${MODEL}_frozen_seed42"
        fi
        run_orbit "$DATASET" "$MODEL" "frozen" "$SAVE_EMB"
    done
done

# ---------------------------------------------------------------------------
# General-purpose models — frozen and finetuned
# ---------------------------------------------------------------------------
GENERAL_MODELS=(convnextv2_tiny convnextv2_base dinov2_s dinov2_b)
GP_DATASETS=(tcga-ut nct-crc-100k nct-crc-nonorm mhist)

echo ""
echo "--- General-purpose models (frozen + finetuned) ---"
for MODEL in "${GENERAL_MODELS[@]}"; do
    for DATASET in "${GP_DATASETS[@]}"; do
        # Save UMAP embeddings for all general models on tcga-ut seed 42 (frozen)
        SAVE_EMB=""
        if [ "$DATASET" = "tcga-ut" ]; then
            SAVE_EMB="tcga-ut_${MODEL}_frozen_seed42"
        fi
        run_orbit "$DATASET" "$MODEL" "frozen" "$SAVE_EMB"

        run_orbit "$DATASET" "$MODEL" "finetuned" ""
    done
done

# ---------------------------------------------------------------------------
# D4WRN — finetuned only
# ---------------------------------------------------------------------------
D4WRN_DATASETS=(tcga-ut nct-crc-100k nct-crc-nonorm)

echo ""
echo "--- D4WRN (finetuned) ---"
for DATASET in "${D4WRN_DATASETS[@]}"; do
    # Save UMAP embeddings for tcga-ut (representative equivariant model)
    SAVE_EMB=""
    if [ "$DATASET" = "tcga-ut" ]; then
        SAVE_EMB="tcga-ut_d4wrn_finetuned_seed42"
    fi
    run_orbit "$DATASET" "d4wrn" "finetuned" "$SAVE_EMB"
done

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "Results: $OUTPUT"
echo "Embeddings: $EMB_DIR"
echo "========================================"
