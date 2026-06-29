#!/bin/bash
#SBATCH --job-name=selective_tta_logits
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
LOGITS_DIR=$PROJECT/logits
OUT_DIR=$PROJECT/results/raw

cd "$PROJECT" || exit 1

echo "===== entropy-based selective TTA from stored logits ====="
echo "Start: $(date)"
echo ""

for DATASET in mhist nct-crc-100k nct-crc-nonorm tcga-ut; do
    DIR="$LOGITS_DIR/$DATASET"
    [[ -d "$DIR" ]] || { echo "SKIP: $DIR not found"; continue; }

    N=$(ls "$DIR"/*.pt 2>/dev/null | wc -l)
    echo "--- $DATASET ($N files) ---"

    $PYTHON eval_selective_from_logits.py \
        --logits_dir "$DIR" \
        --out_dir "$OUT_DIR"

    echo ""
done

echo "Done: $(date)"
