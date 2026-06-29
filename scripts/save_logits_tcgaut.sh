#!/bin/bash
#SBATCH --job-name=save_logits_tcgaut
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=20:00:00

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
CKPT_DIR=$PROJECT/checkpoints
LOGIT_DIR=$PROJECT/logits
CACHE_DIR=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}
DATASET=tcga-ut

# Safe batch sizes per model (actual GPU batch = batch_size × n_views)
declare -A BATCH_FOR
BATCH_FOR[gigapath]=4
BATCH_FOR[hoptimus]=4
BATCH_FOR[virchow]=16
BATCH_FOR[virchow2]=16
BATCH_FOR[uni2]=16
BATCH_FOR[phikon]=32
BATCH_FOR[phikon2]=32
BATCH_FOR[uni]=32
BATCH_FOR[dinov2_b]=64
BATCH_FOR[dinov2_s]=64
BATCH_FOR[convnextv2_base]=64
BATCH_FOR[convnextv2_tiny]=64

cd "$PROJECT" || exit 1

shopt -s nullglob
for CKPT_FILE in "$CKPT_DIR"/${DATASET}_*_best.pt; do
    [[ -f "$CKPT_FILE" ]] || continue
    [[ "$CKPT_FILE" == *d4wrn* ]] && continue
    [[ "$CKPT_FILE" == *_sub* ]] && continue

    STEM=$(basename "$CKPT_FILE" .pt)
    LOGIT_FILE="$LOGIT_DIR/$DATASET/${STEM}.pt"
    if [[ -f "$LOGIT_FILE" ]]; then
        echo "SKIP $STEM (logit file exists)"
        continue
    fi

    read -r MODEL SEED < <(python3 - <<EOF
import re
stem = "$STEM"
dataset = "tcga-ut"
m = re.match(r'^' + re.escape(dataset) + r'_(.+?)_(frozen|finetuned)_(?:no)?aug_seed(\d+)_best$', stem)
if m:
    print(m.group(1), m.group(3))
EOF
)
    [[ -z "$MODEL" ]] && { echo "WARN: could not parse $STEM"; continue; }

    BATCH=${BATCH_FOR[$MODEL]:-32}
    echo "---- $MODEL $DATASET seed$SEED (batch=$BATCH) ----"
    $PYTHON evaluate_tta.py \
        --model "$MODEL" \
        --checkpoint "$CKPT_FILE" \
        --dataset "$DATASET" \
        --tta_strategies none flips d4 \
        --aggregations mean logit_mean \
        --save_logits \
        --no_save_results \
        --batch_size "$BATCH" \
        --amp \
        --seed "$SEED" \
        --cache_dir "$CACHE_DIR"
    echo ""
done

echo "Done: $(date)"
