#!/bin/bash
#SBATCH --job-name=panda_patch
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.err
#SBATCH --partition=cpu_medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-31

PANDA=/gpfs/scratch/lpc8816/panda
CLAM=/gpfs/data/mankowskilab/chen/CLAM
PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$CLAM/.venv/bin/python
PYTHON_PROJECT=$PROJECT/.venv/bin/python

SLIDES=$PANDA/train_images
SAVE=$PANDA/clam_patches

mkdir -p "$SAVE/patches" "$SAVE/masks" "$SAVE/stitches" "$SAVE/process_lists"

# Build per-task process list (shard 10616 slides across 32 tasks)
TASK=$SLURM_ARRAY_TASK_ID
N_TASKS=32
PROCESS_LIST=$SAVE/process_lists/task_${TASK}.csv

$PYTHON_PROJECT - <<EOF
import pandas as pd

df = pd.read_csv("$PANDA/train.csv")
# CLAM expects full filename with extension in slide_id column
df = df[["image_id"]].rename(columns={"image_id": "slide_id"})
df["slide_id"] = df["slide_id"] + ".tiff"
shard = df.iloc[$TASK::$N_TASKS].reset_index(drop=True)
shard.to_csv("$PROCESS_LIST", index=False)
print(f"Task $TASK: {len(shard)} slides")
EOF

echo "========================================"
echo "Task: $TASK/$N_TASKS  Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================"

cd "$CLAM" || exit 1

$PYTHON create_patches_fp.py \
    --source        "$SLIDES" \
    --save_dir      "$SAVE" \
    --patch_size    256 \
    --step_size     256 \
    --patch_level   0 \
    --preset        bwh_biopsy.csv \
    --seg \
    --patch \
    --stitch \
    --process_list  "$PROCESS_LIST"

echo "Finished: $(date)"
