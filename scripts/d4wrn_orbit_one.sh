#!/bin/bash
#SBATCH --job-name=d4wrn_orb
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%A_%a.err
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --array=0-2
PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PY=$PROJECT/.venv/bin/python
export PYTHONPATH=$PROJECT:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CACHE=${HF_CACHE_DIR:-/gpfs/scratch/lpc8816/.cache/huggingface}
cd "$PROJECT" || exit 1
DATASETS=(tcga-ut nct-crc-100k nct-crc-nonorm)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}
CKPT="checkpoints/${DS}_d4wrn_finetuned_aug_seed3_best.pt"
[ -f "$CKPT" ] || CKPT="checkpoints/${DS}_d4wrn_finetuned_aug_seed2_best.pt"
echo "start $(date)  $DS  $CKPT"
$PY utils/analysis/eval_orbit_tightness.py --model d4wrn --checkpoint "$CKPT" \
    --dataset "$DS" --backbone_mode finetuned --seed 3 --batch_size 16 \
    --cache_dir "$CACHE" --output "results/orbit_tightness_d4wrn_${DS}.csv"
echo "end $(date)"
