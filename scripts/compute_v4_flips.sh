#!/bin/bash
#SBATCH --job-name=v4_flips
#SBATCH --output=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.out
#SBATCH --error=/gpfs/data/mankowskilab/chen/histology-tta/logs/%x_%j.err
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# ---------------------------------------------------------------------------
# V4 (Klein four-group, flips) re-aggregation of cached 8-view embeddings.
# Pure inference: loads saved linear-probe checkpoints, runs the V4 view
# subset {0,2,4,6} through each head, writes results/v4_flips_results.csv.
# No retraining, no GPU, no overwrite of existing results.
# ---------------------------------------------------------------------------

PROJECT=/gpfs/data/mankowskilab/chen/histology-tta
PYTHON=$PROJECT/.venv/bin/python
cd "$PROJECT" || exit 1

echo "Started: $(date)"
$PYTHON -u compute_v4_flips.py
echo "Finished: $(date)"
