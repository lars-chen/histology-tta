# When Do Histology Foundation Models Need Geometric Test-Time Augmentation?

Code for the paper *"When Do Histology Foundation Models Need Geometric Test-Time Augmentation? Evidence from Large-Scale Experiments"*.

> **TL;DR:** $D_4$ test-time augmentation (TTA) is a reliable post hoc improvement for patch classification, but on TCGA-UT the gain shrinks from +1.98 pp for general-purpose backbones to +0.50 pp for histology foundation models (FMs). Entropy-routed selective TTA recovers over half the gain while augmenting only ~13% of patches. Patch-level gains do not transfer to slide-level multiple instance learning (MIL).

---

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install torch torchvision datasets timm huggingface_hub pillow \
    pyyaml tqdm scikit-learn scipy matplotlib pandas pyarrow escnn einops \
    h5py openslide-python
```

`h5py` and `openslide-python` are only needed for the whole-slide (WSI) experiments.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `HISTO_EMB_DIR` | Cached patch embeddings (written by `extract_embeddings.py`, read by probes and analyses) | `embeddings` |
| `HF_TOKEN` | Hugging Face token for gated models (UNI, UNI-v2, Virchow, Virchow-v2, GigaPath, H-Optimus-1) | none |
| `HF_HOME` | Hugging Face cache location | `~/.cache/huggingface` |
| `CAMELYON17_DIR` | Camelyon17 challenge data (lesion annotations under `training/lesion_annotations`) | `data/camelyon17` |
| `CAMELYON17_PATCHED_DIR` | CLAM output for Camelyon17 (`patches/`, `slides_symlinks/`) | `data/camelyon17_patched` |
| `PANDA_DIR` | PANDA data (`train.csv`, `train_images/`, `train_label_masks/`, `clam_patches/patches/`) | `data/panda` |

All commands below are run from the repository root.

---

## Models

| Family | Keys |
|---|---|
| Histology FMs (frozen backbone) | `ctranspath`, `phikon`, `phikon2`, `uni`, `uni2`, `virchow`, `virchow2`, `gigapath`, `hoptimus` |
| General-purpose (frozen or finetuned) | `resnet18`, `resnet50`, `convnextv2_tiny`, `convnextv2_base`, `dinov2_s`, `dinov2_b` |
| Equivariant reference (trained from scratch) | `d4wrn`, a $D_4$-invariant Wide ResNet built with `escnn` |

Exact repository identifiers, pinned revisions and access conditions are listed in the paper's appendix (provenance table).

## Datasets

| Key | Source | Classes | Test tiles |
|---|---|---|---|
| `tcga-ut` | Hugging Face `dakomura/tcga-ut` | 31 | 40.9k |
| `nct-crc-100k` | Hugging Face `1aurent/NCT-CRC-HE` (stain-normalized) | 9 | 7.2k |
| `nct-crc-nonorm` | Hugging Face `1aurent/NCT-CRC-HE` (no stain normalization) | 9 | 7.2k |
| `mhist` | Local, `data/mhist/` (`images/` + `annotations.csv`), from the dataset authors | 2 | 977 |

The Hugging Face datasets download automatically on first use. MHIST must be requested from its authors and placed in `data/mhist/`. Camelyon17 and PANDA are obtained from the challenge organizers and Kaggle, and patched with [CLAM](https://github.com/mahmoodlab/CLAM).

---

## Patch-level pipeline

### 1. Extract embeddings (frozen backbones)

All eight $D_4$ views of every test patch, and one view of every training patch, are embedded once and cached under `$HISTO_EMB_DIR/{model}/{dataset}/{split}/`:

```bash
python extract_embeddings.py --model uni --dataset tcga-ut --amp
```

### 2. Train heads and evaluate TTA

`train_probe.py` selects hyperparameters by 3-fold cross-validation, trains one head on the cached embeddings, and evaluates all TTA strategies and aggregations from the cached views:

```bash
python train_probe.py --model uni --dataset tcga-ut --head_type linear --seed 0
```

- `--head_type` is one of `linear`, `mlp_1h`, `mlp_2h`, `knn_5`, `knn_10`. The paper's main results use `linear`.
- The paper uses seeds `0 1 2 3 42`. kNN heads are deterministic, so they only need seed 0.
- Outputs go to `results/raw/`: `probe_{dataset}_{model}_{head}_seed{N}.json` (aggregate metrics) and `probe_persample_..._seed{N}.parquet` (per-patch predictions, confidence, entropy). Checkpoints go to `checkpoints/frozen/`.

To run the full sweep (15 models × 4 datasets × 5 seeds, all heads) as a SLURM array:

```bash
sbatch scripts/train_probe_array.sh
```

### 3. Finetune general-purpose models (optional)

```bash
python train.py --model convnextv2_base --dataset tcga-ut \
    --epochs 30 --batch_size 64 --lr 1e-4 --patience 5 --seed 0 --amp
python evaluate_tta.py --model convnextv2_base --dataset tcga-ut \
    --checkpoint checkpoints/tcga-ut_convnextv2_base_finetuned_aug_seed0_best.pt \
    --tta_strategies none flips d4 --aggregations mean vote confidence --seed 0 --amp
```

`--lr` sets the backbone learning rate; the head uses ten times that value. The paper uses `--lr 1e-5` for the DINOv2 models. To run all datasets and seeds for one model and evaluate them, use:

```bash
sbatch scripts/train_ft.sh convnextv2_base   # skips runs that already have a checkpoint
```

Edit the `#SBATCH` partition lines in both scripts for your cluster.

### 4. Compile results

```bash
python analysis/compile_results.py
```

This writes `results/canonical_results.csv` (all frozen-probe and finetuned results) and `results/selective_tta_results.csv` (selective TTA trade-off curves). Both files are included in the repository, so Figures 2 and 4 can be regenerated without rerunning the experiments. The other figures, tables and analyses also need the per-sample outputs in `results/raw/` or the cached embeddings.

---

## TTA strategies and aggregation

| Strategy | Views | Description |
|---|---|---|
| `none` | 1 | Single-view baseline |
| `flips` ($V_4$) | 4 | Identity, horizontal flip, vertical flip, 180° rotation |
| `d4` ($D_4$) | 8 | Four rotations, each with and without a horizontal flip |

Aggregations (`tta/aggregator.py`): `mean` (probability mean, the default used in the paper), `logit_mean`, `confidence` (confidence-weighted), `vote` (majority vote). Probability mean and logit mean give the same average gain.

**Selective TTA** applies $D_4$ only when the base-view normalized entropy $H/\ln C$ exceeds a threshold $t$. Around $t = 0.25$ is a reasonable default; lower $t$ augments more patches and recovers more of the gain. Trade-off curves for all thresholds are in `results/selective_tta_results.csv`.

---

## Analyses, tables and figures

| Paper item | Command |
|---|---|
| Figure 2 | `python utils/plot_analysis.py --plot fig2` |
| Figure 3 | `python utils/plot_figure3.py --panel journal` |
| Figure 4 (selective TTA) | `python utils/plot_figure4.py` |
| Per-model and head-comparison tables | `python analysis/compute_v4_flips.py` then `python utils/generate_tables.py` |
| Flip predictors: margin, straddle ratio, orbit size (Section 4.7) | `python analysis/analysis_orbit_geometry.py --seed 0` |
| Corrections vs. corruptions among flipped patches (Section 4.8) | `python analysis/analysis_flip_discrimination.py --seed 0` |
| Reliability diagram and ECE (appendix) | `python utils/plot_reliability.py` |

Figures are written to `figures/` unless stated otherwise. The orbit-geometry and flip-discrimination analyses read the cached embeddings and the saved linear probes, so run them once per seed.

---

## Whole-slide experiments

Patch features are extracted from CLAM patch coordinates, with all eight $D_4$ views saved per patch:

```bash
# Camelyon17: 256x256 patches at level 1 (20x)
python -m camelyon17.extract_features --model uni --tta_modes d4_all

# PANDA: 256x256 patches at level 0
python -m camelyon17.extract_features --model uni --tta_modes d4_all \
    --patches_dir "$PANDA_DIR/clam_patches/patches" --slides_dir "$PANDA_DIR/train_images" \
    --out_dir panda_features --patch_level 0
```

Slide-level MIL (heads trained on single-view features; TTA applied only at inference):

```bash
python -m camelyon17.train_abmil --model uni --mil_type abmil --train_tta none --seed 42
python -m camelyon17.evaluate_abmil \
    --checkpoint checkpoints/camelyon17/abmil_uni_none_seed42/best.pt \
    --features_dir camelyon17_features/uni/d4_all --tta_eval none d4_mean d4_bag

python -m panda.train_abmil --model uni --mil_type abmil --fold 0
python -m panda.evaluate_abmil --help          # evaluation options
python analysis/compile_wsl_results.py         # merge into results/wsl_results.csv
```

Patch-level segmentation on WSIs:

```bash
python -m camelyon17.lopo_patch_probe --model uni   # leave-one-patient-out patch probe, per-slide Dice
python -m panda.train_patch_probe --model uni
```

---

## Repository layout

```
histology-tta/
├── data/                 # Dataset loaders (Hugging Face datasets + MHIST)
├── models/               # Backbones: histology FMs, general-purpose models, D4WRN
├── tta/                  # D4 views and aggregation
├── utils/                # Plotting, table generation, trainer, metrics
├── analysis/             # Result compilation and post hoc analyses (orbit geometry, flips, calibration)
├── camelyon17/           # Camelyon17 feature extraction, MIL, patch probes
├── panda/                # PANDA MIL and patch probes
├── scripts/              # Example SLURM sweeps
├── configs/              # Default configuration
├── results/              # Compiled results (canonical, selective TTA, WSI)
├── extract_embeddings.py # Cache D4-view embeddings for frozen backbones
├── train_probe.py        # Heads on cached embeddings + TTA evaluation
├── train.py              # End-to-end training (finetuned and D4WRN models)
└── evaluate_tta.py       # TTA evaluation for end-to-end models
```

Cached embeddings and checkpoints are not distributed because of the license terms of several source models. They can be regenerated with the commands above.
