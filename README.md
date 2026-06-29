# When Do Histology Foundation Models Need Geometric Test-Time Augmentation?

Code for the paper: *"When Do Histology Foundation Models Need Geometric Test-Time Augmentation? Evidence from Large-Scale Experiments"*

> **TL;DR** — $D_4$ TTA is a reliable post hoc improvement for patch classification, but gains shrink from +1.98 pp for general-purpose models to +0.50 pp for histology FMs. Entropy-based selective TTA recovers most gains while augmenting only ~13% of patches. Patch-level gains do not transfer to slide-level MIL.

Code: [https://github.com/lars-chen/histology-tta](https://github.com/lars-chen/histology-tta)

---

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install torch torchvision datasets timm huggingface_hub pillow \
    pyyaml tqdm scikit-learn matplotlib pandas escnn einops
```

Set `HF_CACHE_DIR` to control HuggingFace cache location.

---

## Models

### Histology Foundation Models (frozen backbone, linear probe)
| Model | Params | Arch |
|---|---|---|
| `ctranspath` | 28 M | Swin-T/14 |
| `phikon` | 86 M | ViT-B/16 |
| `phikon2` | 300 M | ViT-L/16 |
| `uni` | 300 M | ViT-L/16 |
| `uni2` | 681 M | ViT-H/16 |
| `virchow` | 632 M | ViT-H/14 |
| `virchow2` | 632 M | ViT-H/14 |
| `gigapath` | 1.1 B | ViT-g/14 |
| `hoptimus` | 1.1 B | ViT-g/14 |

### General-Purpose Models (frozen or finetuned)
| Model | Params |
|---|---|
| `resnet18` | 12 M |
| `resnet50` | 26 M |
| `convnextv2_tiny` | 28 M |
| `convnextv2_base` | 89 M |
| `dinov2_s` | 22 M |
| `dinov2_b` | 86 M |

### Equivariant Reference
| Model | Params |
|---|---|
| `d4wrn` | 12 M — architecturally $D_4$-invariant Wide ResNet |

---

## Datasets

| Key | Source | Classes | Test set |
|---|---|---|---|
| `tcga-ut` | HF: `dakomura/tcga-ut` | 31 | 40.9k tiles |
| `nct-crc-100k` | HF: `1aurent/NCT-CRC-HE` | 9 | 7.2k tiles |
| `nct-crc-nonorm` | HF: `1aurent/NCT-CRC-HE` (no stain norm) | 9 | 7.2k tiles |
| `mhist` | Local: `data/mhist/` | 2 | 977 tiles |

WSI datasets (not downloaded via HF):
- **Camelyon17** — 500 lymph-node WSIs, 5 centers
- **PANDA** — prostate biopsy WSIs with Gleason-grade annotations

---

## Pipeline

### 1. Extract embeddings (frozen FM backbones)

Pre-extract all 8 $D_4$ views per patch once, cache to disk:

```bash
python extract_embeddings.py --model phikon2 --dataset tcga-ut --amp
```

### 2. Train lightweight classifier heads

```bash
# Linear probe
python train_probe.py --model phikon2 --dataset tcga-ut --head linear --seeds 0 1 2 3 42

# MLP probes and kNN
python train_probe.py --model phikon2 --dataset tcga-ut --head mlp1h mlp2h knn
```

### 3. Evaluate TTA

```bash
python evaluate_tta.py \
    --model phikon2 \
    --checkpoint checkpoints/tcga-ut_phikon2_frozen_aug_seed0_best.pt \
    --dataset tcga-ut \
    --tta_strategies none flips d4 \
    --aggregations mean logit_mean vote confidence \
    --amp
```

For general-purpose models, train end-to-end with standard augmentation:

```bash
python train.py --model convnextv2_base --dataset tcga-ut --epochs 20 --amp
```

### 4. Compile results and generate tables

```bash
python compile_results.py          # patch-level results → figures/canonical_results.csv
python compile_wsl_results.py      # WSI results
python utils/generate_tables.py    # regenerate paper/latex/tables/*.tex
```

### 5. Generate figures

Each figure has a dedicated script in `utils/`:

```bash
python utils/plot_figure3.py       # Figure 3: entropy / correction / corruption
python utils/plot_figure4.py       # Figure 4: selective TTA Pareto curves
python utils/plot_graphabs_inset.py  # Figure 1 inset
```

---

## TTA Strategies

| Strategy | Views | Description |
|---|---|---|
| `none` | 1 | Baseline (single view) |
| `flips` ($V_4$) | 4 | Klein four-group: H-flip, V-flip, 180° |
| `d4` ($D_4$) | 8 | Full dihedral group: 4 rotations × identity/H-flip |

Aggregation methods: `logit_mean` (default), `mean`, `confidence`, `vote`.

### Selective TTA

Apply $D_4$ only to high-entropy samples:

```python
# threshold t ∈ [0,1]; t=0.3 augments ~13% of patches for histology FMs
f_sel(x) = f(x)                    if H(f(x)) / ln(C) ≤ t
           (1/|G|) Σ f(g·x)        otherwise
```

---

## SLURM Scripts

Pre-configured scripts in `scripts/`. Naming convention: `train_{model}_{cls|ft}.sh`.

```bash
sbatch scripts/train_phikon2_cls.sh
sbatch scripts/train_convnextv2_b_ft.sh
sbatch scripts/train_probe_array.sh   # array job for all frozen FMs
```

---

## Smoke Test

```bash
python smoke_test.py   # synthetic data, no download needed; covers D4WRN invariance check
```

---

## Repository Layout

```
histology-tta/
├── data/              # Dataset loaders (HuggingFace + MHIST)
├── models/            # Model factory: FM backbones, general-purpose, D4WRN
├── tta/               # TTA augmentations + aggregation strategies
├── utils/             # Plotting scripts, table generation, trainer, metrics
├── scripts/           # SLURM job scripts
├── analysis_*.py      # Post-hoc analyses (orbit geometry, flip discrimination, etc.)
├── paper/             # LaTeX source + figures/tables for submission
│   └── latex/
│       ├── figures/   # figure2.pdf, figure3_journal.pdf, figure4_selective_pareto.pdf
│       └── tables/    # full_data_model.tex, table_head_comp.tex
├── train.py           # End-to-end training (general-purpose + finetuned)
├── train_probe.py     # Probe training on cached embeddings
├── extract_embeddings.py
├── evaluate_tta.py
└── compile_results.py
```
