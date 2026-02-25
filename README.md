# Histological TTA (Test-Time Augmentation)

A research framework for training histology classifiers and evaluating the effect of test-time augmentation (TTA) across multiple model families and datasets.

## Project Structure

```
histology-tta/
├── data/
│   ├── data.py              # HuggingFace dataset loader (HistoDataset, get_dataloaders)
│   ├── dataset.py           # Thin wrapper / re-exports for backwards compatibility
│   └── transforms.py        # Train/val/TTA transform pipelines
├── models/
│   ├── __init__.py          # get_model() factory — dispatches by name
│   ├── base.py              # Abstract HistoBaseModel (backbone + linear head)
│   ├── torchvision_models.py  # ResNet-18/50, EfficientNet-B7, VGG-16, MobileNet-V3
│   ├── dinov2.py            # DINOv2 ViT-S/B/L/G (facebookresearch/dinov2)
│   └── convnextv2.py        # ConvNeXt V2 T/S/B/L/H (timm)
├── tta/
│   ├── augmentations.py     # Histology TTA strategies (flips, D4, D4+color)
│   └── aggregator.py        # Prediction aggregation (mean, vote, confidence)
├── utils/
│   ├── trainer.py           # Training loop with AMP, early stopping, checkpointing
│   ├── metrics.py           # Per-class accuracy and F1
│   └── plot_results.py      # Load JSON results → publication figures
├── scripts/                 # SLURM job scripts (one per model × training mode)
├── train.py                 # Training entry point
├── evaluate_tta.py          # TTA evaluation entry point
└── smoke_test.py            # Sanity check (synthetic data, no download needed)
```

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install torch torchvision datasets timm huggingface_hub pillow pyyaml tqdm scikit-learn matplotlib pandas
```

## Datasets

| Key | HuggingFace repo | Train split | Test split | Classes |
|---|---|---|---|---|
| `tcga-ut` | `dakomura/tcga-ut` | `train` | `test` | 31 |
| `nct-crc-100k` | `1aurent/NCT-CRC-HE` | `NCT_CRC_HE_100K` | `CRC_VAL_HE_7K` | 9 |
| `nct-crc-nonorm` | `1aurent/NCT-CRC-HE` | `NCT_CRC_HE_100K_NONORM` | `CRC_VAL_HE_7K` | 9 |

Datasets are downloaded from HuggingFace Hub on first use and cached locally.
Set `HF_CACHE_DIR` to override the default cache location, or pass `--cache_dir`.

For TCGA-UT the validation split is done at the **patient level** (parsed from the `__key__` TCGA barcode) to prevent slide-leakage between train and val.

## Models

### CNN (torchvision)
| Model | Params |
|---|---|
| `resnet18` | 11 M |
| `resnet50` | 25 M |
| `vgg16` | 138 M |
| `efficientnet_b7` | 66 M |
| `mobilenet_v3` | 5 M |

### Vision Transformer — DINOv2
| Model | Params |
|---|---|
| `dinov2_s` | 22 M |
| `dinov2_b` | 86 M |
| `dinov2_l` | 307 M |
| `dinov2_g` | 1.1 B |

### ConvNeXt V2 (timm) — short aliases in parentheses
| Model | Params |
|---|---|
| `convnextv2_tiny` (`convnextv2_t`) | 28 M |
| `convnextv2_small` (`convnextv2_s`) | 50 M |
| `convnextv2_base` (`convnextv2_b`) | 89 M |
| `convnextv2_large` (`convnextv2_l`) | 198 M |
| `convnextv2_huge` (`convnextv2_h`) | 660 M |

All models share the same interface: `get_model(name, num_classes, freeze_backbone=...)`.

**Training modes**
- `cls` — frozen backbone, linear classifier head only (linear probe)
- `ft` — full fine-tune of backbone + head

## Quick Start

### Smoke test (no data download)
```bash
python smoke_test.py
```

### Train
```bash
# Linear probe (frozen backbone)
python train.py --model dinov2_b --dataset tcga-ut --freeze_backbone --epochs 10

# Full fine-tune with AMP
python train.py --model convnextv2_base --dataset nct-crc-100k --epochs 20 --lr 1e-5 --amp

# Two-stage: linear probe then unfreeze backbone
python train.py --model dinov2_l --dataset tcga-ut --freeze_backbone --unfreeze_after 5 --epochs 20 --amp
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--model` | `resnet18` | Model name |
| `--dataset` | `tcga-ut` | Dataset key |
| `--freeze_backbone` | off | Linear probe mode |
| `--unfreeze_after N` | off | Unfreeze backbone after N epochs |
| `--lr` | `1e-4` | Backbone LR (head uses 10×) |
| `--amp` | off | Mixed-precision (fp16) |
| `--patience N` | `0` | Early stopping (0 = disabled) |
| `--checkpoint_dir` | `checkpoints/` | Where to save `.pt` files |
| `--cache_dir` | HF default | HuggingFace dataset cache |
| `--seed` | `42` | Random seed (submit separate jobs with different seeds for variance) |

Checkpoints are saved as `{dataset}_{model}_{frozen|finetuned}_seed{seed}_best.pt`.

### Evaluate with TTA
```bash
python evaluate_tta.py \
    --model convnextv2_base \
    --checkpoint checkpoints/tcga-ut_convnextv2_base_finetuned_seed42_best.pt \
    --dataset tcga-ut \
    --tta_strategies none flips d4 d4_color \
    --aggregations mean vote confidence
```

For variance estimation, train and evaluate with multiple seeds (submit as separate SLURM jobs for parallelism):
```bash
for SEED in 42 137 2024; do
    # Train
    python train.py \
        --model convnextv2_base \
        --dataset tcga-ut \
        --epochs 20 --lr 1e-5 --amp --patience 2 \
        --seed $SEED

    # Evaluate
    python evaluate_tta.py \
        --model convnextv2_base \
        --checkpoint checkpoints/tcga-ut_convnextv2_base_finetuned_seed${SEED}_best.pt \
        --dataset tcga-ut --seed $SEED \
        --tta_strategies none flips d4 d4_color \
        --aggregations mean vote confidence
done
```

Each run produces `tta_results_{dataset}_{model}_{frozen|finetuned}_seed{seed}.json` with one row per (strategy, aggregation) including accuracy, balanced accuracy, macro-F1, per-class F1, and corrected/corrupted counts. Aggregate across seed JSON files externally for mean±std.

### Plot results
```bash
# All JSON files in current directory → figures/
python utils/plot_results.py

# Specific files, custom output directory
python utils/plot_results.py tta_results_tcga-ut_*.json --out_dir figures/tcga/

# Use plain accuracy instead of balanced accuracy
python utils/plot_results.py --metric acc
```

Figures produced:

| File | Description |
|---|---|
| `acc_by_strategy_{dataset}_{metric}.png` | Grouped bar chart per (model, mode) |
| `heatmap_{dataset}_{metric}.png` | Model × strategy grid (best aggregation per cell) |
| `correction_{dataset}.png` | TTA corrected vs. corrupted samples per strategy |
| `per_class_f1_{dataset}.png` | Per-class F1 heatmap for best strategy |

## SLURM Scripts

Pre-configured scripts live in `scripts/`. Naming convention:

```
scripts/train_{model}_{cls|ft}.sh
```

Each script trains on all three datasets sequentially, then runs TTA evaluation on each.

```bash
sbatch scripts/train_dinov2_b_cls.sh
sbatch scripts/train_convnextv2_b_ft.sh
```

Set `HF_CACHE_DIR` before submitting to override the cache path:
```bash
HF_CACHE_DIR=/path/to/cache sbatch scripts/train_dinov2_b_ft.sh
```

## TTA Strategies

| Strategy | Views | Description |
|---|---|---|
| `none` | 1 | No augmentation (baseline) |
| `flips` | 4 | Horizontal + vertical flips |
| `d4` | 8 | Full dihedral group (4 rotations × 2 flips) |
| `d4_color` | 8 | D4 + color jitter (hue/saturation) |

Aggregation methods: `mean` (average logits), `vote` (majority class), `confidence` (highest max-softmax).

## Histological Symmetries

Histology images have well-known symmetries that TTA can exploit:

| Symmetry | Rationale | TTA strategy |
|---|---|---|
| Rotational (D4) | H&E slides have no canonical orientation | `d4`, `d4_color` |
| Color jitter | Stain variation across labs and batches | `d4_color` |
| Horizontal / vertical flip | Tissue has no inherent handedness | `flips`, `d4` |
