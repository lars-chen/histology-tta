# Histological TTA (Test-Time Augmentation)

A research framework for exploring symmetries in histological data via test-time augmentation.

## Project Structure

```
histology-tta/
├── data/               # Dataset loaders and preprocessing
│   ├── dataset.py      # HuggingFace TCGA-UT dataset loader
│   └── transforms.py   # Train/val/TTA transform pipelines
├── models/             # Model zoo with pluggable classifier heads
│   ├── base.py         # Abstract base model
│   ├── torchvision_models.py  # ResNet18/50, EfficientNet-B7, VGG-16
│   └── dinov2.py       # DINOv2 (ViT-S/B/L) backbone
├── tta/                # TTA strategies
│   ├── augmentations.py  # Histology-specific TTA transforms
│   └── aggregator.py     # Prediction aggregation (mean, max, vote)
├── utils/
│   ├── trainer.py      # Training loop
│   └── metrics.py      # Evaluation metrics
├── configs/
│   └── default.yaml    # Hyperparameters
├── notebooks/
│   └── explore_tta.ipynb  # Exploration notebook
├── train.py            # Training entry point
└── evaluate_tta.py     # TTA evaluation entry point
```

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install torch torchvision datasets timm huggingface_hub pillow pyyaml tqdm scikit-learn
```

## Quick Start

### 1. Train a model

```bash
# Lightweight — fast iteration
python train.py --model resnet18 --epochs 10

# Medium
python train.py --model resnet50 --epochs 20

# Heavy
python train.py --model vgg16 --epochs 20
python train.py --model efficientnet_b7 --epochs 20
python train.py --model dinov2_s --epochs 10  # ViT-Small
python train.py --model dinov2_b --epochs 10  # ViT-Base
```

### 2. Evaluate with TTA

```bash
python evaluate_tta.py --model resnet50 --checkpoint checkpoints/resnet50_best.pt \
    --tta_strategies none d4 d4_color flips
```

## Histological Symmetries

Histology images have well-known symmetries that TTA can exploit:

| Symmetry | Description | TTA strategy |
|---|---|---|
| Rotational (D4) | H&E slides have no canonical orientation | 0°, 90°, 180°, 270° + flips |
| Color jitter | Stain variation across labs/batches | Hue/saturation perturbations |
| Multi-scale | Pathologists look at multiple magnifications | Scale jittering |
| Elastic deformation | Tissue warping artifacts | Elastic transforms |

## Models Available

| Model | Params | Speed | Notes |
|---|---|---|---|
| `resnet18` | 11M | ⚡⚡⚡ | Great for rapid prototyping |
| `resnet50` | 25M | ⚡⚡ | Good accuracy/speed balance |
| `vgg16` | 138M | ⚡ | Heavy but well-studied |
| `efficientnet_b7` | 66M | ⚡ | State-of-art CNN scaling |
| `dinov2_s` | 22M | ⚡⚡ | Self-supervised ViT-Small |
| `dinov2_b` | 86M | ⚡ | Self-supervised ViT-Base |
| `dinov2_l` | 307M | 🐢 | Self-supervised ViT-Large |