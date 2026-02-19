"""
Transform pipelines for histological image classification.

Key insight: H&E histology patches have several important symmetries:
  1. D4 symmetry (dihedral group of order 8): 4 rotations × 2 flips
     — slides have NO canonical orientation
  2. Stain variation: hue/saturation shift from different scanners / staining labs
  3. Multi-scale context: pathologists zoom in/out
  4. Compression artifacts, slight blur

This module provides:
  - get_train_transform()  : standard augmented pipeline for training
  - get_val_transform()    : deterministic pipeline for validation/testing
  - get_tta_transforms()   : list of deterministic transforms for TTA inference
"""

from typing import List
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch
from PIL import Image

# ImageNet stats are a reasonable starting point even for H&E patches.
# For domain-specific normalization, compute stats from your training set.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# TCGA-UT patches are already 256×256; models usually want 224×224
DEFAULT_CROP = 224


# ---------------------------------------------------------------------------
# Standard training transform
# ---------------------------------------------------------------------------

def get_train_transform(crop_size: int = DEFAULT_CROP) -> T.Compose:
    """
    Augmented pipeline for training.

    Includes:
      - Random crop (scale jitter)
      - D4 flips + 90° rotations
      - Color jitter (stain simulation)
      - Gaussian blur (scanner PSF variation)
      - Normalization
    """
    return T.Compose([
        T.RandomResizedCrop(
            crop_size,
            scale=(0.6, 1.0),   # multi-scale context
            ratio=(0.85, 1.15),
        ),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        # 90° grid rotation — crucial for histology
        T.RandomApply([T.RandomRotation(degrees=(90, 90))], p=0.5),
        T.RandomApply([T.RandomRotation(degrees=(180, 180))], p=0.25),
        T.RandomApply([T.RandomRotation(degrees=(270, 270))], p=0.25),
        # Stain variation simulation
        T.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25,
            hue=0.05,          # H&E hue range is narrow
        ),
        T.RandomGrayscale(p=0.02),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Validation / test transform (deterministic)
# ---------------------------------------------------------------------------

def get_val_transform(crop_size: int = DEFAULT_CROP) -> T.Compose:
    """Minimal, deterministic pipeline for evaluation."""
    return T.Compose([
        T.Resize(int(crop_size * 256 / 224)),  # standard resize ratio
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# TTA transform factory
# ---------------------------------------------------------------------------

class TTATransform:
    """
    A single deterministic TTA augmentation.

    Wraps a sequence of PIL-space transforms (before ToTensor) so they
    can be composed with the base val transform.

    Args:
        name: human-readable label (used in results tables)
        pil_ops: list of callables PIL→PIL applied before normalization
        crop_size: final crop size
    """

    def __init__(self, name: str, pil_ops: list, crop_size: int = DEFAULT_CROP):
        self.name = name
        self._transform = T.Compose([
            T.Resize(int(crop_size * 256 / 224)),
            T.CenterCrop(crop_size),
            *pil_ops,
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self._transform(img)

    def __repr__(self) -> str:
        return f"TTATransform({self.name})"


def get_tta_transforms(
    strategy: str = "d4",
    crop_size: int = DEFAULT_CROP,
) -> List[TTATransform]:
    """
    Return a list of TTA transforms for a given strategy.

    Args:
        strategy: one of
            "none"      — identity only (baseline)
            "flips"     — hflip, vflip, both
            "d4"        — all 8 elements of the dihedral group D4
                          (4 rotations × 2 flips) — recommended for histology
            "d4_color"  — D4 + mild color jitter variants
            "multiscale"— centre crop at 3 different scales
            "full"      — D4 + multiscale + color (aggressive, ~30 views)

    Returns:
        List of TTATransform objects. Pass each to the model and aggregate.
    """

    def rot(deg):
        """PIL rotation, expand=False (square patches stay square)."""
        return T.Lambda(lambda img: TF.rotate(img, deg))

    def hflip():
        return T.Lambda(lambda img: TF.hflip(img))

    identity = []  # no-op PIL ops

    # --- D4 group: 8 deterministic transforms ---
    d4_ops = [
        ("orig",      []),
        ("rot90",     [rot(90)]),
        ("rot180",    [rot(180)]),
        ("rot270",    [rot(270)]),
        ("hflip",     [hflip()]),
        ("hflip+rot90",  [hflip(), rot(90)]),
        ("hflip+rot180", [hflip(), rot(180)]),
        ("hflip+rot270", [hflip(), rot(270)]),
    ]

    # --- Color variants ---
    def mild_color(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03):
        return T.ColorJitter(brightness, contrast, saturation, hue)

    color_variants = [
        ("color+bright", [mild_color(brightness=0.2)]),
        ("color+contrast", [mild_color(contrast=0.2)]),
        ("color+saturation", [mild_color(saturation=0.2)]),
    ]

    # --- Multi-scale crops ---
    def scale_crop(scale: float):
        size = int(crop_size * scale)
        return T.Lambda(
            lambda img: TF.center_crop(TF.resize(img, int(size * 256 / 224)), size)
        )

    multiscale_ops = [
        ("scale_0.85", [T.Resize(int(crop_size * 0.85 * 256 / 224)), T.CenterCrop(int(crop_size * 0.85))]),
        ("scale_1.15", [T.Resize(int(crop_size * 1.15 * 256 / 224)), T.CenterCrop(crop_size)]),
    ]

    # --- Build strategy ---
    if strategy == "none":
        selected = [("orig", [])]

    elif strategy == "flips":
        selected = [
            ("orig",   []),
            ("hflip",  [hflip()]),
            ("vflip",  [T.Lambda(lambda img: TF.vflip(img))]),
            ("hvflip", [hflip(), T.Lambda(lambda img: TF.vflip(img))]),
        ]

    elif strategy == "d4":
        selected = d4_ops

    elif strategy == "d4_color":
        selected = d4_ops + color_variants

    elif strategy == "multiscale":
        selected = [("orig", [])] + multiscale_ops

    elif strategy == "full":
        selected = d4_ops + color_variants + multiscale_ops

    else:
        raise ValueError(
            f"Unknown TTA strategy '{strategy}'. "
            "Choose from: none, flips, d4, d4_color, multiscale, full"
        )

    transforms = [
        TTATransform(name, ops, crop_size=crop_size)
        for name, ops in selected
    ]

    print(
        f"TTA strategy '{strategy}': {len(transforms)} views per sample "
        f"({[t.name for t in transforms]})"
    )
    return transforms