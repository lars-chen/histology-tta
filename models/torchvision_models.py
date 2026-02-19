"""
Torchvision model backbones with pluggable classifier heads.

Models:
  - ResNet-18     (11M params)  — fastest, good for rapid prototyping
  - ResNet-50     (25M params)  — balanced
  - VGG-16        (138M params) — classic, heavy
  - EfficientNet-B7 (66M params) — high accuracy CNN

All use ImageNet pretrained weights by default.

Usage:
    from models.torchvision_models import get_model
    model = get_model("resnet50", num_classes=33, freeze_backbone=True)
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights,
    VGG16_BN_Weights, EfficientNet_B7_Weights,
    MobileNet_V3_Large_Weights,
)
from .base import HistoBaseModel


# ---------------------------------------------------------------------------
# ResNet-18
# ---------------------------------------------------------------------------

class ResNet18(HistoBaseModel):
    """
    ResNet-18 backbone (11M params).
    Great for fast iteration and baselines.
    """
    _feature_dim = 512

    def build_backbone(self) -> nn.Module:
        weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = tvm.resnet18(weights=weights)
        # Remove final FC layer — we supply our own head
        backbone.fc = nn.Identity()
        return backbone

    @property
    def feature_dim(self) -> int:
        return self._feature_dim


# ---------------------------------------------------------------------------
# ResNet-50
# ---------------------------------------------------------------------------

class ResNet50(HistoBaseModel):
    """
    ResNet-50 backbone (25M params).
    Good accuracy/speed trade-off.
    """
    _feature_dim = 2048

    def build_backbone(self) -> nn.Module:
        weights = ResNet50_Weights.IMAGENET1K_V2
        backbone = tvm.resnet50(weights=weights)
        backbone.fc = nn.Identity()
        return backbone

    @property
    def feature_dim(self) -> int:
        return self._feature_dim


# ---------------------------------------------------------------------------
# VGG-16 (with BatchNorm)
# ---------------------------------------------------------------------------

class VGG16(HistoBaseModel):
    """
    VGG-16 with BatchNorm (138M params).
    Classic architecture, slower but historically well-studied.
    Uses only the convolutional feature extractor (features + avgpool).
    """
    _feature_dim = 4096

    def build_backbone(self) -> nn.Module:
        weights = VGG16_BN_Weights.IMAGENET1K_V1
        vgg = tvm.vgg16_bn(weights=weights)
        # VGG: features (conv) → avgpool → classifier (FC layers)
        # We keep features + avgpool + first two FC layers as "backbone"
        # to get a 4096-d representation, then add our own head.
        backbone = nn.Sequential(
            vgg.features,
            vgg.avgpool,
            nn.Flatten(),
            vgg.classifier[0],   # Linear 25088→4096
            vgg.classifier[1],   # ReLU
            vgg.classifier[2],   # Dropout
            vgg.classifier[3],   # Linear 4096→4096
            vgg.classifier[4],   # ReLU
            vgg.classifier[5],   # Dropout
        )
        return backbone

    @property
    def feature_dim(self) -> int:
        return self._feature_dim


# ---------------------------------------------------------------------------
# EfficientNet-B7
# ---------------------------------------------------------------------------

class EfficientNetB7(HistoBaseModel):
    """
    EfficientNet-B7 (66M params).
    State-of-the-art CNN scaling — excellent accuracy but slow to train.
    Expects 600×600 images for optimal performance, works fine at 224.
    """
    _feature_dim = 2560

    def build_backbone(self) -> nn.Module:
        weights = EfficientNet_B7_Weights.IMAGENET1K_V1
        eff = tvm.efficientnet_b7(weights=weights)
        # EfficientNet: features → avgpool → classifier
        # We take features + adaptive pooling → flat vector
        backbone = nn.Sequential(
            eff.features,
            eff.avgpool,
            nn.Flatten(),
        )
        return backbone

    @property
    def feature_dim(self) -> int:
        return self._feature_dim


# ---------------------------------------------------------------------------
# MobileNetV3-Large (bonus: very lightweight)
# ---------------------------------------------------------------------------

class MobileNetV3Large(HistoBaseModel):
    """
    MobileNet-V3 Large (5.4M params).
    Extremely fast — good sanity-check baseline.
    """
    _feature_dim = 960

    def build_backbone(self) -> nn.Module:
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        mob = tvm.mobilenet_v3_large(weights=weights)
        # Remove classifier, keep features + avgpool
        backbone = nn.Sequential(
            mob.features,
            mob.avgpool,
            nn.Flatten(),
        )
        return backbone

    @property
    def feature_dim(self) -> int:
        return self._feature_dim


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_MODEL_REGISTRY = {
    "resnet18":        ResNet18,
    "resnet50":        ResNet50,
    "vgg16":           VGG16,
    "efficientnet_b7": EfficientNetB7,
    "mobilenet_v3":    MobileNetV3Large,
}


def get_torchvision_model(
    name: str,
    num_classes: int,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
) -> HistoBaseModel:
    """
    Instantiate a torchvision-based model.

    Args:
        name: model key (resnet18, resnet50, vgg16, efficientnet_b7, mobilenet_v3)
        num_classes: number of output classes
        dropout: head dropout
        freeze_backbone: freeze backbone weights (fast linear probe)

    Returns:
        HistoBaseModel instance
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list(_MODEL_REGISTRY.keys())}"
        )
    model = _MODEL_REGISTRY[name](
        num_classes=num_classes,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
    print(f"Loaded {model}")
    return model