"""
ConvNeXt V2 backbones with pluggable classifier heads (loaded via timm).

ConvNeXt V2 improves on V1 with Global Response Normalization (GRN) and
masked-autoencoder pretraining (FCMAE), giving stronger features with the
same efficient fully-convolutional architecture.

Available variants (ImageNet-22k → 1k fine-tuned weights):
  convnextv2_tiny   (28M params,  feature_dim=768)  — fast, recommended start
  convnextv2_small  (50M params,  feature_dim=768)  — balanced
  convnextv2_base   (89M params,  feature_dim=1024) — strong baseline
  convnextv2_large  (198M params, feature_dim=1536) — high accuracy
  convnextv2_huge   (660M params, feature_dim=2816) — maximum accuracy

Short aliases:  convnextv2_t, convnextv2_s, convnextv2_b, convnextv2_l, convnextv2_h

References:
  Woo et al., 2023. ConvNeXt V2: Co-designing and Scaling ConvNets with
  Masked Autoencoders. https://arxiv.org/abs/2301.00808

Usage:
    from models.convnextv2 import get_convnextv2_model
    model = get_convnextv2_model("convnextv2_base", num_classes=31, freeze_backbone=True)
"""

import timm
import torch.nn as nn
from .base import HistoBaseModel


# ---------------------------------------------------------------------------
# Feature dimensions (output of timm model with num_classes=0)
# ---------------------------------------------------------------------------

_CONVNEXTV2_DIMS: dict[str, int] = {
    "convnextv2_tiny":  768,
    "convnextv2_small": 768,
    "convnextv2_base":  1024,
    "convnextv2_large": 1536,
    "convnextv2_huge":  2816,
}

# Short aliases → canonical timm names
_ALIASES: dict[str, str] = {
    "convnextv2_t": "convnextv2_tiny",
    "convnextv2_s": "convnextv2_small",
    "convnextv2_b": "convnextv2_base",
    "convnextv2_l": "convnextv2_large",
    "convnextv2_h": "convnextv2_huge",
    # canonical names are also valid
    **{k: k for k in _CONVNEXTV2_DIMS},
}


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class ConvNeXtV2Model(HistoBaseModel):
    """
    ConvNeXt V2 backbone + linear classifier head (timm-backed).

    The timm model is created with num_classes=0, which removes the original
    classification head and returns a pooled (B, feature_dim) feature vector
    directly — compatible with HistoBaseModel.

    Args:
        timm_name: canonical timm model name (e.g. "convnextv2_base")
        num_classes: number of output classes
        dropout: head dropout probability
        freeze_backbone: freeze backbone weights for linear probing
        pretrained: load ImageNet weights (default: True)
    """

    def __init__(
        self,
        timm_name: str,
        num_classes: int,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
        pretrained: bool = True,
        mlp_hidden: int = None,
    ):
        self._timm_name = timm_name
        self._dim = _CONVNEXTV2_DIMS[timm_name]
        self._pretrained = pretrained
        # super().__init__ calls build_backbone(), so _timm_name must be set first
        super().__init__(
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            mlp_hidden=mlp_hidden,
        )

    def build_backbone(self) -> nn.Module:
        print(f"  Loading {self._timm_name} from timm (pretrained={self._pretrained})...")
        backbone = timm.create_model(
            self._timm_name,
            pretrained=self._pretrained,
            num_classes=0,  # remove head → returns pooled feature vector
        )
        return backbone

    @property
    def feature_dim(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_convnextv2_model(
    name: str,
    num_classes: int,
    dropout: float = 0.0,
    freeze_backbone: bool = False,
    pretrained: bool = True,
    mlp_hidden: int = None,
) -> ConvNeXtV2Model:
    """
    Instantiate a ConvNeXt V2 model.

    Args:
        name: variant key — full name (e.g. "convnextv2_base") or short alias
              (convnextv2_t / _s / _b / _l / _h)
        num_classes: number of output classes
        dropout: head dropout (0.2 is a sensible default)
        freeze_backbone: freeze backbone for linear probing
        pretrained: load ImageNet-22k → 1k fine-tuned weights

    Returns:
        ConvNeXtV2Model instance
    """
    if name not in _ALIASES:
        raise ValueError(
            f"Unknown ConvNeXt V2 variant '{name}'. "
            f"Choose from: {list(_ALIASES.keys())}"
        )
    timm_name = _ALIASES[name]
    model = ConvNeXtV2Model(
        timm_name=timm_name,
        num_classes=num_classes,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        pretrained=pretrained,
        mlp_hidden=mlp_hidden,
    )
    print(f"Loaded {model}")
    return model
