"""
DINOv2 (Vision Transformer) backbones with pluggable classifier heads.

DINOv2 is a self-supervised ViT pretrained on a large curated dataset
(LVD-142M). It produces rich semantic features that transfer exceptionally
well to medical imaging, including histology.

Available variants:
  dinov2_vits14  (ViT-S/14)  — 22M params  — fast
  dinov2_vitb14  (ViT-B/14)  — 86M params  — balanced ✓ recommended
  dinov2_vitl14  (ViT-L/14)  — 307M params — heavy
  dinov2_vitg14  (ViT-G/14)  — 1.1B params — very heavy

References:
  Oquab et al., 2023. DINOv2: Learning Robust Visual Features without Supervision.
  https://arxiv.org/abs/2304.07193

Usage:
    from models.dinov2 import get_dinov2_model
    model = get_dinov2_model("dinov2_s", num_classes=33, freeze_backbone=True)
    # Then fine-tune head only for fast training
"""

import torch
import torch.nn as nn
from .base import HistoBaseModel


# Feature dims for DINOv2 [CLS] token
_DINOV2_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

# Short aliases → full hub names
_ALIASES = {
    "dinov2_s": "dinov2_vits14",
    "dinov2_b": "dinov2_vitb14",
    "dinov2_l": "dinov2_vitl14",
    "dinov2_g": "dinov2_vitg14",
    "dinov2_vits14": "dinov2_vits14",
    "dinov2_vitb14": "dinov2_vitb14",
    "dinov2_vitl14": "dinov2_vitl14",
    "dinov2_vitg14": "dinov2_vitg14",
}


class DINOv2Model(HistoBaseModel):
    """
    DINOv2 ViT backbone + linear classifier head.

    The backbone is loaded from torch.hub (facebookresearch/dinov2).
    Uses the [CLS] token embedding as the feature vector.

    Args:
        hub_name: full DINOv2 hub model name (e.g. "dinov2_vitb14")
        num_classes: number of output classes
        dropout: head dropout
        freeze_backbone: freeze backbone for linear probing
        use_registers: use register tokens variant (_reg suffix) if available
    """

    def __init__(
        self,
        hub_name: str,
        num_classes: int,
        dropout: float = 0.0,
        freeze_backbone: bool = True,
        use_registers: bool = False,
    ):
        self._hub_name = hub_name + ("_reg" if use_registers else "")
        self._dim = _DINOV2_DIMS[hub_name]
        # super().__init__ calls build_backbone(), so _hub_name must be set first
        super().__init__(
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

    def build_backbone(self) -> nn.Module:
        print(f"  Loading {self._hub_name} from torch.hub (facebookresearch/dinov2)...")
        backbone = torch.hub.load(
            "facebookresearch/dinov2",
            self._hub_name,
            pretrained=True,
        )
        return backbone

    @property
    def feature_dim(self) -> int:
        return self._dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract [CLS] token features from DINOv2.

        DINOv2 forward() returns the [CLS] token by default.
        For richer features you can use forward_features() and
        combine [CLS] with patch token averages.
        """
        return self.backbone(x)  # shape: (B, feature_dim)

    def extract_features_rich(self, x: torch.Tensor) -> torch.Tensor:
        """
        Concatenate [CLS] + mean of patch tokens for richer representation.
        Doubles the feature dimension.
        """
        out = self.backbone.forward_features(x)
        cls_token = out["x_norm_clstoken"]       # (B, D)
        patch_tokens = out["x_norm_patchtokens"]  # (B, N_patches, D)
        patch_mean = patch_tokens.mean(dim=1)     # (B, D)
        return torch.cat([cls_token, patch_mean], dim=1)  # (B, 2D)


class DINOv2WithAttentionProbe(DINOv2Model):
    """
    DINOv2 variant that also returns the last-layer attention map.
    Useful for visualizing which tissue regions the model attends to.

    Usage:
        model = DINOv2WithAttentionProbe("dinov2_vitb14", num_classes=33)
        logits, attn = model.forward_with_attn(x)
        # attn: (B, n_heads, H_patches, W_patches)
    """

    def forward_with_attn(self, x: torch.Tensor):
        # Get attention weights from the last transformer block
        attn = self.backbone.get_last_selfattention(x)  # (B, n_heads, N+1, N+1)
        # CLS token attention to patches: (B, n_heads, N_patches)
        attn_cls = attn[:, :, 0, 1:]
        # Reshape to spatial grid
        h = w = int(x.shape[-1] / self.backbone.patch_size)
        attn_map = attn_cls.reshape(attn_cls.shape[0], attn_cls.shape[1], h, w)
        logits = self.forward(x)
        return logits, attn_map


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_dinov2_model(
    name: str,
    num_classes: int,
    dropout: float = 0.0,
    freeze_backbone: bool = True,
    use_registers: bool = False,
    pretrained: bool = True,
) -> DINOv2Model:
    """
    Instantiate a DINOv2 model.

    Args:
        name: "dinov2_s" | "dinov2_b" | "dinov2_l" | "dinov2_g"
              or full names like "dinov2_vitb14"
        num_classes: number of output classes
        dropout: head dropout (recommended: 0.0 for frozen backbone)
        freeze_backbone: freeze ViT — fast linear probe training
        use_registers: use register variant (better for dense prediction tasks)

    Returns:
        DINOv2Model instance
    """
    if not pretrained:
        print(f"  WARNING: DINOv2 models are always loaded with pretrained weights. Ignoring pretrained=False.")
    if name not in _ALIASES:
        raise ValueError(
            f"Unknown DINOv2 variant '{name}'. "
            f"Choose from: {list(_ALIASES.keys())}"
        )
    hub_name = _ALIASES[name]
    model = DINOv2Model(
        hub_name=hub_name,
        num_classes=num_classes,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        use_registers=use_registers,
    )
    print(f"Loaded {model}")
    return model