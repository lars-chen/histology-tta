"""
Pre-trained histology foundation models with pluggable classifier heads.

Supports models loaded via timm (HF Hub) or HuggingFace Transformers.
All models return pooled (B, feature_dim) feature vectors compatible with
HistoBaseModel.

Available models:
  gigapath   (1.1B params, feature_dim=1536) — Prov-GigaPath ViT-g
  hoptimus   (1.1B params, feature_dim=1536) — H-optimus-1 ViT-g (bioptimus)
  uni        (300M params, feature_dim=1024) — UNI ViT-L
  uni2       (681M params, feature_dim=1536) — UNI2-h ViT-H
  phikon     (86M params,  feature_dim=768)  — Phikon ViT-B
  phikon2    (300M params, feature_dim=1024) — Phikon-v2 ViT-L
  virchow    (632M params, feature_dim=1280) — Virchow ViT-H
  virchow2   (632M params, feature_dim=1280) — Virchow2 ViT-H

Gated models (gigapath, hoptimus, uni, uni2, virchow, virchow2) require a HuggingFace
token. Set the HF_TOKEN environment variable before running.

Usage:
    from models.foundation import get_foundation_model
    model = get_foundation_model("phikon", num_classes=31, freeze_backbone=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .base import HistoBaseModel


# ---------------------------------------------------------------------------
# Transformer-output adapter (for HF Transformers models)
# ---------------------------------------------------------------------------

class _TransformersBackbone(nn.Module):
    """Wraps a HuggingFace Transformers ViT to return CLS token as (B, D)."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.hf_model(pixel_values=x)
        # CLS token is the first token of last_hidden_state
        return out.last_hidden_state[:, 0, :]


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass
class _ModelConfig:
    repo: str
    dim: int
    loader: str  # "timm" or "transformers"
    timm_kwargs: dict[str, Any] = field(default_factory=dict)
    gated: bool = False


def _swiglu_kwargs() -> dict[str, Any]:
    """Kwargs needed for Virchow-family models (SwiGLU MLP)."""
    from timm.layers import SwiGLUPacked
    return {"mlp_layer": SwiGLUPacked, "act_layer": nn.SiLU}


def _uni2_kwargs() -> dict[str, Any]:
    """Kwargs needed for UNI2-h model."""
    from timm.layers import SwiGLUPacked
    return {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "no_embed_class": True,
        "mlp_layer": SwiGLUPacked,
        "act_layer": nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }


_FOUNDATION_MODELS: dict[str, _ModelConfig] = {
    "gigapath": _ModelConfig(
        repo="prov-gigapath/prov-gigapath",
        dim=1536,
        loader="timm",
        gated=True,
    ),
    "hoptimus": _ModelConfig(
        repo="bioptimus/H-optimus-1",
        dim=1536,
        loader="timm",
        timm_kwargs={"init_values": 1e-5, "dynamic_img_size": False},
        gated=True,
    ),
    "uni": _ModelConfig(
        repo="MahmoodLab/UNI",
        dim=1024,
        loader="timm",
        timm_kwargs={"init_values": 1e-5, "dynamic_img_size": True},
        gated=True,
    ),
    "uni2": _ModelConfig(
        repo="MahmoodLab/UNI2-h",
        dim=1536,
        loader="timm",
        # kwargs built lazily via _uni2_kwargs() because they import SwiGLUPacked
        gated=True,
    ),
    "phikon": _ModelConfig(
        repo="owkin/phikon",
        dim=768,
        loader="transformers",
    ),
    "phikon2": _ModelConfig(
        repo="owkin/phikon-v2",
        dim=1024,
        loader="transformers",
    ),
    "virchow": _ModelConfig(
        repo="paige-ai/Virchow",
        dim=1280,
        loader="timm",
        # kwargs built lazily via _swiglu_kwargs()
        gated=True,
    ),
    "virchow2": _ModelConfig(
        repo="paige-ai/Virchow2",
        dim=1280,
        loader="timm",
        # kwargs built lazily via _swiglu_kwargs()
        gated=True,
    ),
}


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class FoundationModel(HistoBaseModel):
    """
    Histology foundation model backbone + linear classifier head.

    Args:
        config: _ModelConfig from the registry
        num_classes: number of output classes
        dropout: head dropout probability
        freeze_backbone: freeze backbone weights for linear probing
    """

    def __init__(
        self,
        config: _ModelConfig,
        num_classes: int,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
    ):
        self._config = config
        # super().__init__ calls build_backbone(), so _config must be set first
        super().__init__(
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

    def build_backbone(self) -> nn.Module:
        cfg = self._config

        if cfg.loader == "timm":
            return self._build_timm_backbone(cfg)
        elif cfg.loader == "transformers":
            return self._build_transformers_backbone(cfg)
        else:
            raise ValueError(f"Unknown loader type: {cfg.loader}")

    def _build_timm_backbone(self, cfg: _ModelConfig) -> nn.Module:
        import timm

        # Build kwargs — some models need lazily-constructed kwargs
        kwargs = dict(cfg.timm_kwargs)
        if cfg.repo == "MahmoodLab/UNI2-h":
            kwargs = _uni2_kwargs()
        elif cfg.repo in ("paige-ai/Virchow", "paige-ai/Virchow2"):
            kwargs = {**_swiglu_kwargs(), "global_pool": "token"}

        hub_name = f"hf_hub:{cfg.repo}"
        print(f"  Loading {hub_name} from timm...")
        try:
            backbone = timm.create_model(
                hub_name,
                pretrained=True,
                num_classes=0,
                **kwargs,
            )
        except Exception as e:
            if cfg.gated:
                raise RuntimeError(
                    f"Failed to load gated model '{cfg.repo}'. "
                    f"Set the HF_TOKEN environment variable with a token that "
                    f"has access to {cfg.repo}. Original error: {e}"
                ) from e
            raise
        return backbone

    def _build_transformers_backbone(self, cfg: _ModelConfig) -> nn.Module:
        from transformers import AutoModel

        print(f"  Loading {cfg.repo} from HuggingFace Transformers...")
        try:
            hf_model = AutoModel.from_pretrained(cfg.repo)
        except Exception as e:
            if cfg.gated:
                raise RuntimeError(
                    f"Failed to load gated model '{cfg.repo}'. "
                    f"Set the HF_TOKEN environment variable with a token that "
                    f"has access to {cfg.repo}. Original error: {e}"
                ) from e
            raise
        return _TransformersBackbone(hf_model)

    @property
    def feature_dim(self) -> int:
        return self._config.dim


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_foundation_model(
    name: str,
    num_classes: int,
    dropout: float = 0.0,
    freeze_backbone: bool = False,
    pretrained: bool = True,
) -> FoundationModel:
    """
    Instantiate a histology foundation model.

    Args:
        name: model key (gigapath, hoptimus, uni, uni2, phikon, phikon2, virchow, virchow2)
        num_classes: number of output classes
        dropout: head dropout (0.2 is a sensible default)
        freeze_backbone: freeze backbone for linear probing
        pretrained: always True for foundation models (warning printed if False)

    Returns:
        FoundationModel instance
    """
    if not pretrained:
        print(f"  WARNING: Foundation models are always loaded with pretrained weights. Ignoring pretrained=False.")
    if name not in _FOUNDATION_MODELS:
        raise ValueError(
            f"Unknown foundation model '{name}'. "
            f"Choose from: {list(_FOUNDATION_MODELS.keys())}"
        )
    config = _FOUNDATION_MODELS[name]
    model = FoundationModel(
        config=config,
        num_classes=num_classes,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
    print(f"Loaded {model}")
    return model
