"""
Pre-trained histology foundation models with pluggable classifier heads.

Supports models loaded via timm (HF Hub) or HuggingFace Transformers.
All models return pooled (B, feature_dim) feature vectors compatible with
HistoBaseModel.

Available models:
  ctranspath (28M params,  feature_dim=768)  — CTransPath Swin-T (SSL, ungated)
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


def _build_ctranspath_backbone() -> nn.Module:
    """Load CTransPath (Swin-T with custom convolutional patch embedding).

    The checkpoint uses an older Swin convention where PatchMerging is stored
    in layers[N] but timm now places it in layers[N+1].  The norm sizes match
    numerically after the shift (4×C_in == 2×C_out since C_out=2×C_in), so a
    simple key remap is sufficient.
    """
    import re
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0)

    # Replace standard single-conv patch embed with CTransPath's Conv-BN-Conv-BN-Conv stem
    # Architecture derived from checkpoint weight shapes: 3→12→24→96, strides 2,2,1
    backbone.patch_embed.proj = nn.Sequential(
        nn.Conv2d(3,  12, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(12),
        nn.GELU(),
        nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(24),
        nn.GELU(),
        nn.Conv2d(24, 96, kernel_size=1, stride=1, bias=True),
    )

    repo = "1aurent/swin_tiny_patch4_window7_224.CTransPath"
    print(f"  Loading {repo} from HuggingFace Hub...")
    weights_path = hf_hub_download(repo, "model.safetensors")
    sd = load_file(weights_path)

    # Remap: checkpoint stores downsample at layers.N, timm expects it at layers.(N+1)
    remapped = {}
    for k, v in sd.items():
        new_k = re.sub(r"^layers\.(\d+)\.downsample",
                       lambda m: f"layers.{int(m.group(1)) + 1}.downsample", k)
        remapped[new_k] = v

    # Load with strict=False to skip non-parameter buffers (relative_position_index,
    # attn_mask) that are recomputed at runtime and not saved in safetensors format.
    buffer_names = {n for n, _ in backbone.named_buffers()}
    missing, unexpected = backbone.load_state_dict(remapped, strict=False)
    real_missing = [k for k in missing if k not in buffer_names]
    if real_missing:
        print(f"  WARNING: genuinely missing keys: {real_missing}")
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected}")

    return backbone


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
    "ctranspath": _ModelConfig(
        repo="1aurent/swin_tiny_patch4_window7_224.CTransPath",
        dim=768,
        loader="timm",
    ),
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
        mlp_hidden: int = None,
    ):
        self._config = config
        # super().__init__ calls build_backbone(), so _config must be set first
        super().__init__(
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            mlp_hidden=mlp_hidden,
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

        if cfg.repo == "1aurent/swin_tiny_patch4_window7_224.CTransPath":
            return _build_ctranspath_backbone()

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
    mlp_hidden: int = None,
) -> FoundationModel:
    """
    Instantiate a histology foundation model.

    Args:
        name: model key (ctranspath, gigapath, hoptimus, uni, uni2, phikon, phikon2, virchow, virchow2)
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
        mlp_hidden=mlp_hidden,
    )
    print(f"Loaded {model}")
    return model
