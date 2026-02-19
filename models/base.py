"""
Abstract base for all models in this project.

Each model is a backbone + classifier head. The backbone is frozen or
fine-tuned; the head is always trained from scratch.

Subclasses implement:
  - build_backbone() → nn.Module  (returns feature extractor)
  - feature_dim     → int         (output dim of backbone)
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional


class HistoBaseModel(nn.Module, ABC):
    """
    Abstract backbone + linear classifier head.

    Args:
        num_classes: number of output classes
        dropout: dropout before the classifier head (0 = disabled)
        freeze_backbone: if True, backbone weights are frozen for fast head-only training
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self._freeze_backbone = freeze_backbone

        self.backbone = self.build_backbone()

        # Simple linear head; easy to swap for MLP
        self.classifier = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.feature_dim, num_classes),
        )

        if freeze_backbone:
            self._freeze()

    @abstractmethod
    def build_backbone(self) -> nn.Module:
        """Return the backbone (without classification head)."""
        ...

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimension of backbone output features."""
        ...

    def _freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        print(f"  Backbone frozen — only training classifier head.")

    def unfreeze_backbone(self, layers_from_end: Optional[int] = None):
        """
        Unfreeze backbone for fine-tuning.

        Args:
            layers_from_end: if given, only unfreeze the last N children of backbone.
                             None = unfreeze everything.
        """
        if layers_from_end is None:
            for p in self.backbone.parameters():
                p.requires_grad_(True)
        else:
            children = list(self.backbone.children())
            for child in children[-layers_from_end:]:
                for p in child.parameters():
                    p.requires_grad_(True)
        print(f"  Backbone (partially) unfrozen.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled backbone features (useful for embeddings / probing)."""
        return self.backbone(x)

    def param_groups(self, backbone_lr: float = 1e-4, head_lr: float = 1e-3):
        """
        Differential learning rates: smaller LR for backbone, larger for head.
        Pass to optimizer as parameter groups.

        Example:
            optimizer = Adam(model.param_groups(backbone_lr=1e-5, head_lr=1e-3))
        """
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.classifier.parameters(), "lr": head_lr},
        ]

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"classes={self.num_classes}, "
            f"feature_dim={self.feature_dim}, "
            f"trainable_params={self.count_trainable_params():,})"
        )