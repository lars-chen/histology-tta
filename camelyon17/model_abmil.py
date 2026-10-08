"""
MIL models for Camelyon17.

MeanPoolMIL  — average all patch embeddings, then classify (simplest baseline)
ABMIL        — gated attention pooling (Ilse et al. 2018)

Both return (logits, weights) where weights is None for MeanPoolMIL and
(N,) attention scores for ABMIL, so callers can use them interchangeably.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPoolMIL(nn.Module):
    """
    Simplest MIL baseline: average patch embeddings → MLP classifier.

    z = (1/N) * sum_k h_k

    Note: D4 feature-level TTA (average embeddings → classify) and logit-level
    TTA (classify each view → average logits) are NOT equivalent here because
    the classifier contains ReLU. Both are worth evaluating.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_cls: int = 512,
        n_classes: int = 2,
        dropout_cls: float = 0.25,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_cls),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(hidden_cls, n_classes),
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Args:
            h: (N, d) patch features for a single bag
        Returns:
            logits: (n_classes,)
            weights: None (no attention weights)
        """
        z = h.mean(dim=0)
        return self.classifier(z), None

    def forward_batch(
        self,
        features_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[None]]:
        all_logits = []
        for h in features_list:
            logits, _ = self.forward(h)
            all_logits.append(logits)
        return torch.stack(all_logits), [None] * len(features_list)


class GatedAttention(nn.Module):
    """
    Gated attention pooling (single-head).

    Args:
        in_dim:    input feature dimension d
        hidden:    attention hidden dimension L (default 256)
        dropout:   dropout on attention weights (default 0.25)
    """

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(in_dim, hidden), nn.Sigmoid())
        self.w = nn.Linear(hidden, 1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (N, d) patch features for one bag

        Returns:
            z:   (d,) bag-level representation
            attn: (N,) attention weights (softmax-normalised)
        """
        e = self.V(h)                          # (N, L)
        g = self.U(h)                          # (N, L)
        a = self.w(self.drop(e * g))           # (N, 1)
        a = F.softmax(a, dim=0)                # (N, 1)
        z = (a * h).sum(dim=0)                 # (d,)
        return z, a.squeeze(1)                 # (d,), (N,)


class ABMIL(nn.Module):
    """
    Full ABMIL model: attention pooling → binary classifier.

    Args:
        in_dim:      patch feature dimension
        hidden_attn: attention hidden dim (default 256)
        hidden_cls:  classifier hidden dim (default 512)
        n_classes:   number of output classes (default 2 for binary)
        dropout_attn: dropout inside gated attention
        dropout_cls:  dropout before classifier
    """

    def __init__(
        self,
        in_dim: int,
        hidden_attn: int = 256,
        hidden_cls: int = 512,
        n_classes: int = 2,
        dropout_attn: float = 0.25,
        dropout_cls: float = 0.25,
    ):
        super().__init__()
        self.attention = GatedAttention(in_dim, hidden_attn, dropout_attn)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_cls),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(hidden_cls, n_classes),
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (N, d) — patch features for a single bag

        Returns:
            logits: (n_classes,) raw logits
            attn:   (N,) attention weights
        """
        z, attn = self.attention(h)   # (d,), (N,)
        logits = self.classifier(z)   # (n_classes,)
        return logits, attn

    def forward_batch(
        self,
        features_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Process a batch of bags (variable N per bag).

        Returns:
            logits: (B, n_classes)
            attns:  list of (N_i,) tensors
        """
        all_logits = []
        all_attns = []
        for h in features_list:
            logits, attn = self.forward(h)
            all_logits.append(logits)
            all_attns.append(attn)
        return torch.stack(all_logits), all_attns


def build_mil_model(
    mil_type: str,
    in_dim: int,
    hidden_attn: int = 256,
    hidden_cls: int = 512,
    dropout: float = 0.25,
    n_classes: int = 2,
) -> "MeanPoolMIL | ABMIL":
    if mil_type == "mean_pool":
        return MeanPoolMIL(in_dim, hidden_cls, n_classes, dropout)
    elif mil_type == "abmil":
        return ABMIL(in_dim, hidden_attn, hidden_cls, n_classes, dropout, dropout)
    else:
        raise ValueError(f"Unknown mil_type '{mil_type}'. Choose: mean_pool, abmil")
