"""
TTA prediction aggregation.

Given a list of softmax probability tensors (one per augmentation),
aggregate them into a single prediction.

Strategies:
  mean       — average probabilities (works best in practice)
  logit_mean — average logits then softmax (avoids Jensen's inequality inflation)
  max        — take the maximum probability per class
  vote       — hard majority vote on argmax
  confidence — weight by max confidence of each view
"""

import torch
import torch.nn.functional as F
from typing import List


def aggregate_predictions(
    logits_list: List[torch.Tensor],
    strategy: str = "mean",
) -> torch.Tensor:
    """
    Aggregate TTA logits/probabilities into a single prediction.

    Args:
        logits_list: list of (B, C) tensors, one per TTA augmentation
        strategy: "mean" | "logit_mean" | "max" | "vote" | "confidence"

    Returns:
        (B, C) aggregated probabilities (sum to 1 for mean/confidence/vote)
        or (B, C) class counts for "vote"
    """
    # Stack: (n_tta, B, C)
    logits = torch.stack(logits_list, dim=0)
    probs = F.softmax(logits, dim=-1)   # (n_tta, B, C)

    if strategy == "mean":
        return probs.mean(dim=0)         # (B, C)

    elif strategy == "logit_mean":
        return F.softmax(logits.mean(dim=0), dim=-1)  # (B, C)

    elif strategy == "max":
        return probs.max(dim=0).values   # (B, C)

    elif strategy == "vote":
        # Hard vote: argmax per augmentation, then count votes per class
        votes = probs.argmax(dim=-1)     # (n_tta, B)
        n_classes = logits.shape[-1]
        batch_size = logits.shape[1]
        vote_counts = torch.zeros(batch_size, n_classes, device=logits.device)
        for c in range(n_classes):
            vote_counts[:, c] = (votes == c).float().sum(dim=0)
        return vote_counts / (vote_counts.sum(dim=-1, keepdim=True) + 1e-8)  # normalize

    elif strategy == "confidence":
        # Weight each view by its max confidence
        max_conf = probs.max(dim=-1).values      # (n_tta, B)
        weights = max_conf / max_conf.sum(dim=0, keepdim=True)  # (n_tta, B)
        weights = weights.unsqueeze(-1)           # (n_tta, B, 1)
        return (probs * weights).sum(dim=0)       # (B, C)

    else:
        raise ValueError(
            f"Unknown aggregation strategy '{strategy}'. "
            "Choose from: mean, logit_mean, max, vote, confidence"
        )


class TTAPredictor:
    """
    Runs TTA inference for a model on a single batch or full dataloader.

    Usage:
        from tta.aggregator import TTAPredictor
        from data.transforms import get_tta_transforms

        tta_transforms = get_tta_transforms("d4")
        predictor = TTAPredictor(model, tta_transforms, device="cuda")

        # Single batch (raw PIL images)
        probs = predictor.predict_batch(pil_images)

        # Full dataloader (returns all logits + labels)
        all_probs, all_labels = predictor.predict_loader(raw_loader)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tta_transforms: list,
        aggregation: str = "mean",
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tta_transforms = tta_transforms
        self.aggregation = aggregation
        self.device = device

    @torch.no_grad()
    def predict_batch(self, pil_images: list) -> torch.Tensor:
        """
        Run TTA on a list of PIL images.

        Returns:
            (B, C) aggregated probability tensor
        """
        logits_list = []
        for transform in self.tta_transforms:
            batch = torch.stack([transform(img) for img in pil_images]).to(self.device)
            logits_list.append(self.model(batch))
        return aggregate_predictions(logits_list, self.aggregation)

    @torch.no_grad()
    def predict_loader(self, raw_loader, verbose: bool = True):
        """
        Run TTA on a DataLoader that returns (PIL images, labels).

        NOTE: the DataLoader must NOT apply any tensor transform (i.e., use
        the raw PIL dataset). The TTAPredictor applies each TTA transform itself.

        Returns:
            all_probs: (N, C) tensor of aggregated probabilities
            all_labels: (N,) tensor of ground-truth labels
        """
        all_probs = []
        all_labels = []

        n = len(self.tta_transforms)
        for i, (images, labels) in enumerate(raw_loader):
            # images: list of PIL images (batch)
            probs = self.predict_batch(images)
            all_probs.append(probs.cpu())
            all_labels.append(labels)
            if verbose and (i + 1) % 20 == 0:
                print(f"  Batch {i+1}/{len(raw_loader)} [{n} TTA views each]")

        return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)