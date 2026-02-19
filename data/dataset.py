"""
Compatibility shim: re-exports from data.data under the names expected by
train.py and evaluate_tta.py.

  TCGAUTDataset          → HistoDataset  (with name="tcga-ut")
  get_tcga_ut_dataloaders → get_dataloaders("tcga-ut", ...)
"""

from typing import Callable, Optional

from .data import HistoDataset, get_dataloaders


class TCGAUTDataset(HistoDataset):
    """HistoDataset pre-configured for TCGA-UT (name='tcga-ut')."""

    def __init__(self, split="train", transform=None, streaming=False, cache_dir=None):
        super().__init__(
            name="tcga-ut",
            split=split,
            transform=transform,
            streaming=streaming,
            cache_dir=cache_dir,
        )


def get_tcga_ut_dataloaders(
    train_transform: Callable,
    val_transform: Callable,
    val_fraction: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
):
    """Convenience wrapper: calls get_dataloaders with dataset='tcga-ut'."""
    return get_dataloaders(
        "tcga-ut",
        train_transform=train_transform,
        val_transform=val_transform,
        val_fraction=val_fraction,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_dir=cache_dir,
    )
