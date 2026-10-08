"""
MIL bag dataset for Camelyon17 (slide-level binary classification).

Bag = one slide. Label = 0 (negative) or 1 (positive).
Features are pre-extracted tensors saved by extract_features.py.

Supported feature shapes per slide:
  (N, d)    — single embedding per patch (no TTA or pre-averaged)
  (N, 8, d) — all 8 D4 views per patch (for bag-level TTA at inference)

The dataset always returns (features, label, slide_id). The caller
(trainer or evaluator) decides how to handle the feature shape.
"""

from __future__ import annotations

import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Standard Camelyon17 domain-generalization split (matches WILDS benchmark):
#   train: centers 0-3  |  val: center 3 (held-out patients)  |  test: center 4
# Center 4 is a different scanner/lab — tests generalization to unseen domain.
_TEST_CENTER  = 4
_VAL_CENTER   = 3   # last 20% of center 3 patients used as val


_LABEL_MAP = {"negative": 0, "macro": 1, "micro": 1, "itc": 1}


def _load_labels(stage_labels_csv: Path) -> dict[str, int]:
    """Return {slide_id: label} for all slides with known binary labels."""
    df = pd.read_csv(stage_labels_csv)
    # Drop patient-level rows (those have .zip in patient column or no node)
    df = df[df["patient"].str.endswith(".tif")]
    labels = {}
    for _, row in df.iterrows():
        slide_id = Path(row["patient"]).stem   # "patient_000_node_0.tif" → "patient_000_node_0"
        label_str = row["stage"].strip().lower()
        if label_str in _LABEL_MAP:
            labels[slide_id] = _LABEL_MAP[label_str]
    return labels


class CamelyonSlideDataset(Dataset):
    """
    Args:
        features_dir: directory containing <slide_id>.pt files
        stage_labels_csv: path to stage_labels.csv
        slide_ids: list of slide IDs to include (enables train/val/test splits)
    """

    def __init__(
        self,
        features_dir: Path,
        stage_labels_csv: Path,
        slide_ids: list[str] | None = None,
    ):
        self.features_dir = Path(features_dir)
        all_labels = _load_labels(stage_labels_csv)

        if slide_ids is None:
            slide_ids = [p.stem for p in sorted(self.features_dir.glob("*.pt"))]

        self.slide_ids = [s for s in slide_ids if s in all_labels and
                          (self.features_dir / f"{s}.pt").exists()]
        self.labels = {s: all_labels[s] for s in self.slide_ids}

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int):
        slide_id = self.slide_ids[idx]
        features = torch.load(
            self.features_dir / f"{slide_id}.pt",
            weights_only=True,
        )   # (N, d) or (N, 8, d)
        label = self.labels[slide_id]
        return features, torch.tensor(label, dtype=torch.long), slide_id


def collate_bags(batch):
    """
    Custom collate: bags have variable N, so return lists not stacked tensors.

    Returns:
        features_list: list of (N_i, d) or (N_i, 8, d) tensors
        labels:        (B,) LongTensor
        slide_ids:     list of str
    """
    features_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    slide_ids = [item[2] for item in batch]
    return features_list, labels, slide_ids


def make_splits(
    features_dir: Path,
    stage_labels_csv: Path,
    slide_metadata_csv: Path,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Center-based train/val/test split matching the WILDS Camelyon17 benchmark.

    - test:  all slides from center 4 (unseen domain)
    - val:   last val_frac of patients from center 3 (stratified by label)
    - train: centers 0-2 + remaining center 3 patients

    Returns (train_ids, val_ids, test_ids) as lists of slide_id strings.
    """
    all_labels = _load_labels(stage_labels_csv)
    meta = pd.read_csv(slide_metadata_csv)

    # Available slides (have features + labels)
    available = {
        p.stem for p in Path(features_dir).glob("*.pt")
        if p.stem in all_labels
    }

    meta = meta[meta["slide_id"].isin(available)].copy()
    meta["label"] = meta["slide_id"].map(all_labels)

    # Test: center 4
    test_slides = meta[meta["center"] == _TEST_CENTER]["slide_id"].tolist()

    # Val: stratified sample of patients from center 3
    c3_patients = meta[meta["center"] == _VAL_CENTER]["patient"].unique().tolist()
    c3_labels = [
        int(meta[meta["patient"] == p]["label"].mean() >= 0.5)
        for p in c3_patients
    ]
    n_val_patients = max(1, int(len(c3_patients) * val_frac))
    pats_c3_train, pats_c3_val = train_test_split(
        c3_patients, test_size=n_val_patients,
        stratify=c3_labels, random_state=seed,
    )

    val_slides   = meta[meta["patient"].isin(pats_c3_val)]["slide_id"].tolist()
    train_slides = meta[
        (meta["center"] < _VAL_CENTER) |
        (meta["patient"].isin(pats_c3_train))
    ]["slide_id"].tolist()

    return train_slides, val_slides, test_slides
