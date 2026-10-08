"""
MIL bag dataset for PANDA (slide-level ISUP grade 0-5 classification).

Bag = one slide. Label = ISUP grade (0-5, 6 classes).
Features are pre-extracted (N, 8, 1024) tensors from panda_features/uni/d4_all/.

Splits: stratified 5-fold CV by ISUP grade.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

_N_FOLDS = 5


class PandaSlideDataset(Dataset):
    def __init__(self, features_dir: Path, slide_ids: list[str], labels: dict[str, int]):
        self.features_dir = Path(features_dir)
        self.slide_ids = [s for s in slide_ids
                          if (self.features_dir / f"{s}.pt").exists()]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int):
        slide_id = self.slide_ids[idx]
        features = torch.load(self.features_dir / f"{slide_id}.pt", weights_only=True)
        label = self.labels[slide_id]
        return features, torch.tensor(label, dtype=torch.long), slide_id


def collate_bags(batch):
    features_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    slide_ids = [item[2] for item in batch]
    return features_list, labels, slide_ids


def make_folds(
    features_dir: Path,
    train_csv: Path,
    n_folds: int = _N_FOLDS,
    seed: int = 42,
) -> list[tuple[list[str], list[str]]]:
    """
    Returns list of (train_ids, val_ids) for each fold.
    Stratified by ISUP grade. Only includes slides with extracted features.
    """
    df = pd.read_csv(train_csv)
    feat_dir = Path(features_dir)
    available = {p.stem for p in feat_dir.glob("*.pt")}
    df = df[df["image_id"].isin(available)].reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(df["image_id"], df["isup_grade"]):
        train_ids = df.iloc[train_idx]["image_id"].tolist()
        val_ids   = df.iloc[val_idx]["image_id"].tolist()
        folds.append((train_ids, val_ids))
    return folds


def load_labels(train_csv: Path) -> dict[str, int]:
    df = pd.read_csv(train_csv)
    return dict(zip(df["image_id"], df["isup_grade"]))
