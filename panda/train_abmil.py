"""
Train MIL model on pre-extracted PANDA features (ISUP grade 0-5).

Features: panda_features/uni/d4_all/<slide_id>.pt  (N, 8, 1024)
Labels:   $PANDA_DIR/train.csv    (isup_grade 0-5)

Split: stratified 5-fold CV. Each run trains on 4 folds, validates on 1.

Metrics: quadratic-weighted kappa (primary) + accuracy
Checkpoint: checkpoints/panda/<mil_type>_<model>_none_fold<fold>/best.pt

Usage:
  python -m panda.train_abmil --fold 0 --mil_type abmil
  python -m panda.train_abmil --fold 0 --mil_type mean_pool
"""

from __future__ import annotations
import os

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from camelyon17.model_abmil import build_mil_model
from panda.dataset_mil import PandaSlideDataset, collate_bags, load_labels, make_folds

_TRAIN_CSV   = Path(os.path.join(os.environ.get("PANDA_DIR", "data/panda"), "train.csv"))
_FEAT_ROOT   = Path("panda_features")
_CKPT_ROOT   = Path("checkpoints/panda")
_RESULTS_CSV = Path("results/panda_tta.csv")

_FEATURE_DIMS = {"uni": 1024}
_N_CLASSES = 6


def prepare_features(feat: torch.Tensor, train_tta: str) -> torch.Tensor:
    if feat.ndim == 2:
        return feat
    if train_tta == "none":
        return feat[:, 0, :]
    elif train_tta == "d4_mean":
        return feat.mean(dim=1)
    elif train_tta == "d4_rand":
        v = torch.randint(0, feat.shape[1], (1,)).item()
        return feat[:, v, :]
    raise ValueError(train_tta)


@torch.no_grad()
def evaluate(model, loader, device, train_tta: str) -> tuple[float, float, float]:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for features_list, labels, _ in loader:
        labels = labels.to(device)
        feats = [prepare_features(f.to(device), train_tta) for f in features_list]
        logits, _ = model.forward_batch(feats)
        total_loss += criterion(logits, labels).item() * len(labels)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    acc   = (np.array(all_preds) == np.array(all_labels)).mean()
    return total_loss / len(loader.dataset), kappa, acc


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    features_dir = args.features_dir or Path(f"panda_features/{args.model}/d4_all")
    labels = load_labels(_TRAIN_CSV)
    folds  = make_folds(features_dir, _TRAIN_CSV, seed=args.seed)

    train_ids, val_ids = folds[args.fold]
    print(f"Fold {args.fold}: {len(train_ids)} train / {len(val_ids)} val")

    train_ds = PandaSlideDataset(features_dir, train_ids, labels)
    val_ds   = PandaSlideDataset(features_dir, val_ids,   labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_bags,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_bags,
                              num_workers=2, pin_memory=True)

    in_dim = _FEATURE_DIMS.get(args.model, 1024)
    model  = build_mil_model(args.mil_type, in_dim,
                             n_classes=_N_CLASSES).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = f"{args.mil_type}_{args.model}_{args.train_tta}_fold{args.fold}"
    out_dir  = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_kappa = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        criterion  = nn.CrossEntropyLoss()
        for features_list, labels_b, _ in train_loader:
            labels_b = labels_b.to(device)
            feats = [prepare_features(f.to(device), args.train_tta)
                     for f in features_list]
            optimizer.zero_grad()
            logits, _ = model.forward_batch(feats)
            loss = criterion(logits, labels_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels_b)
        scheduler.step()

        val_loss, kappa, acc = evaluate(model, val_loader, device, "none")
        star = "*" if kappa > best_kappa else ""
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={total_loss/len(train_ds):.4f}  "
              f"val={val_loss:.4f}  kappa={kappa:.4f}{star}  acc={acc:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
        if kappa > best_kappa:
            best_kappa = kappa
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_kappa": kappa, "args": vars(args)},
                       out_dir / "best.pt")

    print(f"\nBest val kappa: {best_kappa:.4f}  → {out_dir / 'best.pt'}")
    return out_dir / "best.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        type=str,  default="uni")
    p.add_argument("--mil_type",     type=str,  default="abmil",
                   choices=["abmil", "mean_pool"])
    p.add_argument("--fold",         type=int,  default=0)
    p.add_argument("--train_tta",    type=str,  default="none",
                   choices=["none", "d4_mean", "d4_rand"])
    p.add_argument("--features_dir", type=Path, default=None)
    p.add_argument("--out_dir",      type=Path, default=_CKPT_ROOT)
    p.add_argument("--epochs",       type=int,  default=50)
    p.add_argument("--batch_size",   type=int,  default=1)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--device",       type=str,  default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
