"""
Train a patch-level linear probe on Radboud PANDA slides.

Input:  view-0 UNI embeddings (1024-d) + pixel-level Radboud mask annotations
Output: checkpoints/panda/patch_probe_<model>_seed<seed>/best.pt

Label:  binary — 1 = cancer (GG3/4/5), 0 = non-cancer (background/stroma/benign)
Split:  patient-stratified 80/20 train/val (Radboud only)

Usage:
  python -m panda.train_patch_probe
  python -m panda.train_patch_probe --epochs 20 --seed 42
"""

from __future__ import annotations
import os

import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from panda.annot_utils import patch_labels

_TRAIN_CSV  = Path(os.path.join(os.environ.get("PANDA_DIR", "data/panda"), "train.csv"))
_PATCHES_DIR = Path(os.path.join(os.environ.get("PANDA_DIR", "data/panda"), "clam_patches/patches"))
_FEAT_ROOT  = Path("panda_features")
_CKPT_ROOT  = Path("checkpoints/panda")


def load_slide(
    slide_id: str,
    features_dir: Path,
    patches_dir: Path,
    patch_size_lv0: int,
) -> tuple[torch.Tensor, np.ndarray] | None:
    feat_path = features_dir / f"{slide_id}.pt"
    h5_path   = patches_dir  / f"{slide_id}.h5"
    if not feat_path.exists() or not h5_path.exists():
        return None

    feat = torch.load(feat_path, weights_only=True)
    if feat.ndim == 2:
        feat = feat.unsqueeze(1)
    feat_v0 = feat[:, 0, :].float()
    del feat

    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]

    labels = patch_labels(slide_id, coords, patch_size_lv0)
    return feat_v0, labels


def run_epoch_train(
    slide_ids, features_dir, patches_dir, patch_size_lv0,
    model, optimizer, loss_weights, batch_size, device, rng,
) -> float:
    model.train()
    total_loss = total_n = 0
    for sid in rng.sample(slide_ids, len(slide_ids)):
        result = load_slide(sid, features_dir, patches_dir, patch_size_lv0)
        if result is None:
            continue
        X, y = result
        y = torch.from_numpy(y).long()
        n_pos = int(y.sum()); n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            w = torch.ones(len(y))
        else:
            w = torch.where(y == 1, torch.tensor(1.0 / n_pos),
                            torch.tensor(1.0 / n_neg))
        sampler = WeightedRandomSampler(w, num_samples=len(y), replacement=True)
        loader  = DataLoader(TensorDataset(X, y), batch_size=batch_size,
                             sampler=sampler)
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(X_b), y_b, weight=loss_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_b)
            total_n    += len(y_b)
        del X, y
    return total_loss / max(total_n, 1)


@torch.no_grad()
def run_epoch_val(
    slide_ids, features_dir, patches_dir, patch_size_lv0,
    model, batch_size, device,
) -> tuple[float, float]:
    model.eval()
    probs_all, trues_all = [], []
    for sid in slide_ids:
        result = load_slide(sid, features_dir, patches_dir, patch_size_lv0)
        if result is None:
            continue
        X, y = result
        loader = DataLoader(TensorDataset(X), batch_size=batch_size * 4)
        for (X_b,) in loader:
            p = F.softmax(model(X_b.to(device)), dim=-1)[:, 1].cpu().numpy()
            probs_all.append(p)
        trues_all.append(y)
        del X, y
    probs = np.concatenate(probs_all)
    trues = np.concatenate(trues_all)
    auc = roc_auc_score(trues, probs) if len(np.unique(trues)) > 1 else float("nan")
    acc = ((probs >= 0.5).astype(int) == trues).mean()
    return auc, acc


def train(args):
    import pandas as pd
    from collections import defaultdict

    if args.features_dir is None:
        args.features_dir = Path(f"panda_features/{args.model}/d4_all")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(_TRAIN_CSV)
    radboud = set(df[df.data_provider == "radboud"]["image_id"].tolist())
    print(f"Radboud slides in CSV: {len(radboud)}")
    # Use slide list file if pre-cached, otherwise scan (slow on GPFS)
    slide_list_path = args.features_dir / "slide_list.txt"
    if slide_list_path.exists():
        feat_ids = set(slide_list_path.read_text().splitlines())
    else:
        print("Scanning feature dir (first run, will be slow)...")
        feat_ids = {p.stem for p in args.features_dir.glob("*.pt")}
        slide_list_path.write_text("\n".join(sorted(feat_ids)))
    patch_list_path = args.patches_dir / "slide_list.txt"
    if patch_list_path.exists():
        patch_ids = set(patch_list_path.read_text().splitlines())
    else:
        patch_ids = {p.stem for p in args.patches_dir.glob("*.h5")}
        patch_list_path.write_text("\n".join(sorted(patch_ids)))
    available = sorted(radboud & feat_ids & patch_ids)
    print(f"Available Radboud slides: {len(available)}")

    # Patient-stratified split (slide ID prefix = patient ID for PANDA)
    # PANDA slide IDs are random hashes — no patient grouping in filenames.
    # Use random split instead.
    rng.shuffle(available)
    n_val   = max(1, int(len(available) * args.val_frac))
    val_ids = available[:n_val]
    tr_ids  = available[n_val:]
    print(f"Train: {len(tr_ids)}  Val: {len(val_ids)}")

    # Infer embed dim
    first = torch.load(args.features_dir / f"{available[0]}.pt", weights_only=True)
    embed_dim = first.shape[-1]
    del first

    model     = nn.Linear(embed_dim, 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=args.epochs)

    # PANDA Radboud: ~25-30% cancer patches — use fixed pos_weight
    loss_weights = torch.tensor([1.0, 3.0], device=device)
    print(f"Using fixed pos_weight={loss_weights[1]:.1f}")

    out_dir = args.out_dir / f"patch_probe_{args.model}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        avg_loss = run_epoch_train(tr_ids, args.features_dir, args.patches_dir,
                                   args.patch_size_lv0, model, optimizer,
                                   loss_weights, args.batch_size, device, rng)
        scheduler.step()
        auc, acc = run_epoch_val(val_ids, args.features_dir, args.patches_dir,
                                 args.patch_size_lv0, model, args.batch_size, device)
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  "
              f"val_auc={auc:.4f}  val_acc={acc:.4f}")
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_auc": auc, "args": vars(args)},
                       out_dir / "best.pt")
            print(f"  ✓ saved best (auc={auc:.4f})")

    print(f"\nBest val AUC: {best_auc:.4f}  → {out_dir / 'best.pt'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          type=str,  default="uni")
    p.add_argument("--features_dir",   type=Path, default=None)
    p.add_argument("--patches_dir",    type=Path, default=_PATCHES_DIR)
    p.add_argument("--out_dir",        type=Path, default=_CKPT_ROOT)
    p.add_argument("--epochs",         type=int,  default=20)
    p.add_argument("--batch_size",     type=int,  default=2048)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--val_frac",       type=float, default=0.2)
    p.add_argument("--patch_size_lv0", type=int,  default=256)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--device",         type=str,  default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
