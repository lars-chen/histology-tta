"""
Train a patch-level linear probe (nn.Linear) on the 50 fully-annotated
Camelyon17 slides using pre-extracted UNI D4 features.

Input:  view-0 patch embeddings (1024-d) + pixel-level tumor annotations (ASAP XML)
Output: checkpoints/camelyon17/patch_probe_uni_seed<seed>/best.pt

Split: 40 slides train / 10 slides val (random, seed-controlled)

Usage:
  python -m camelyon17.train_patch_probe
  python -m camelyon17.train_patch_probe --epochs 30 --seed 0
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from camelyon17.annot_utils import parse_lesion_polygons, patch_labels

_ANNOT_DIR  = Path("/gpfs/data/oermannlab/public_data/camelyon17/training/lesion_annotations")
_DATA_ROOT  = Path("/gpfs/data/mankowskilab/chen/camelyon17_patched")
_FEAT_ROOT  = Path("camelyon17_features/uni/d4_all")
_CKPT_ROOT  = Path("checkpoints/camelyon17")


# ---------------------------------------------------------------------------
# Per-slide data loading
# ---------------------------------------------------------------------------

def load_slide(
    slide_id: str,
    features_dir: Path,
    patches_dir: Path,
    patch_size_lv0: int,
) -> tuple[torch.Tensor, np.ndarray] | None:
    """Return (view0_features, labels) for one slide, or None if files missing."""
    feat_path = features_dir / f"{slide_id}.pt"
    h5_path   = patches_dir  / f"{slide_id}.h5"
    xml_path  = _ANNOT_DIR   / f"{slide_id}.xml"
    if not feat_path.exists() or not h5_path.exists():
        return None

    feat = torch.load(feat_path, weights_only=True)   # (N, 8, d)
    if feat.ndim == 2:
        feat = feat.unsqueeze(1)
    feat_v0 = feat[:, 0, :].float()                   # (N, d) — free full tensor immediately
    del feat

    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]
    tumor_region = parse_lesion_polygons(xml_path) if xml_path.exists() else None
    labels = patch_labels(coords, patch_size_lv0, tumor_region)
    return feat_v0, labels


# ---------------------------------------------------------------------------
# Training — slide-level streaming: one slide in RAM at a time
# ---------------------------------------------------------------------------

def run_epoch_train(
    slide_ids: list[str],
    features_dir: Path,
    patches_dir: Path,
    patch_size_lv0: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_weights: torch.Tensor,
    batch_size: int,
    device: torch.device,
    rng: random.Random,
) -> float:
    model.train()
    total_loss = total_n = 0
    for sid in rng.sample(slide_ids, len(slide_ids)):   # shuffle slide order
        result = load_slide(sid, features_dir, patches_dir, patch_size_lv0)
        if result is None:
            continue
        X, y = result
        y = torch.from_numpy(y).long()

        # balanced sampling within slide
        n_pos = int(y.sum()); n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            w = torch.ones(len(y))
        else:
            w = torch.where(y == 1, torch.tensor(1.0 / n_pos), torch.tensor(1.0 / n_neg))
        sampler = WeightedRandomSampler(w, num_samples=len(y), replacement=True)
        loader  = DataLoader(TensorDataset(X, y), batch_size=batch_size, sampler=sampler)

        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(X_b), y_b, weight=loss_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_b)
            total_n    += len(y_b)
        del X, y   # free before loading next slide

    return total_loss / max(total_n, 1)


@torch.no_grad()
def run_epoch_val(
    slide_ids: list[str],
    features_dir: Path,
    patches_dir: Path,
    patch_size_lv0: int,
    model: nn.Module,
    batch_size: int,
    device: torch.device,
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
    if args.features_dir is None:
        args.features_dir = Path(f"camelyon17_features/{args.model}/d4_all")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    all_slides = sorted(p.stem for p in _ANNOT_DIR.glob("*.xml")
                        if (args.features_dir / f"{p.stem}.pt").exists())

    patient_to_slides: dict[str, list[str]] = defaultdict(list)
    for s in all_slides:
        patient_to_slides["_".join(s.split("_")[:2])].append(s)
    patients = sorted(patient_to_slides.keys())
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * args.val_frac))
    val_patients = set(patients[:n_val])
    val_ids = [s for s in all_slides if "_".join(s.split("_")[:2]) in val_patients]
    tr_ids  = [s for s in all_slides if "_".join(s.split("_")[:2]) not in val_patients]
    print(f"Train: {len(tr_ids)} slides ({len(patients)-n_val} patients) | "
          f"Val: {len(val_ids)} slides ({n_val} patients)")

    # infer embedding dim from first available slide
    first = torch.load(args.features_dir / f"{all_slides[0]}.pt", weights_only=True)
    embed_dim = first.shape[-1]
    del first

    model     = nn.Linear(embed_dim, 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # global class imbalance estimate (just count labels, no feature loading)
    n_pos = n_neg = 0
    for sid in tr_ids:
        h5 = args.patches_dir / f"{sid}.h5"
        xml = _ANNOT_DIR / f"{sid}.xml"
        if not h5.exists(): continue
        with h5py.File(h5, "r") as f: coords = f["coords"][:]
        tumor_region = parse_lesion_polygons(xml) if xml.exists() else None
        lbls = patch_labels(coords, args.patch_size_lv0, tumor_region)
        n_pos += int(lbls.sum()); n_neg += len(lbls) - int(lbls.sum())
    loss_weights = torch.tensor([1.0, n_neg / max(n_pos, 1)], device=device)
    print(f"Global: {n_pos:,} tumor / {n_neg:,} normal  (pos_weight={loss_weights[1]:.1f})")

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
        if auc > best_auc:
            best_auc = auc
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_auc": auc, "args": vars(args)}, out_dir / "best.pt")
            print(f"  ✓ saved best (auc={auc:.4f})")

    print(f"\nBest val AUC: {best_auc:.4f}  → {out_dir / 'best.pt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          type=str,  default="uni")
    p.add_argument("--features_dir",   type=Path, default=None)
    p.add_argument("--patches_dir",    type=Path, default=_DATA_ROOT / "patches")
    p.add_argument("--out_dir",        type=Path, default=_CKPT_ROOT)
    p.add_argument("--epochs",         type=int,  default=20)
    p.add_argument("--batch_size",     type=int,  default=2048)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--val_frac",       type=float, default=0.2)
    p.add_argument("--patch_size_lv0", type=int,  default=512)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--device",         type=str,  default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
