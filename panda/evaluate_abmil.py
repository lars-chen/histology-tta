"""
Evaluate a trained PANDA MIL model with all TTA strategies.

TTA modes at inference:
  none     — view 0 only (baseline)
  d4_mean  — mean of 8 patch embeddings before aggregation
  d4_bag   — run model 8 times (one per D4 view), average logits
  d4_both  — d4_mean patches + d4_bag ensemble

Results appended to results/panda_tta.csv.

Usage:
  python -m panda.evaluate_abmil \
      --checkpoint checkpoints/panda/abmil_uni_none_fold0/best.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader

from camelyon17.model_abmil import build_mil_model
from panda.dataset_mil import PandaSlideDataset, collate_bags, load_labels, make_folds

_TRAIN_CSV   = Path("/gpfs/scratch/lpc8816/panda/train.csv")
_RESULTS_CSV = Path("results/panda_tta.csv")
_N_CLASSES   = 6
_D4_VIEWS    = 8


def get_bag_logits(
    model, features_list: list[torch.Tensor],
    tta_eval: str, device: torch.device,
) -> torch.Tensor:
    """Return (B, n_classes) logits for a batch under the given TTA mode."""
    if tta_eval == "none":
        feats = [f[:, 0, :].to(device) if f.ndim == 3 else f.to(device)
                 for f in features_list]
        logits, _ = model.forward_batch(feats)
        return logits

    elif tta_eval == "d4_mean":
        feats = [f.mean(dim=1).to(device) if f.ndim == 3 else f.to(device)
                 for f in features_list]
        logits, _ = model.forward_batch(feats)
        return logits

    elif tta_eval in ("d4_bag", "d4_both"):
        all_logits = []
        for v in range(_D4_VIEWS):
            if tta_eval == "d4_bag":
                feats = [f[:, v, :].to(device) if f.ndim == 3 else f.to(device)
                         for f in features_list]
            else:  # d4_both: mean-embed then ensemble over views
                feats = [f.mean(dim=1).to(device) if f.ndim == 3 else f.to(device)
                         for f in features_list]
            logits, _ = model.forward_batch(feats)
            all_logits.append(logits)
            if tta_eval == "d4_both":
                break  # mean already encodes all views; no need to loop
        return torch.stack(all_logits).mean(dim=0)

    raise ValueError(f"Unknown tta_eval: {tta_eval}")


@torch.no_grad()
def evaluate_tta(
    model, loader, device, tta_eval: str,
) -> tuple[float, float]:
    model.eval()
    all_preds, all_labels = [], []
    for features_list, labels, _ in loader:
        logits = get_bag_logits(model, features_list, tta_eval, device)
        preds  = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    acc   = (np.array(all_preds) == np.array(all_labels)).mean()
    return kappa, acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   type=Path, required=True)
    p.add_argument("--features_dir", type=Path, default=None)
    p.add_argument("--tta_eval",     nargs="+",
                   default=["none", "d4_mean", "d4_bag", "d4_both"])
    p.add_argument("--out_csv",      type=Path, default=_RESULTS_CSV)
    p.add_argument("--batch_size",   type=int,  default=1)
    p.add_argument("--device",       type=str,  default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.checkpoint, weights_only=False, map_location=device)
    saved  = ckpt["args"]

    mil_type = saved["mil_type"]
    model_name = saved["model"]
    fold     = saved["fold"]
    seed     = saved.get("seed", 42)

    features_dir = args.features_dir or Path(f"panda_features/{model_name}/d4_all")
    labels = load_labels(_TRAIN_CSV)
    folds  = make_folds(features_dir, _TRAIN_CSV, seed=seed)
    _, val_ids = folds[fold]

    val_ds     = PandaSlideDataset(features_dir, val_ids, labels)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_bags,
                            num_workers=2)

    in_dim = ckpt["model"]["classifier.0.weight"].shape[1] \
             if "classifier.0.weight" in ckpt["model"] else \
             ckpt["model"]["attention.V.0.weight"].shape[1]
    model  = build_mil_model(mil_type, in_dim, n_classes=_N_CLASSES).to(device)
    model.load_state_dict(ckpt["model"])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out_csv.exists()
    with open(args.out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mil_type", "model", "train_tta", "tta_eval",
            "fold", "kappa", "acc", "n"])
        if write_header:
            writer.writeheader()
        for tta_eval in args.tta_eval:
            kappa, acc = evaluate_tta(model, val_loader, device, tta_eval)
            print(f"  {tta_eval:12s}  kappa={kappa:.4f}  acc={acc:.4f}")
            writer.writerow({
                "mil_type":  mil_type,
                "model":     model_name,
                "train_tta": saved.get("train_tta", "none"),
                "tta_eval":  tta_eval,
                "fold":      fold,
                "kappa":     round(kappa, 6),
                "acc":       round(acc, 4),
                "n":         len(val_ds),
            })


if __name__ == "__main__":
    main()
