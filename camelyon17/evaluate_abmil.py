"""
Evaluate a trained ABMIL on Camelyon17 test slides with optional D4 TTA.

TTA modes at inference time (independent of which features_dir is given):
  none      — single forward pass with stored features
  d4_mean   — if features are (N, 8, d): average 8 views → (N, d) → ABMIL
  d4_bag    — if features are (N, 8, d): run ABMIL 8× (one view at a time), average logits
  d4_both   — d4_mean features + 8× ABMIL prediction averaging

Usage:
  python -m camelyon17.evaluate_abmil \
    --checkpoint   checkpoints/camelyon17/abmil_phikon2_none_seed42/best.pt \
    --features_dir camelyon17_features/phikon2/d4_all \
    --tta_eval     none d4_mean d4_bag \
    --out_csv      results/camelyon17_tta.csv
"""

from __future__ import annotations
import os

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from camelyon17.dataset_mil import CamelyonSlideDataset, collate_bags, make_splits
from camelyon17.model_abmil import build_mil_model
from torch.utils.data import DataLoader

_DATA_ROOT = Path(os.environ.get("CAMELYON17_PATCHED_DIR", "data/camelyon17_patched"))
_FEATURE_DIMS = {
    "phikon": 768, "phikon2": 1024, "uni": 1024,
    "virchow2": 1280, "gigapath": 1536,
    "dinov2-b": 768, "dinov2-l": 1024,
    "resnet50": 2048, "convnextv2": 1024,
}


def _infer_feature_shape(features_dir: Path) -> str:
    """Return '2d' (N,d) or '3d' (N,8,d) by inspecting the first .pt file."""
    for pt in Path(features_dir).glob("*.pt"):
        t = torch.load(pt, weights_only=True)
        return "3d" if t.ndim == 3 else "2d"
    return "2d"


@torch.no_grad()
def eval_mode(
    model: ABMIL,
    loader: DataLoader,
    device: torch.device,
    tta: str,
    feat_shape: str,
) -> dict:
    """
    Run inference under a given TTA strategy.

    tta='none':    use features as-is (N,d) or if (N,8,d) use view 0 only
    tta='d4_mean': if (N,8,d) average 8 views → (N,d) → ABMIL
    tta='d4_bag':  if (N,8,d) run ABMIL 8× → average logits
    tta='d4_both': d4_mean features AND d4_bag average
    """
    model.eval()
    all_probs, all_labels = [], []

    for features_list, labels, _ in loader:
        labels = labels.to(device)

        for feat, lbl in zip(features_list, labels):
            feat = feat.to(device)

            if feat.ndim == 3 and tta in ("d4_mean", "d4_both"):
                # Average 8 views → (N, d)
                feat_input = feat.mean(dim=1)
            elif feat.ndim == 3 and tta == "none":
                feat_input = feat[:, 0, :]   # view 0 only
            else:
                feat_input = feat            # already (N, d)

            if tta in ("d4_bag", "d4_both") and feat.ndim == 3:
                # Run ABMIL once per view, average logits
                logits_views = []
                for v in range(feat.shape[1]):
                    view_feat = feat[:, v, :]   # (N, d)
                    logits_v, _ = model(view_feat)
                    logits_views.append(logits_v)
                logits = torch.stack(logits_views).mean(dim=0)   # (n_classes,)
            else:
                logits, _ = model(feat_input)

            prob = torch.softmax(logits, dim=-1)[1].item()
            all_probs.append(prob)
            all_labels.append(lbl.item())

    preds = [int(p > 0.5) for p in all_probs]
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    acc = accuracy_score(all_labels, preds)
    f1  = f1_score(all_labels, preds, zero_division=0)
    return {"auc": auc, "acc": acc, "f1": f1, "n": len(all_labels)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   type=Path, required=True)
    p.add_argument("--features_dir", type=Path, required=True,
                   help="Should match the shape needed for --tta_eval modes")
    p.add_argument("--tta_eval",     nargs="+", default=["none", "d4_mean", "d4_bag"],
                   choices=["none", "d4_mean", "d4_bag", "d4_both"])
    p.add_argument("--out_csv",      type=Path, default=Path("results/camelyon17_tta.csv"))
    p.add_argument("--device",       type=str,  default="cuda")
    p.add_argument("--stage_labels_csv", type=Path,
                   default=_DATA_ROOT / "stage_labels.csv")
    p.add_argument("--slide_metadata_csv", type=Path,
                   default=_DATA_ROOT / "slide_metadata.csv")
    p.add_argument("--seed",         type=int,  default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    saved_args = ckpt.get("args", {})
    model_name = saved_args.get("model", "phikon2")
    mil_type   = saved_args.get("mil_type", "mean_pool")

    model = build_mil_model(
        mil_type=mil_type,
        in_dim=_FEATURE_DIMS[model_name],
        hidden_attn=saved_args.get("hidden_attn", 256),
        hidden_cls=saved_args.get("hidden_cls", 512),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch','?')}, "
          f"val_auc={ckpt.get('val_auc', '?'):.4f}")

    # Same test split as training
    _, _, test_ids = make_splits(
        args.features_dir, args.stage_labels_csv, args.slide_metadata_csv, seed=args.seed,
    )
    test_ds = CamelyonSlideDataset(args.features_dir, args.stage_labels_csv, test_ids)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=collate_bags, num_workers=2)

    feat_shape = _infer_feature_shape(args.features_dir)
    print(f"Feature shape: {feat_shape}  |  {len(test_ds)} test slides")

    rows = []
    for tta in args.tta_eval:
        metrics = eval_mode(model, test_loader, device, tta, feat_shape)
        print(f"TTA={tta:12s}  AUC={metrics['auc']:.4f}  "
              f"Acc={metrics['acc']:.4f}  F1={metrics['f1']:.4f}  n={metrics['n']}")
        rows.append({
            "mil_type": mil_type, "model": model_name,
            "train_tta": saved_args.get("train_tta", "none"),
            "tta_eval": tta, "seed": args.seed, **metrics,
        })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out_csv.exists()
    with open(args.out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"Appended to {args.out_csv}")


if __name__ == "__main__":
    main()
