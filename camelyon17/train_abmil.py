"""
Train MIL model on pre-extracted Camelyon17 features.

Features must be (N, 8, d) from camelyon17_features/uni/d4_all/.

--train_tta controls which representation is used during training:
  none    — view 0 only (identity transform), standard baseline
  d4_mean — mean of all 8 views per patch (bakes in D4 invariance at train time)

At eval time, evaluate_abmil.py tests all TTA modes (none/d4_mean/d4_bag/d4_both)
regardless of how the model was trained.

Checkpoints saved to:
  checkpoints/camelyon17/<mil_type>_<model>_<train_tta>_seed<seed>/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from camelyon17.dataset_mil import CamelyonSlideDataset, collate_bags, make_splits
from camelyon17.model_abmil import build_mil_model

_DATA_ROOT = Path("/gpfs/data/mankowskilab/chen/camelyon17_patched")
_FEATURE_DIMS = {
    "phikon": 768, "phikon2": 1024, "uni": 1024,
    "virchow2": 1280, "gigapath": 1536,
    "dinov2-b": 768, "dinov2-l": 1024,
    "resnet50": 2048, "convnextv2": 1024,
}


def prepare_features(feat: torch.Tensor, train_tta: str) -> torch.Tensor:
    """
    Convert (N, 8, d) bag tensor to (N, d) for MIL forward pass.
      none    → view 0 (identity transform)
      d4_mean → mean across 8 views
      d4_rand → one randomly sampled view per bag (for training only)
    Also accepts (N, d) tensors and passes them through unchanged.
    """
    if feat.ndim == 2:
        return feat
    if train_tta == "none":
        return feat[:, 0, :]
    elif train_tta == "d4_mean":
        return feat.mean(dim=1)
    elif train_tta == "d4_rand":
        v = torch.randint(0, feat.shape[1], (1,)).item()
        return feat[:, v, :]
    else:
        raise ValueError(f"Unknown train_tta '{train_tta}'")


def evaluate(model, loader, device, train_tta: str) -> tuple[float, float]:
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for features_list, labels, _ in loader:
            labels = labels.to(device)
            features_list = [prepare_features(f.to(device), train_tta) for f in features_list]
            logits, _ = model.forward_batch(features_list)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return total_loss / len(loader.dataset), auc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir",      type=Path, default=None,
                   help="Dir with <slide_id>.pt tensors of shape (N, 8, d); "
                        "defaults to camelyon17_features/<model>/d4_all")
    p.add_argument("--model",             type=str,  default="uni",
                   choices=list(_FEATURE_DIMS))
    p.add_argument("--mil_type",          type=str,  default="mean_pool",
                   choices=["mean_pool", "abmil"])
    p.add_argument("--train_tta",         type=str,  default="none",
                   choices=["none", "d4_mean", "d4_rand"],
                   help="View selection during training: none=view0, d4_mean=avg all views, d4_rand=random view per bag")
    p.add_argument("--out_dir",           type=Path, default=Path("checkpoints/camelyon17"))
    p.add_argument("--epochs",            type=int,  default=50)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--weight_decay",      type=float, default=1e-5)
    p.add_argument("--hidden_attn",       type=int,  default=256)
    p.add_argument("--hidden_cls",        type=int,  default=512)
    p.add_argument("--dropout",           type=float, default=0.25)
    p.add_argument("--seed",              type=int,  default=42)
    p.add_argument("--device",            type=str,  default="cuda")
    p.add_argument("--patience",          type=int,  default=15)
    p.add_argument("--stage_labels_csv",  type=Path,
                   default=_DATA_ROOT / "stage_labels.csv")
    p.add_argument("--slide_metadata_csv", type=Path,
                   default=_DATA_ROOT / "slide_metadata.csv")
    return p.parse_args()


def main():
    args = parse_args()
    if args.features_dir is None:
        args.features_dir = Path(f"camelyon17_features/{args.model}/d4_all")
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | mil={args.mil_type} | train_tta={args.train_tta}")

    train_ids, val_ids, test_ids = make_splits(
        args.features_dir, args.stage_labels_csv, args.slide_metadata_csv, seed=args.seed,
    )
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    train_ds = CamelyonSlideDataset(args.features_dir, args.stage_labels_csv, train_ids)
    val_ds   = CamelyonSlideDataset(args.features_dir, args.stage_labels_csv, val_ids)
    test_ds  = CamelyonSlideDataset(args.features_dir, args.stage_labels_csv, test_ids)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_bags, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              collate_fn=collate_bags, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                              collate_fn=collate_bags, num_workers=2, pin_memory=True)

    model = build_mil_model(
        mil_type=args.mil_type,
        in_dim=_FEATURE_DIMS[args.model],
        hidden_attn=args.hidden_attn,
        hidden_cls=args.hidden_cls,
        dropout=args.dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    run_name = f"{args.mil_type}_{args.model}_{args.train_tta}_seed{args.seed}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = 0.0
    patience_count = 0
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for features_list, labels, _ in train_loader:
            labels = labels.to(device)
            features_list = [prepare_features(f.to(device), args.train_tta) for f in features_list]
            optimizer.zero_grad()
            logits, _ = model.forward_batch(features_list)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        scheduler.step()

        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_auc = evaluate(model, val_loader, device, "none")

        improved = val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            patience_count = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_auc": val_auc, "args": vars(args)},
                       out_dir / "best.pt")
        else:
            patience_count += 1

        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train={train_loss:.4f} | val={val_loss:.4f} | "
              f"auc={val_auc:.4f}{'*' if improved else ' '} | lr={lr:.2e}")

        log_rows.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "val_auc": val_auc, "lr": lr})

        if patience_count >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    with open(out_dir / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader(); w.writerows(log_rows)

    ckpt = torch.load(out_dir / "best.pt", weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_loss, test_auc = evaluate(model, test_loader, device, "none")
    print(f"\nTest — loss={test_loss:.4f}  AUC={test_auc:.4f}")

    with open(out_dir / "result.json", "w") as f:
        json.dump({
            "mil_type": args.mil_type, "model": args.model,
            "train_tta": args.train_tta, "seed": args.seed,
            "best_val_auc": best_val_auc, "test_auc": test_auc,
        }, f, indent=2)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
