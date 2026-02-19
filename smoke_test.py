#!/usr/bin/env python3
"""
End-to-end smoke test using synthetic data.

Exercises the full pipeline:
  1. ResNet18 model instantiation
  2. Training loop (1 epoch, tiny synthetic dataset)
  3. TTA transforms (d4 strategy)
  4. TTA evaluation loop with mean/vote aggregation

No dataset download required.

Usage:
    python smoke_test.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

from models import get_model
from data.transforms import get_train_transform, get_val_transform, get_tta_transforms
from utils.trainer import train_one_epoch, evaluate
from tta.aggregator import aggregate_predictions, TTAPredictor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_CLASSES = 9
IMG_SIZE = 224
N_TRAIN = 256
N_VAL = 64
N_TEST = 64
BATCH = 32
EPOCHS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = Path("checkpoints")


def make_fake_loader(n, batch_size, shuffle=True):
    """Random (ImageNet-normalised) tensors + integer labels."""
    images = torch.randn(n, 3, IMG_SIZE, IMG_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (n,))
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def make_pil_loader(n, batch_size):
    """Return raw PIL images (needed for TTA predictor)."""
    from PIL import Image
    import numpy as np
    from torch.utils.data import Dataset

    class FakePILDataset(Dataset):
        def __init__(self, n):
            self.n = n
            self.labels = torch.randint(0, NUM_CLASSES, (n,))

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            arr = (torch.rand(IMG_SIZE, IMG_SIZE, 3) * 255).byte().numpy()
            img = Image.fromarray(arr, mode="RGB")
            return img, int(self.labels[idx])

    def collate_pil(batch):
        imgs, labels = zip(*batch)
        return list(imgs), torch.tensor(labels)

    ds = FakePILDataset(n)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, collate_fn=collate_pil)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # ---- 1. Model ----
    print("\n[1/4] Building ResNet18 model...")
    model = get_model("resnet18", num_classes=NUM_CLASSES, dropout=0.2).to(DEVICE)
    print(f"  {model}")

    # ---- 2. Training ----
    print(f"\n[2/4] Training for {EPOCHS} epochs on {N_TRAIN} synthetic samples...")
    train_loader = make_fake_loader(N_TRAIN, BATCH, shuffle=True)
    val_loader   = make_fake_loader(N_VAL,   BATCH, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(
        model.param_groups(backbone_lr=1e-4, head_lr=1e-3),
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        scheduler.step()
        vl = evaluate(model, val_loader, criterion, DEVICE)
        print(f"  Epoch {epoch}/{EPOCHS} | "
              f"train loss={tr['loss']:.4f} acc={tr['acc']:.3f} | "
              f"val   loss={vl['loss']:.4f} acc={vl['acc']:.3f}")

    # Save checkpoint
    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_path = CKPT_DIR / "resnet18_smoke.pt"
    torch.save({
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "val_acc": vl["acc"],
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_path)
    print(f"  Checkpoint saved → {ckpt_path}")

    # ---- 3. TTA transforms ----
    print("\n[3/4] Building TTA transform sets...")
    for strategy in ["none", "flips", "d4", "d4_color"]:
        tfs = get_tta_transforms(strategy)
        print(f"  {strategy}: {len(tfs)} views")

    # ---- 4. TTA evaluation ----
    print(f"\n[4/4] Running TTA evaluation on {N_TEST} synthetic PIL samples...")
    model.eval()

    pil_loader = make_pil_loader(N_TEST, BATCH)
    results = {}
    for strategy in ["none", "d4"]:
        tta_tfs = get_tta_transforms(strategy)
        for agg in ["mean", "vote"]:
            if strategy == "none" and agg != "mean":
                continue
            predictor = TTAPredictor(model, tta_tfs, aggregation=agg, device=DEVICE)
            all_probs, all_labels = predictor.predict_loader(pil_loader, verbose=False)
            preds = all_probs.argmax(dim=1)
            acc = (preds == all_labels).float().mean().item()
            key = f"{strategy}/{agg}"
            results[key] = acc
            print(f"  [{key}]  acc={acc:.4f}  ({len(tta_tfs)} views)")

    print("\n" + "=" * 60)
    print("Smoke test PASSED — all pipeline components work correctly.")
    print("=" * 60)
    print("\nTo train on real data (TCGA-UT, 33 classes):")
    print("  python train.py --model resnet18 --epochs 10 --batch_size 64 --freeze_backbone")
    print("\nTo run TTA evaluation on a saved checkpoint:")
    print("  python evaluate_tta.py \\")
    print("      --model resnet18 \\")
    print("      --checkpoint checkpoints/resnet18_best.pt \\")
    print("      --tta_strategies none flips d4 d4_color \\")
    print("      --aggregations mean vote confidence")


if __name__ == "__main__":
    main()
