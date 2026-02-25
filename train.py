#!/usr/bin/env python3
"""
Training entry point.

Examples:
    # Fast linear probe with frozen backbone (train only classifier head)
    python train.py --model resnet18 --freeze_backbone --epochs 5 --lr 1e-3

    # Fine-tune ResNet-50
    python train.py --model resnet50 --epochs 20 --lr 5e-4

    # DINOv2-B linear probe (very fast, great accuracy)
    python train.py --model dinov2_b --freeze_backbone --epochs 5 --lr 1e-3

    # Full EfficientNet-B7 fine-tune
    python train.py --model efficientnet_b7 --epochs 30 --lr 1e-4

    # Train with a specific seed (for multi-seed variance, submit separate jobs)
    python train.py --model dinov2_b --freeze_backbone --seed 137
"""

import argparse
import random
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import get_model
from data.data import get_dataloaders
from data.transforms import get_train_transform, get_val_transform
from utils.trainer import train


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train histology classifier")
    parser.add_argument("--model", type=str, default="resnet18",
                        help="Model name (resnet18, resnet50, vgg16, efficientnet_b7, dinov2_b, ...)")
    parser.add_argument("--dataset", type=str, default="tcga-ut",
                        help="Dataset name (tcga-ut, nct-crc-100k, nct-crc-7k, nct-crc-nonorm)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for backbone (head uses 10x)")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone — only train classifier head")
    parser.add_argument("--unfreeze_after", type=int, default=None,
                        help="Unfreeze backbone after this many epochs (two-stage training)")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace dataset cache dir")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable training augmentation (use val transform for training)")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed-precision training (fp16 AMP) — speeds up large models")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (submit separate jobs with different seeds for variance estimation)")
    return parser.parse_args()


def train_one_seed(args, seed, device):
    """Train a single model with the given seed. Returns training history."""
    seed_everything(seed)

    # --- Transforms ---
    val_tf = get_val_transform()
    train_tf = val_tf if args.no_augment else get_train_transform()
    print(f"Training augmentation: {'OFF (using val transform)' if args.no_augment else 'ON'}")

    # --- Data ---
    train_loader, val_loader, test_loader, num_classes, class_weights = get_dataloaders(
        args.dataset,
        train_transform=train_tf,
        val_transform=val_tf,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        seed=seed,
    )

    # Checkpoint name includes seed for unique identification
    backbone_mode = "frozen" if args.freeze_backbone else "finetuned"
    aug_tag = "noaug" if args.no_augment else "aug"
    ckpt_name = f"{args.dataset}_{args.model}_{backbone_mode}_{aug_tag}_seed{seed}"

    # --- Model ---
    model = get_model(
        args.model,
        num_classes=num_classes,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # --- Optimizer: differential LRs ---
    optimizer = AdamW(
        model.param_groups(backbone_lr=args.lr, head_lr=args.lr * 10),
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Two-stage training: head → fine-tune ---
    if args.freeze_backbone and args.unfreeze_after:
        # Stage 1: train head only
        print(f"\n[Stage 1] Training head for {args.unfreeze_after} epochs...")
        history_s1 = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_epochs=args.unfreeze_after,
            device=device,
            scheduler=scheduler,
            checkpoint_dir=args.checkpoint_dir,
            model_name=f"{args.dataset}_{args.model}_frozen_{aug_tag}_stage1_seed{seed}",
            label_smoothing=args.label_smoothing,
            patience=args.patience,
            amp=args.amp,
            class_weights=class_weights,
        )
        # Stage 2: unfreeze and fine-tune
        model.unfreeze_backbone()
        optimizer = AdamW(
            model.param_groups(backbone_lr=args.lr / 10, head_lr=args.lr),
            weight_decay=1e-4,
        )
        remaining = args.epochs - args.unfreeze_after
        scheduler = CosineAnnealingLR(optimizer, T_max=remaining)
        print(f"\n[Stage 2] Fine-tuning all layers for {remaining} epochs...")
        history_s2 = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_epochs=remaining,
            device=device,
            scheduler=scheduler,
            checkpoint_dir=args.checkpoint_dir,
            model_name=f"{args.dataset}_{args.model}_finetuned_{aug_tag}_seed{seed}",
            label_smoothing=args.label_smoothing,
            patience=args.patience,
            amp=args.amp,
            class_weights=class_weights,
        )
        return history_s1 + history_s2
    else:
        return train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_epochs=args.epochs,
            device=device,
            scheduler=scheduler,
            checkpoint_dir=args.checkpoint_dir,
            model_name=ckpt_name,
            label_smoothing=args.label_smoothing,
            patience=args.patience,
            amp=args.amp,
            class_weights=class_weights,
        )


def main():
    args = parse_args()
    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    train_one_seed(args, args.seed, device)


if __name__ == "__main__":
    main()
