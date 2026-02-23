"""
Training and evaluation utilities.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    amp: bool = False,
) -> Dict[str, float]:
    model.train()
    use_amp = amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_loss = 0.0
    correct = 0
    total = 0
    t0 = time.time()

    pbar = tqdm(loader, desc="  train", leave=False, dynamic_ncols=True)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")

    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "time": time.time() - t0,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / total, "acc": correct / total}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: str,
    scheduler=None,
    checkpoint_dir: str = "checkpoints",
    model_name: str = "model",
    label_smoothing: float = 0.1,
    patience: int = 0,
    amp: bool = False,
):
    """
    Full training loop with checkpointing.

    Args:
        model: the model to train
        train_loader: training DataLoader
        val_loader: validation DataLoader
        optimizer: optimizer (use model.param_groups() for differential LR)
        num_epochs: number of training epochs
        device: "cuda" | "cpu" | "mps"
        scheduler: optional LR scheduler
        checkpoint_dir: directory to save best model
        model_name: used for checkpoint filename
        label_smoothing: cross-entropy label smoothing (helps generalization)
        patience: stop after this many epochs with no val_acc improvement
                  (0 = disabled)
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)

    best_val_acc = 0.0
    epochs_no_improve = 0
    history = []

    print(f"\n{'='*60}")
    print(f"Training {model_name} for {num_epochs} epochs on {device}")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    if patience:
        print(f"  Early stopping patience: {patience}")
    if amp:
        print(f"  Mixed precision: fp16 AMP enabled")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device, amp=amp
        )
        val_stats = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
        })

        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"train loss={train_stats['loss']:.4f} acc={train_stats['acc']:.4f} | "
            f"val loss={val_stats['loss']:.4f} acc={val_stats['acc']:.4f} | "
            f"({train_stats['time']:.1f}s)"
        )

        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            epochs_no_improve = 0
            ckpt_path = ckpt_dir / f"{model_name}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"  ✓ Saved best model (val_acc={best_val_acc:.4f}) → {ckpt_path}")
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                print(f"  Early stopping: no improvement for {patience} epochs.")
                break

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    return history