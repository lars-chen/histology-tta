#!/usr/bin/env python3
"""
TTA evaluation entry point.

Loads a trained checkpoint and evaluates with multiple TTA strategies,
producing a comparison table.

Performance optimizations:
  - All TTA views are batched into a single forward pass per data batch
  - The superset of all requested views is computed once; per-strategy
    results are derived by slicing the cached logits (no redundant inference)
  - Optional AMP (fp16) for faster inference on large models

Usage:
    python evaluate_tta.py \
        --model resnet50 \
        --checkpoint checkpoints/resnet50_best.pt \
        --num_classes 33 \
        --tta_strategies none flips d4 d4_color full \
        --aggregations mean vote confidence
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm

from models import get_model
from data.data import HistoDataset
from data.transforms import get_tta_transforms
from tta.aggregator import aggregate_predictions
from utils.metrics import per_class_metrics, print_per_class_table
from torch.utils.data import DataLoader


def set_seed(seed: int):
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy over the last (class) dimension. Input must sum to 1."""
    return -(probs * (probs + 1e-8).log()).sum(dim=-1)


def parse_args():
    parser = argparse.ArgumentParser(description="TTA evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a trained checkpoint")
    parser.add_argument("--dataset", type=str, default="tcga-ut",
                        help="Dataset name (tcga-ut, nct-crc-100k, nct-crc-nonorm)")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="If None, inferred from dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--tta_strategies", nargs="+",
        default=["none", "flips", "d4", "d4_color"],
        help="TTA strategies to evaluate"
    )
    parser.add_argument(
        "--aggregations", nargs="+",
        default=["mean", "vote", "confidence"],
        help="Aggregation strategies"
    )
    parser.add_argument("--no_augment", action="store_true",
                        help="Flag indicating model was trained without augmentation (for labeling outputs)")
    parser.add_argument("--amp", action="store_true",
                        help="Enable fp16 AMP for faster inference")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (submit separate jobs with different seeds for variance estimation)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Compute per-view logits for ALL unique views in one pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_all_view_logits(model, dataset, all_transforms, device, batch_size,
                            num_workers, amp=False):
    """
    Run inference once for every unique TTA view and return cached logits.

    All views within a data batch are stacked into a single forward pass
    (B*n_views, C, H, W) → (B*n_views, num_classes), then reshaped.

    Args:
        model: trained model in eval mode
        dataset: HistoDataset (transform will be temporarily set to None)
        all_transforms: list of TTATransform objects (the superset of all views)
        device: "cuda" / "cpu"
        batch_size: per-view batch size (total GPU batch = batch_size * n_views)
        num_workers: DataLoader workers
        amp: use fp16 automatic mixed precision

    Returns:
        all_logits: (n_views, N, C) tensor of logits on CPU
        all_labels: (N,) tensor of ground-truth labels on CPU
    """
    model.eval()
    n_views = len(all_transforms)
    use_amp = amp and device == "cuda"

    orig_transform = dataset.transform

    def _collate_pil(batch):
        imgs, labels = zip(*batch)
        return list(imgs), torch.tensor(labels)

    dataset.transform = None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_pil,
    )

    chunks_logits = []  # list of (n_views, B_i, C)
    chunks_labels = []

    pbar = tqdm(loader, desc=f"  inference ({n_views} views)", leave=False,
                dynamic_ncols=True)
    for images, labels in pbar:
        B = len(images)
        # Apply all transforms and stack into one big batch: (n_views * B, C, H, W)
        view_tensors = []
        for tf in all_transforms:
            view_tensors.append(torch.stack([tf(img) for img in images]))
        mega_batch = torch.cat(view_tensors, dim=0).to(device, non_blocking=True)

        # Single forward pass for all views
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            mega_logits = model(mega_batch)  # (n_views * B, C)

        # Reshape to (n_views, B, C)
        mega_logits = mega_logits.reshape(n_views, B, -1).cpu()
        chunks_logits.append(mega_logits)
        chunks_labels.append(labels)

    dataset.transform = orig_transform

    # Concatenate along the sample dimension: (n_views, N, C)
    all_logits = torch.cat(chunks_logits, dim=1)
    all_labels = torch.cat(chunks_labels)
    return all_logits, all_labels


# ---------------------------------------------------------------------------
# Derive metrics from cached logits for a given view subset + aggregation
# ---------------------------------------------------------------------------

def _compute_metrics(all_logits, all_labels, view_indices, aggregation):
    """
    Compute accuracy metrics from cached per-view logits.

    Args:
        all_logits: (n_views, N, C) full cached logits
        all_labels: (N,) ground-truth labels
        view_indices: list of int indices into the view dimension
        aggregation: aggregation strategy name

    Returns:
        dict with acc, balanced_acc, macro_f1, preds, labels, uncertainty, etc.
    """
    # Slice the views for this strategy: (n_strategy_views, N, C)
    strategy_logits = all_logits[view_indices]
    n_views = len(view_indices)

    # aggregate_predictions expects a list of (N, C) tensors
    logits_list = [strategy_logits[v] for v in range(n_views)]
    probs = aggregate_predictions(logits_list, aggregation)

    preds = probs.argmax(dim=1).numpy()
    labels = all_labels.numpy()

    n_total = len(labels)
    n_correct = int((preds == labels).sum())
    n_wrong = n_total - n_correct

    # Uncertainty decomposition (Gal & Ghahramani 2016)
    view_probs = F.softmax(strategy_logits, dim=-1)   # (n_views, N, C)
    mean_view_probs = view_probs.mean(dim=0)           # (N, C)
    total_unc = _entropy(mean_view_probs).mean().item()
    aleatoric_unc = _entropy(view_probs).mean(dim=0).mean().item()
    epistemic_unc = total_unc - aleatoric_unc

    # View agreement
    view_preds = view_probs.argmax(dim=-1)             # (n_views, N)
    agreement_rate = (view_preds == view_preds[0:1]).all(dim=0).float().mean().item()

    return {
        "acc": accuracy_score(labels, preds),
        "balanced_acc": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_total": n_total,
        "preds": preds,
        "labels": labels,
        "total_unc": total_unc,
        "aleatoric_unc": aleatoric_unc,
        "epistemic_unc": epistemic_unc,
        "agreement_rate": agreement_rate,
    }


# ---------------------------------------------------------------------------
# Build a deduplicated superset of TTA views across all strategies
# ---------------------------------------------------------------------------

def _build_view_superset(strategies):
    """
    Build a deduplicated, ordered list of all TTA views needed across all
    requested strategies.

    Returns:
        all_transforms: list of TTATransform (the superset)
        strategy_view_indices: dict mapping strategy name → list of int indices
                               into all_transforms
    """
    name_to_idx = {}       # view_name → index in all_transforms
    all_transforms = []
    strategy_view_indices = {}

    for strategy in strategies:
        transforms = get_tta_transforms(strategy)
        indices = []
        for tf in transforms:
            if tf.name not in name_to_idx:
                name_to_idx[tf.name] = len(all_transforms)
                all_transforms.append(tf)
            indices.append(name_to_idx[tf.name])
        strategy_view_indices[strategy] = indices

    return all_transforms, strategy_view_indices


def _infer_backbone_mode(ckpt_path):
    """Infer backbone mode (frozen/finetuned) from checkpoint filename."""
    stem = os.path.basename(ckpt_path)
    if "_frozen" in stem:
        return "frozen"
    elif "_finetuned" in stem:
        return "finetuned"
    return "unknown"


def _infer_no_augment(ckpt_path):
    """Infer whether model was trained without augmentation from checkpoint filename."""
    stem = os.path.basename(ckpt_path)
    return "_noaug_" in stem


def main():
    args = parse_args()
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    set_seed(args.seed)

    # Load test dataset
    test_set = HistoDataset(args.dataset, split="test", transform=None, cache_dir=args.cache_dir)
    num_classes = args.num_classes or test_set.num_classes

    backbone_mode = _infer_backbone_mode(args.checkpoint)
    no_augment = args.no_augment or _infer_no_augment(args.checkpoint)
    aug_tag = "noaug" if no_augment else "aug"

    # Load model
    model = get_model(args.model, num_classes=num_classes).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    ckpt_name = os.path.basename(args.checkpoint)
    print(f"\nLoaded {ckpt_name} (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")
    print(f"Seed: {args.seed}")
    print(f"Train augmentation: {'OFF' if no_augment else 'ON'}")
    if args.amp:
        print(f"Mixed precision: fp16 AMP enabled")

    # --- Build deduplicated view superset and run inference ONCE ---
    all_transforms, strategy_view_indices = _build_view_superset(args.tta_strategies)
    print(f"\nRunning inference for {len(all_transforms)} unique views "
          f"(deduplicated from {sum(len(v) for v in strategy_view_indices.values())} total)...")

    t0 = time.time()
    all_logits, all_labels = compute_all_view_logits(
        model=model,
        dataset=test_set,
        all_transforms=all_transforms,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
    )
    inference_time = time.time() - t0
    print(f"Inference complete: {inference_time:.1f}s for {all_logits.shape[1]:,} samples × "
          f"{all_logits.shape[0]} views")

    # --- Derive metrics for each strategy × aggregation from cached logits ---
    results = {}
    all_rows = []
    baseline_preds = None

    for strategy in args.tta_strategies:
        view_indices = strategy_view_indices[strategy]
        for agg in args.aggregations:
            if strategy == "none" and agg != "mean":
                continue
            metrics = _compute_metrics(all_logits, all_labels, view_indices, agg)
            n_views = len(view_indices)

            # Compare to the no-TTA baseline
            if strategy == "none":
                baseline_preds = metrics["preds"]
                n_corrected = n_corrupted = None
            elif baseline_preds is not None:
                labels_arr = metrics["labels"]
                tta_preds = metrics["preds"]
                baseline_wrong = baseline_preds != labels_arr
                baseline_right = ~baseline_wrong
                n_corrected = int((baseline_wrong & (tta_preds == labels_arr)).sum())
                n_corrupted = int((baseline_right & (tta_preds != labels_arr)).sum())
            else:
                n_corrected = n_corrupted = None

            row = {
                "model": args.model,
                "dataset": args.dataset,
                "backbone_mode": backbone_mode,
                "train_augment": not no_augment,
                "checkpoint": ckpt_name,
                "seed": args.seed,
                "strategy": strategy,
                "aggregation": agg,
                "acc": metrics["acc"],
                "balanced_acc": metrics["balanced_acc"],
                "macro_f1": metrics["macro_f1"],
                "n_correct": metrics["n_correct"],
                "n_wrong": metrics["n_wrong"],
                "n_total": metrics["n_total"],
                "n_views": n_views,
                "n_corrected": n_corrected,
                "n_corrupted": n_corrupted,
                "total_unc": metrics["total_unc"],
                "aleatoric_unc": metrics["aleatoric_unc"],
                "epistemic_unc": metrics["epistemic_unc"],
                "agreement_rate": metrics["agreement_rate"],
                "per_class": per_class_metrics(
                    metrics["labels"], metrics["preds"], test_set.classes
                ),
            }
            results[(strategy, agg)] = row
            all_rows.append(row)

            corrected_str = f"{n_corrected:,}" if n_corrected is not None else "—"
            corrupted_str = f"{n_corrupted:,}" if n_corrupted is not None else "—"
            print(
                f"  {strategy}/{agg}: "
                f"acc={metrics['acc']:.4f} bal={metrics['balanced_acc']:.4f} "
                f"f1={metrics['macro_f1']:.4f} "
                f"corr={corrected_str} corrupt={corrupted_str}"
            )

    # --- Print summary table ---
    W = 152
    print(f"\n{'='*W}")
    print(
        f"{'Strategy':<20} {'Agg':<12} {'Acc':>8} {'BalAcc':>10} {'MacroF1':>9} "
        f"{'Correct':>9} {'Wrong':>7} {'Views':>7} "
        f"{'Corrected':>11} {'Corrupted':>11} "
        f"{'TotUnc':>8} {'AleaUnc':>9} {'EpiUnc':>8} {'Agree':>7}"
    )
    print(f"{'='*W}")
    for (strategy, agg), row in results.items():
        n_corrected = row["n_corrected"]
        n_corrupted = row["n_corrupted"]
        corrected_str = f"{n_corrected:>11,}" if n_corrected is not None else f"{'—':>11}"
        corrupted_str = f"{n_corrupted:>11,}" if n_corrupted is not None else f"{'—':>11}"
        print(
            f"{strategy:<20} {agg:<12} {row['acc']:>8.4f} "
            f"{row['balanced_acc']:>10.4f} {row['macro_f1']:>9.4f} "
            f"{row['n_correct']:>9,} "
            f"{row['n_wrong']:>7,} {row['n_views']:>7} "
            f"{corrected_str} {corrupted_str} "
            f"{row['total_unc']:>8.4f} {row['aleatoric_unc']:>9.4f} "
            f"{row['epistemic_unc']:>8.4f} {row['agreement_rate']:>7.4f}"
        )
    print(f"{'='*W}")

    # Per-class breakdown
    for row in results.values():
        print_per_class_table(row["per_class"], row["strategy"], row["aggregation"])

    # Find best by balanced_acc
    best_key = max(results, key=lambda k: results[k]["balanced_acc"])
    print(
        f"\nBest: strategy='{best_key[0]}' agg='{best_key[1]}' "
        f"balanced_acc={results[best_key]['balanced_acc']:.4f}"
    )

    # Save results with seed included
    _SKIP = {"preds", "labels"}
    serialisable = [{k: v for k, v in r.items() if k not in _SKIP} for r in all_rows]
    out_path = f"tta_results_{args.dataset}_{args.model}_{backbone_mode}_{aug_tag}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
