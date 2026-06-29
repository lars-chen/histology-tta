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
        --tta_strategies none flips d4 \
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
from data.data import HistoDataset, MHISTDataset
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
        default=["none", "flips", "d4"],
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
    parser.add_argument("--save_logits", action="store_true",
                        help="Save raw per-view logits (n_views, N, C) and labels to logits/<dataset>/")
    parser.add_argument("--no_save_results", action="store_true",
                        help="Skip writing results JSON to results/raw/ (use when saving logits only)")
    parser.add_argument("--save_persample", action="store_true",
                        help="Save per-sample parquet (probe_persample format) to results/raw/")
    parser.add_argument("--selective_tta", action="store_true",
                        help="Two-pass selective TTA: run original view for all samples, then apply "
                             "d4 augmented views only to uncertain samples (entropy > threshold)")
    parser.add_argument("--selective_threshold", type=float, default=0.3,
                        help="Entropy threshold as fraction of log(C); samples above get full TTA "
                             "(default: 0.3)")
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
# Two-pass selective TTA
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_selective_tta_logits(model, dataset, threshold, device, batch_size,
                                  num_workers, amp=False):
    """
    Two-pass selective TTA.

    Pass 1: run the original (un-augmented) view for all N samples, compute
            per-sample Shannon entropy to identify uncertain samples.
    Pass 2: run the 7 remaining d4 augmented views only for uncertain samples.

    Returns (8, N, C) logits where confident samples have view 0 replicated
    across all view slots — this means _compute_metrics works unchanged:
      - strategy "none"  → view 0 (correct for everyone)
      - strategy "d4"    → mean of all 8 views (real aug for uncertain,
                           view-0-repeated = view 0 for confident)

    Also returns coverage (fraction of samples that received TTA) and
    avg_views_per_sample (1 + coverage * 7).
    """
    from torch.utils.data import Subset

    model.eval()
    use_amp = amp and device == "cuda"

    orig_tf   = get_tta_transforms("none")[0]          # original view
    extra_tfs = get_tta_transforms("d4")[1:]            # 7 augmented views
    n_extra   = len(extra_tfs)                          # 7
    n_views   = 1 + n_extra                             # 8

    def _collate_pil(batch):
        imgs, labels = zip(*batch)
        return list(imgs), torch.tensor(labels)

    dataset.transform = None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=_collate_pil)

    # --- Pass 1: original view for all N samples ---
    base_chunks, label_chunks = [], []
    for images, labels in tqdm(loader, desc="  pass 1 (all N, orig view)", leave=False):
        x = torch.stack([orig_tf(img) for img in images]).to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        base_chunks.append(logits.cpu())
        label_chunks.append(labels)

    base_logits = torch.cat(base_chunks)   # (N, C)
    all_labels  = torch.cat(label_chunks)  # (N,)
    N, C = base_logits.shape

    # Entropy threshold
    base_probs    = base_logits.softmax(-1)
    entropy       = -(base_probs * (base_probs + 1e-8).log()).sum(-1)  # (N,)
    thresh_val    = threshold * float(np.log(C))
    uncertain_mask = entropy > thresh_val
    uncertain_idx  = uncertain_mask.nonzero(as_tuple=False).squeeze(1)
    N_uncertain    = uncertain_idx.numel()
    coverage       = N_uncertain / N
    avg_views      = 1.0 + coverage * n_extra

    print(f"  threshold={threshold:.3f} × log({C}) = {thresh_val:.4f} | "
          f"uncertain={N_uncertain:,}/{N:,} ({coverage*100:.1f}%) | "
          f"avg_views={avg_views:.2f} ({avg_views/n_views*100:.1f}% compute)")

    # Build output tensor: start with view 0 replicated for all samples
    # Shape (n_views, N, C); confident samples keep view-0 in all slots
    all_logits = base_logits.unsqueeze(0).expand(n_views, -1, -1).clone()

    if N_uncertain > 0:
        # --- Pass 2: 7 augmented views for uncertain samples only ---
        subset        = Subset(dataset, uncertain_idx.tolist())
        subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, collate_fn=_collate_pil)
        extra_chunks = []
        for images, _ in tqdm(subset_loader,
                               desc=f"  pass 2 ({N_uncertain:,} uncertain, {n_extra} views)",
                               leave=False):
            B = len(images)
            view_tensors = [torch.stack([tf(img) for img in images]) for tf in extra_tfs]
            mega = torch.cat(view_tensors, dim=0).to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                mega_logits = model(mega)
            extra_chunks.append(mega_logits.reshape(n_extra, B, -1).cpu())

        extra_logits = torch.cat(extra_chunks, dim=1)          # (7, N_uncertain, C)
        all_logits[1:, uncertain_idx, :] = extra_logits

    dataset.transform = None   # restore (was already None)
    return all_logits, all_labels, coverage, avg_views


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

    # Expected Calibration Error (ECE) — 15 equal-width bins
    max_probs = probs.max(dim=1).values.numpy()        # confidence per sample
    n_bins = 15
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (max_probs > lo) & (max_probs <= hi)
        if b == 0:
            mask |= (max_probs == lo)   # include 0.0 in first bin
        if mask.sum() == 0:
            continue
        bin_acc = (preds[mask] == labels[mask]).mean()
        bin_conf = max_probs[mask].mean()
        ece += mask.sum() / n_total * abs(bin_acc - bin_conf)

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
        "ece": float(ece),
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
    if args.dataset == "mhist":
        test_set = MHISTDataset(split="test", transform=None)
    else:
        test_set = HistoDataset(args.dataset, split="test", transform=None, cache_dir=args.cache_dir)
    num_classes = args.num_classes or test_set.num_classes

    backbone_mode = _infer_backbone_mode(args.checkpoint)
    no_augment = args.no_augment or _infer_no_augment(args.checkpoint)
    aug_tag = "noaug" if no_augment else "aug"

    # Load model (match backbone mode from training so state_dict keys align)
    freeze = backbone_mode == "frozen"
    ckpt = torch.load(args.checkpoint, map_location=device)
    mlp_hidden = ckpt.get("mlp_hidden", None)
    model = get_model(args.model, num_classes=num_classes, freeze_backbone=freeze, mlp_hidden=mlp_hidden).to(device)
    if ckpt.get("head_only"):
        # Head-only checkpoint: backbone uses pretrained weights, load classifier only
        model.classifier.load_state_dict(ckpt["classifier_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    ckpt_name = os.path.basename(args.checkpoint)
    print(f"\nLoaded {ckpt_name} (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")
    print(f"Seed: {args.seed}")
    print(f"Train augmentation: {'OFF' if no_augment else 'ON'}")
    if args.amp:
        print(f"Mixed precision: fp16 AMP enabled")

    # --- Build deduplicated view superset and run inference ONCE ---
    all_transforms, strategy_view_indices = _build_view_superset(args.tta_strategies)

    t0 = time.time()
    if args.selective_tta:
        print(f"\nSelective TTA (threshold={args.selective_threshold}) ...")
        all_logits, all_labels, coverage, avg_views = compute_selective_tta_logits(
            model=model,
            dataset=test_set,
            threshold=args.selective_threshold,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp=args.amp,
        )
        # Remap strategy_view_indices to the 8-view d4 layout produced by
        # compute_selective_tta_logits (view 0 = orig, views 1-7 = d4 augmented)
        d4_all = get_tta_transforms("d4")
        _name_to_sel_idx = {tf.name: i for i, tf in enumerate(d4_all)}
        strategy_view_indices = {}
        for strategy in args.tta_strategies:
            tfs = get_tta_transforms(strategy)
            strategy_view_indices[strategy] = [_name_to_sel_idx[tf.name] for tf in tfs]
    else:
        print(f"\nRunning inference for {len(all_transforms)} unique views "
              f"(deduplicated from {sum(len(v) for v in strategy_view_indices.values())} total)...")
        all_logits, all_labels = compute_all_view_logits(
            model=model,
            dataset=test_set,
            all_transforms=all_transforms,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp=args.amp,
        )
        coverage, avg_views = None, None

    inference_time = time.time() - t0
    print(f"Inference complete: {inference_time:.1f}s for {all_logits.shape[1]:,} samples × "
          f"{all_logits.shape[0]} views")

    # --- Optionally save raw logits ---
    if args.save_logits:
        logit_dir = os.path.join("logits", args.dataset)
        os.makedirs(logit_dir, exist_ok=True)
        ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        logit_path = os.path.join(logit_dir, f"{ckpt_stem}.pt")
        torch.save({
            "logits": all_logits,          # (n_views, N, C) float32 on CPU
            "labels": all_labels,          # (N,) int64
            "view_names": [tf.name for tf in all_transforms],
            "model": args.model,
            "dataset": args.dataset,
            "backbone_mode": backbone_mode,
            "aug_tag": aug_tag,
            "seed": args.seed,
        }, logit_path)
        print(f"Logits saved to {logit_path}  [{all_logits.shape}]")

    # --- Optionally save per-sample parquet (probe_persample format) ---
    if args.save_persample:
        import pandas as pd
        _raw_dir = os.path.join("results", "raw")
        os.makedirs(_raw_dir, exist_ok=True)

        # sample_ids: MHIST has .filenames; others fall back to index strings
        if hasattr(test_set, "filenames"):
            _sample_ids = test_set.filenames
        elif hasattr(test_set, "dataset") and hasattr(test_set.dataset, "filenames"):
            _sample_ids = [test_set.dataset.filenames[i] for i in test_set.indices]
        else:
            _sample_ids = [str(i) for i in range(all_logits.shape[1])]

        # class names
        _classes = test_set.classes if hasattr(test_set, "classes") else [str(i) for i in range(all_logits.shape[2])]

        # view indices for "none" (orig) and "d4"
        _none_idx = strategy_view_indices.get("none", [0])
        _d4_idx   = strategy_view_indices.get("d4",   list(range(all_logits.shape[0])))

        _probs_all = torch.softmax(all_logits.float(), dim=-1)  # (n_views, N, C)
        _probs_0   = _probs_all[_none_idx[0]]                   # (N, C)
        _probs_d4  = _probs_all[_d4_idx].mean(0)               # (N, C)

        def _np_entropy(p):  # p: (N, C) numpy
            return -(p * np.log(p + 1e-8)).sum(1)

        _p0_np  = _probs_0.numpy()
        _pd4_np = _probs_d4.numpy()
        _labels_np = all_labels.numpy()

        _pred_0   = _p0_np.argmax(1)
        _pred_d4  = _pd4_np.argmax(1)
        _epist    = _probs_all[_d4_idx].numpy().std(0).mean(1)  # mean std across classes

        _ps_df = pd.DataFrame({
            "model":           args.model,
            "dataset":         args.dataset,
            "head_type":       "linear",
            "seed":            args.seed,
            "sample_id":       _sample_ids,
            "true_label":      _labels_np,
            "true_label_name": [_classes[l] for l in _labels_np],
            "pred_0":          _pred_0,
            "pred_d4":         _pred_d4,
            "entropy_0":       _np_entropy(_p0_np).astype(np.float64),
            "entropy_d4":      _np_entropy(_pd4_np).astype(np.float64),
            "epistemic_unc":   _epist.astype(np.float32),
            "conf_0":          _p0_np.max(1).astype(np.float32),
            "conf_d4":         _pd4_np.max(1).astype(np.float32),
            "correct_0":       _pred_0  == _labels_np,
            "correct_d4":      _pred_d4 == _labels_np,
        })
        _tag = f"{args.dataset}_{args.model}_linear_seed{args.seed}"
        _pq_path = os.path.join(_raw_dir, f"probe_persample_{_tag}.parquet")
        _ps_df.to_parquet(_pq_path, index=False)
        print(f"Per-sample parquet saved to {_pq_path}  [{len(_ps_df)} rows]")

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
                "mlp_hidden": mlp_hidden,
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
                "ece": metrics["ece"],
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

    # Selective TTA compute summary
    if args.selective_tta and avg_views is not None:
        n_total_views = all_logits.shape[0]
        print(f"\nSelective TTA compute summary (threshold={args.selective_threshold}):")
        print(f"  coverage (samples receiving TTA): {coverage*100:.1f}%")
        print(f"  avg views per sample:             {avg_views:.2f} / {n_total_views}")
        print(f"  compute vs full TTA:              {avg_views/n_total_views*100:.1f}%  "
              f"({(1-avg_views/n_total_views)*100:.1f}% saved)")

    # Find best by balanced_acc
    best_key = max(results, key=lambda k: results[k]["balanced_acc"])
    print(
        f"\nBest: strategy='{best_key[0]}' agg='{best_key[1]}' "
        f"balanced_acc={results[best_key]['balanced_acc']:.4f}"
    )

    # Save results with seed included
    if not args.no_save_results:
        _SKIP = {"preds", "labels"}
        serialisable = [{k: v for k, v in r.items() if k not in _SKIP} for r in all_rows]
        os.makedirs("results/raw", exist_ok=True)
        import re as _re
        _ckpt_base = os.path.basename(args.checkpoint)
        _m = _re.search(r'(_sub\d+)', _ckpt_base)
        _sub_tag = _m.group(1) if _m else ""
        _mlp_tag = f"_mlp{mlp_hidden}" if mlp_hidden is not None else ""
        out_path = f"results/raw/tta_results_{args.dataset}_{args.model}_{backbone_mode}_{aug_tag}{_sub_tag}{_mlp_tag}_seed{args.seed}.json"
        with open(out_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
