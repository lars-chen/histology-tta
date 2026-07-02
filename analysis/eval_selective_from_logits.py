#!/usr/bin/env python3
"""
Post-hoc entropy-based selective TTA from stored logits.

Loads a pre-saved (n_views, N, C) logit file and evaluates selective TTA
at multiple entropy thresholds — no model inference required.

View layout expected:
  view 0 = original (no augmentation)
  views 1-7 = the remaining D4 symmetry augmentations

Usage:
    python eval_selective_from_logits.py --logits_file logits/mhist/mhist_phikon2_frozen_aug_seed42_best.pt
    python eval_selective_from_logits.py --logits_dir logits/mhist  [processes all .pt files]
"""

import argparse
import json
import math
import os
import re

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * (probs + 1e-8).log()).sum(dim=-1)


def _compute_metrics(logits_list, labels):
    """logits_list: list of (N, C) tensors."""
    probs = F.softmax(torch.stack(logits_list, dim=0), dim=-1)  # (V, N, C)
    mean_probs = probs.mean(dim=0)                               # (N, C)
    preds = mean_probs.argmax(dim=1).numpy()
    labels_np = labels.numpy()
    n_total = len(labels_np)
    n_correct = int((preds == labels_np).sum())

    total_unc = _entropy(mean_probs).mean().item()
    aleatoric_unc = _entropy(probs).mean(dim=0).mean().item()
    epistemic_unc = total_unc - aleatoric_unc

    view_preds = probs.argmax(dim=-1)
    agreement_rate = (view_preds == view_preds[0:1]).all(dim=0).float().mean().item()

    max_probs = mean_probs.max(dim=1).values.numpy()
    n_bins = 15
    bins = torch.linspace(0.0, 1.0, n_bins + 1).numpy()
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (max_probs > lo) & (max_probs <= hi)
        if b == 0:
            mask |= max_probs == lo
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n_total * abs((preds[mask] == labels_np[mask]).mean() - max_probs[mask].mean())

    return {
        "acc": float(accuracy_score(labels_np, preds)),
        "balanced_acc": float(balanced_accuracy_score(labels_np, preds)),
        "macro_f1": float(f1_score(labels_np, preds, average="macro", zero_division=0)),
        "n_correct": n_correct,
        "n_wrong": n_total - n_correct,
        "n_total": n_total,
        "total_unc": total_unc,
        "aleatoric_unc": aleatoric_unc,
        "epistemic_unc": epistemic_unc,
        "agreement_rate": agreement_rate,
        "ece": float(ece),
    }


def eval_file(logits_path: str, out_dir: str):
    data = torch.load(logits_path, map_location="cpu")
    all_logits = data["logits"]   # (n_views, N, C)
    all_labels = data["labels"]   # (N,)
    model = data["model"]
    dataset = data["dataset"]
    backbone_mode = data.get("backbone_mode", "unknown")
    aug_tag = data.get("aug_tag", "aug")
    seed = data.get("seed", 42)
    view_names = data.get("view_names", [])

    n_views, N, C = all_logits.shape
    log_C = math.log(C)

    ckpt_stem = os.path.splitext(os.path.basename(logits_path))[0]
    sub_match = re.search(r'(_sub\d+)', ckpt_stem)
    sub_tag = sub_match.group(1) if sub_match else ""

    base_probs = F.softmax(all_logits[0], dim=-1)   # (N, C) — original view
    base_entropy = _entropy(base_probs)              # (N,)

    # --- Baseline: no TTA (original view only) ---
    baseline_metrics = _compute_metrics([all_logits[0]], all_labels)
    baseline_preds_np = F.softmax(all_logits[0], dim=-1).argmax(dim=1).numpy()

    # --- Full D4 TTA ---
    full_metrics = _compute_metrics([all_logits[v] for v in range(n_views)], all_labels)
    full_preds_np = F.softmax(all_logits.mean(dim=0), dim=-1).argmax(dim=1).numpy()

    rows = []

    def _make_row(strategy, threshold_frac, metrics, n_sel_views, coverage, avg_views_per_sample, ref_preds_np=None):
        labels_np = all_labels.numpy()
        preds = F.softmax(
            torch.stack([all_logits[v] for v in range(n_sel_views)] if strategy == "d4"
                        else [all_logits[0]], dim=0).mean(dim=0),
            dim=-1
        ).argmax(dim=1).numpy()
        # corrected/corrupted vs. no-TTA baseline
        if ref_preds_np is not None:
            baseline_wrong = ref_preds_np != labels_np
            baseline_right = ~baseline_wrong
            n_corrected = int((baseline_wrong & (preds == labels_np)).sum())
            n_corrupted = int((baseline_right & (preds != labels_np)).sum())
        else:
            n_corrected = n_corrupted = None
        return {
            "model": model,
            "dataset": dataset,
            "backbone_mode": backbone_mode,
            "train_augment": aug_tag == "aug",
            "checkpoint": ckpt_stem + ".pt",
            "seed": seed,
            "strategy": strategy,
            "aggregation": "mean",
            "threshold_frac": threshold_frac,
            "coverage": coverage,
            "avg_views_per_sample": avg_views_per_sample,
            **metrics,
            "n_views": n_sel_views,
            "n_corrected": n_corrected,
            "n_corrupted": n_corrupted,
        }

    # baseline row
    rows.append({
        "model": model, "dataset": dataset, "backbone_mode": backbone_mode,
        "train_augment": aug_tag == "aug", "checkpoint": ckpt_stem + ".pt", "seed": seed,
        "strategy": "none", "aggregation": "mean", "threshold_frac": None,
        "coverage": 0.0, "avg_views_per_sample": 1.0,
        **baseline_metrics, "n_views": 1, "n_corrected": None, "n_corrupted": None,
    })

    # full d4 row
    labels_np = all_labels.numpy()
    full_preds_arr = F.softmax(all_logits.mean(dim=0), dim=-1).argmax(dim=1).numpy()
    baseline_wrong = baseline_preds_np != labels_np
    rows.append({
        "model": model, "dataset": dataset, "backbone_mode": backbone_mode,
        "train_augment": aug_tag == "aug", "checkpoint": ckpt_stem + ".pt", "seed": seed,
        "strategy": "d4", "aggregation": "mean", "threshold_frac": None,
        "coverage": 1.0, "avg_views_per_sample": float(n_views),
        **full_metrics, "n_views": n_views,
        "n_corrected": int((baseline_wrong & (full_preds_arr == labels_np)).sum()),
        "n_corrupted": int(((~baseline_wrong) & (full_preds_arr != labels_np)).sum()),
    })

    # selective TTA rows — one per threshold
    for t in THRESHOLDS:
        thresh_val = t * log_C
        uncertain_mask = base_entropy > thresh_val          # (N,) bool
        n_uncertain = int(uncertain_mask.sum())
        coverage = n_uncertain / N
        avg_views = 1.0 + coverage * (n_views - 1)

        # Build (n_views, N, C) where confident samples have view 0 replicated
        sel_logits = all_logits[0:1].expand(n_views, -1, -1).clone()
        if n_uncertain > 0:
            sel_logits[:, uncertain_mask, :] = all_logits[:, uncertain_mask, :]

        sel_metrics = _compute_metrics([sel_logits[v] for v in range(n_views)], all_labels)

        sel_preds = F.softmax(sel_logits.mean(dim=0), dim=-1).argmax(dim=1).numpy()
        n_corrected = int((baseline_wrong & (sel_preds == labels_np)).sum())
        n_corrupted = int(((~baseline_wrong) & (sel_preds != labels_np)).sum())

        full_passes = N * n_views
        sel_passes = N + n_uncertain * (n_views - 1)  # orig pass for all + aug passes for uncertain
        passes_saved = full_passes - sel_passes
        pct_saved = passes_saved / full_passes * 100
        print(
            f"  t={t:.1f} coverage={coverage*100:.1f}% avg_views={avg_views:.2f} "
            f"passes={sel_passes:,}/{full_passes:,} saved={passes_saved:,} ({pct_saved:.1f}%) "
            f"acc={sel_metrics['acc']:.4f} bal={sel_metrics['balanced_acc']:.4f} "
            f"corrected={n_corrected} corrupted={n_corrupted}"
        )

        rows.append({
            "model": model, "dataset": dataset, "backbone_mode": backbone_mode,
            "train_augment": aug_tag == "aug", "checkpoint": ckpt_stem + ".pt", "seed": seed,
            "strategy": f"selective_t{t:.1f}", "aggregation": "mean",
            "threshold_frac": t, "coverage": coverage, "avg_views_per_sample": avg_views,
            "passes_used": sel_passes, "passes_full_tta": full_passes,
            "passes_saved": passes_saved, "pct_passes_saved": pct_saved,
            **sel_metrics, "n_views": n_views,
            "n_corrected": n_corrected, "n_corrupted": n_corrupted,
        })

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"selective_tta_{dataset}_{model}_{backbone_mode}_{aug_tag}{sub_tag}_seed{seed}.json")
    # append to existing file if present (multiple seeds)
    existing = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing.extend(rows)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  -> saved to {out_path} ({len(rows)} rows, {len(existing)} total)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits_file", type=str, default=None,
                        help="Path to a single .pt logits file")
    parser.add_argument("--logits_dir", type=str, default=None,
                        help="Directory of .pt logits files (all will be processed)")
    parser.add_argument("--out_dir", type=str, default="results/raw",
                        help="Output directory for JSON results")
    args = parser.parse_args()

    if args.logits_file:
        files = [args.logits_file]
    elif args.logits_dir:
        files = sorted(
            os.path.join(args.logits_dir, f)
            for f in os.listdir(args.logits_dir)
            if f.endswith(".pt")
        )
    else:
        parser.error("Provide --logits_file or --logits_dir")

    for fpath in files:
        print(f"\n=== {os.path.basename(fpath)} ===")
        try:
            eval_file(fpath, args.out_dir)
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
