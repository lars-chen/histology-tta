"""
Per-class evaluation metrics.
"""
import numpy as np
from sklearn.metrics import f1_score


def per_class_metrics(labels, preds, class_names):
    """
    Compute per-class accuracy and F1.

    Args:
        labels:      1-D array-like of integer ground-truth class indices
        preds:       1-D array-like of integer predicted class indices
        class_names: sequence of class name strings (index i → class i)

    Returns:
        List of dicts ordered by class index:
            {"class": str, "acc": float, "f1": float, "support": int}
        acc is NaN for classes absent from the label set.
    """
    labels = np.asarray(labels)
    preds  = np.asarray(preds)
    n_classes = len(class_names)

    f1_per_class = f1_score(
        labels, preds,
        labels=list(range(n_classes)),
        average=None,
        zero_division=0,
    )

    rows = []
    for i, name in enumerate(class_names):
        mask    = labels == i
        support = int(mask.sum())
        acc     = float((preds[mask] == i).sum() / support) if support > 0 else float("nan")
        rows.append({
            "class":   name,
            "acc":     round(acc, 4),
            "f1":      round(float(f1_per_class[i]), 4),
            "support": support,
        })
    return rows


def print_per_class_table(class_metrics, strategy, agg):
    """Pretty-print a per-class accuracy / F1 table to stdout."""
    name_w = min(max(len(r["class"]) for r in class_metrics), 50)
    W = name_w + 34
    print(f"\n--- Per-class metrics: {strategy} / {agg} ---")
    print(f"{'Class':<{name_w}}  {'Acc':>8}  {'F1':>8}  {'Support':>8}")
    print("-" * W)
    for r in class_metrics:
        name    = r["class"][:name_w]
        acc_str = f"{r['acc']:.4f}" if r["acc"] == r["acc"] else "     nan"  # NaN check
        print(f"{name:<{name_w}}  {acc_str:>8}  {r['f1']:>8.4f}  {r['support']:>8,}")
    print("-" * W)
