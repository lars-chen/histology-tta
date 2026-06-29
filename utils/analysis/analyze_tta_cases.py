#!/usr/bin/env python3
"""
Per-sample analysis of where TTA corrupts vs. corrects.

Hypotheses tested:
  A. Low-entropy (confident) baseline predictions → TTA corrupts
  B. Entropy delta: corrected cases → TTA sharpens (ΔH < 0)
  C. Low per-view agreement → TTA corrects (high epistemic uncertainty)
  D. Low class support → TTA corrects more
  E. High whitespace (tissue-edge patches) → TTA corrects more
  F. Low annotator agreement (MHIST) → TTA corrects more

Usage:
    python analyze_tta_cases.py --dataset tcga-ut --model phikon
    python analyze_tta_cases.py --dataset mhist --model phikon
    python analyze_tta_cases.py --dataset tcga-ut --model phikon --logits_dir logits/
"""

import argparse
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn.functional as F


OUTCOME_ORDER  = ["corrected", "both_correct", "both_wrong", "corrupted"]
OUTCOME_COLORS = {
    "corrected":    "#2196F3",
    "both_correct": "#4CAF50",
    "both_wrong":   "#FF9800",
    "corrupted":    "#F44336",
}
OUTCOME_LABELS = {
    "corrected":    "Corrected\n(wrong→right)",
    "both_correct": "Both\ncorrect",
    "both_wrong":   "Both\nwrong",
    "corrupted":    "Corrupted\n(right→wrong)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      default="tcga-ut")
    p.add_argument("--model",        default="phikon")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--per_sample_dir", default="results/per_sample")
    p.add_argument("--per_class_csv",  default="results/tta_per_class.csv")
    p.add_argument("--logits_dir",     default="logits")
    p.add_argument("--out_dir",        default="figures")
    p.add_argument("--whitespace_threshold", type=int, default=220,
                   help="Grayscale pixel value above which a pixel is 'background'")
    return p.parse_args()


def load_per_sample(args) -> pd.DataFrame:
    path = Path(args.per_sample_dir) / f"per_sample_{args.dataset}_{args.model}_seed{args.seed}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Per-sample CSV not found: {path}\n"
            f"Run: python eval_per_sample.py --dataset {args.dataset} --model {args.model} --seed {args.seed}"
        )
    df = pd.read_csv(path)
    # Parse softmax string lists → numpy arrays stored as object column
    for col in ("baseline_softmax", "tta_softmax"):
        df[col] = df[col].apply(ast.literal_eval)
    return df


def shannon_entropy(probs_list, eps=1e-9):
    """Per-row Shannon entropy, normalized by log(C)."""
    arr = np.array(probs_list, dtype=np.float64)  # (N, C)
    H = -(arr * np.log(arr + eps)).sum(axis=1)
    H_max = np.log(arr.shape[1])
    return H / H_max


def mw_pval(a, b):
    """Mann-Whitney U p-value (two-sided)."""
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return p


def violin_by_outcome(ax, df, col, outcomes, title, ylabel):
    data   = [df.loc[df["outcome"] == o, col].dropna().values for o in outcomes]
    colors = [OUTCOME_COLORS[o] for o in outcomes]
    labels = [OUTCOME_LABELS[o] for o in outcomes]
    parts = ax.violinplot(data, positions=range(len(outcomes)), showmedians=True, showextrema=False)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    ax.set_xticks(range(len(outcomes)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    # Annotate n per group
    for i, (o, d) in enumerate(zip(outcomes, data)):
        ax.text(i, ax.get_ylim()[0], f"n={len(d)}", ha="center", va="bottom", fontsize=7, color="gray")
    return ax


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# Analysis A+B: Entropy & entropy delta
# ---------------------------------------------------------------------------

def analysis_entropy(df, out_dir, dataset):
    df = df.copy()
    df["H_base"] = shannon_entropy(df["baseline_softmax"].tolist())
    df["H_tta"]  = shannon_entropy(df["tta_softmax"].tolist())
    df["dH"]     = df["H_tta"] - df["H_base"]

    present = [o for o in OUTCOME_ORDER if o in df["outcome"].values]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Entropy analysis — {dataset} / {df.attrs.get('model','')}", fontsize=11)

    violin_by_outcome(axes[0], df, "H_base", present,
                      "Baseline entropy by outcome\n(low = confident)",
                      "Normalized Shannon entropy")

    violin_by_outcome(axes[1], df, "baseline_conf", present,
                      "Baseline confidence by outcome",
                      "Max softmax probability")

    violin_by_outcome(axes[2], df, "dH", present,
                      "Entropy delta (TTA − baseline)\n(negative = TTA sharpens)",
                      "ΔH (normalized)")

    # Add p-value annotation between corrected and corrupted
    if "corrected" in present and "corrupted" in present:
        for ax, col in zip(axes, ["H_base", "baseline_conf", "dH"]):
            a = df.loc[df["outcome"] == "corrected",  col].dropna()
            b = df.loc[df["outcome"] == "corrupted",  col].dropna()
            p = mw_pval(a, b)
            ax.text(0.5, 0.97, f"corrected vs corrupted: p={p:.2e}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7, color="dimgray")

    fig.tight_layout()
    save(fig, Path(out_dir) / "tta_case_entropy.png")


# ---------------------------------------------------------------------------
# Analysis B (supplemental): entropy scatter
# ---------------------------------------------------------------------------

def analysis_entropy_scatter(df, out_dir, dataset):
    df = df.copy()
    df["H_base"] = shannon_entropy(df["baseline_softmax"].tolist())
    df["H_tta"]  = shannon_entropy(df["tta_softmax"].tolist())
    df["dH"]     = df["H_tta"] - df["H_base"]

    present = [o for o in OUTCOME_ORDER if o in df["outcome"].values]

    fig, ax = plt.subplots(figsize=(6, 5))
    for o in present:
        sub = df[df["outcome"] == o]
        ax.scatter(sub["H_base"], sub["dH"], s=6, alpha=0.4,
                   color=OUTCOME_COLORS[o], label=OUTCOME_LABELS[o].replace("\n", " "))
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Baseline entropy (normalized)", fontsize=10)
    ax.set_ylabel("ΔH = H_tta − H_baseline (normalized)", fontsize=10)
    ax.set_title(f"Entropy delta vs. baseline entropy\n{dataset}", fontsize=10)
    ax.legend(fontsize=8, markerscale=3)
    fig.tight_layout()
    save(fig, Path(out_dir) / "tta_case_entropy_scatter.png")


# ---------------------------------------------------------------------------
# Analysis C: Per-view agreement from logits
# ---------------------------------------------------------------------------

def analysis_view_agreement(df, logits_dir, out_dir, dataset, model, seed):
    logits_path_dir = Path(logits_dir) / dataset
    if not logits_path_dir.exists():
        print(f"  [C] Skipping — no logits dir: {logits_path_dir}")
        return

    # Find a matching .pt file
    candidates = list(logits_path_dir.glob(f"*{model}*seed{seed}*.pt")) or \
                 list(logits_path_dir.glob(f"*{model}*.pt"))
    if not candidates:
        print(f"  [C] Skipping — no logit file found for {model} seed={seed} in {logits_path_dir}")
        return

    pt = torch.load(candidates[0], map_location="cpu", weights_only=False)
    # logits: (n_views, N, C)  or  (N, n_views, C)
    logits = pt["logits"]
    if logits.shape[0] != 8:
        logits = logits.permute(1, 0, 2)   # → (n_views, N, C)

    probs    = F.softmax(logits.float(), dim=-1)            # (n_views, N, C)
    preds    = probs.argmax(dim=-1)                          # (n_views, N)
    majority = preds.mode(dim=0).values                      # (N,) majority argmax
    agreement = (preds == majority.unsqueeze(0)).float().mean(dim=0).numpy()  # (N,)

    # Match rows: logit file order should match per-sample CSV order
    if len(agreement) != len(df):
        print(f"  [C] Skipping — logit N={len(agreement)} ≠ CSV N={len(df)}")
        return

    df = df.copy()
    df["view_agreement"] = agreement
    present = [o for o in OUTCOME_ORDER if o in df["outcome"].values]

    fig, ax = plt.subplots(figsize=(6, 4))
    violin_by_outcome(ax, df, "view_agreement", present,
                      f"Per-view D4 agreement by outcome\n{dataset} / {model}",
                      "Fraction of views agreeing with majority")
    if "corrected" in present and "corrupted" in present:
        a = df.loc[df["outcome"] == "corrected",  "view_agreement"].dropna()
        b = df.loc[df["outcome"] == "corrupted",  "view_agreement"].dropna()
        p = mw_pval(a, b)
        ax.text(0.5, 0.97, f"corrected vs corrupted: p={p:.2e}",
                transform=ax.transAxes, ha="center", va="top", fontsize=7, color="dimgray")
    fig.tight_layout()
    save(fig, Path(out_dir) / "tta_case_agreement.png")


# ---------------------------------------------------------------------------
# Analysis D: Class support — scatter at class level
# (per-sample violin would be degenerate: only 31 discrete values for TCGA-UT)
# ---------------------------------------------------------------------------

def analysis_class_support(df, per_class_csv, out_dir, dataset, model):
    pc = pd.read_csv(per_class_csv)
    pc = pc[(pc["dataset"] == dataset) &
            (pc["model"]   == model)   &
            (pc["strategy"] == "d4")   &
            (pc["aggregation"] == "mean")]

    if pc.empty:
        pc = pd.read_csv(per_class_csv)
        pc = pc[(pc["dataset"] == dataset) & (pc["model"] == model)]

    if pc.empty:
        print(f"  [D] Skipping — no per-class data for {model}/{dataset}")
        return

    support_map = pc.groupby("class_name")["support"].mean().to_dict()

    # Compute per-class correction / corruption rates
    rows = []
    for cls, grp in df.groupby("true_label"):
        n = len(grp)
        if n == 0:
            continue
        rows.append({
            "class":           cls,
            "support":         support_map.get(cls, np.nan),
            "n":               n,
            "correction_rate": (grp["outcome"] == "corrected").mean(),
            "corruption_rate": (grp["outcome"] == "corrupted").mean(),
            "both_correct":    (grp["outcome"] == "both_correct").mean(),
        })
    cls_df = pd.DataFrame(rows).dropna(subset=["support"])

    if cls_df.empty:
        print("  [D] Skipping — could not map any class names to support values")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Class support vs. TTA outcome rate — {dataset} / {model}", fontsize=10)

    for ax, rate_col, color, label in [
        (axes[0], "correction_rate", OUTCOME_COLORS["corrected"],  "Correction rate (wrong→right)"),
        (axes[1], "corruption_rate", OUTCOME_COLORS["corrupted"], "Corruption rate (right→wrong)"),
    ]:
        ax.scatter(cls_df["support"], cls_df[rate_col],
                   s=cls_df["n"] / cls_df["n"].max() * 200 + 10,
                   color=color, alpha=0.7, edgecolors="white", linewidth=0.5)
        for _, row in cls_df.iterrows():
            ax.annotate(row["class"].split("_")[0], (row["support"], row[rate_col]),
                        fontsize=5, alpha=0.6, ha="left", va="bottom")
        # Spearman correlation
        r, p = stats.spearmanr(cls_df["support"], cls_df[rate_col])
        ax.set_xscale("log")
        ax.set_xlabel("Test-set class support (log scale)", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(f"ρ={r:.2f}, p={p:.3f}", fontsize=9)

    fig.tight_layout()
    save(fig, Path(out_dir) / "tta_case_support.png")


# ---------------------------------------------------------------------------
# Analysis E: Whitespace / tissue-edge
# ---------------------------------------------------------------------------

def compute_whitespace(pil_img, threshold=220):
    gray = np.array(pil_img.convert("L"), dtype=np.uint8)
    return (gray > threshold).mean()


def analysis_whitespace(df, out_dir, dataset, whitespace_threshold):
    from PIL import Image as PILImage

    ws_vals = []

    if dataset == "mhist":
        if "image_id" not in df.columns:
            print("  [E] Skipping MHIST whitespace — no image_id column")
            return
        img_dir = Path("data/mhist/images")
        for fname in df["image_id"]:
            img = PILImage.open(img_dir / fname).convert("RGB")
            ws_vals.append(compute_whitespace(img, whitespace_threshold))

    elif dataset == "tcga-ut":
        if "sample_idx" not in df.columns:
            print("  [E] Skipping TCGA-UT whitespace — no sample_idx column")
            return
        print("  [E] Loading TCGA-UT images from HuggingFace for whitespace computation...")
        from data.data import DATASETS
        from datasets import load_dataset
        cfg   = DATASETS["tcga-ut"]
        hf_ds = load_dataset(cfg[0], cfg[1], split=cfg[3])
        for idx in df["sample_idx"]:
            pil = hf_ds[int(idx)][cfg[4]].convert("RGB")
            ws_vals.append(compute_whitespace(pil, whitespace_threshold))

    else:
        print(f"  [E] Skipping whitespace — not implemented for {dataset}")
        return

    df = df.copy()
    df["whitespace_pct"] = [v * 100 for v in ws_vals]
    present = [o for o in OUTCOME_ORDER if o in df["outcome"].values]

    fig, ax = plt.subplots(figsize=(6, 4))
    violin_by_outcome(ax, df, "whitespace_pct", present,
                      f"Whitespace % by outcome\n{dataset} (threshold L>{whitespace_threshold})",
                      "Background pixel % (higher = more whitespace)")
    if "corrected" in present and "corrupted" in present:
        a = df.loc[df["outcome"] == "corrected",  "whitespace_pct"].dropna()
        b = df.loc[df["outcome"] == "corrupted",  "whitespace_pct"].dropna()
        p = mw_pval(a, b)
        ax.text(0.5, 0.97, f"corrected vs corrupted: p={p:.2e}",
                transform=ax.transAxes, ha="center", va="top", fontsize=7, color="dimgray")
    fig.tight_layout()
    save(fig, Path(out_dir) / "tta_case_whitespace.png")


# ---------------------------------------------------------------------------
# Analysis F: Annotator disagreement (MHIST only)
# ---------------------------------------------------------------------------

def analysis_mhist_agreement(df, out_dir):
    if "agreement" not in df.columns:
        print("  [F] Skipping — no annotator agreement column (MHIST only)")
        return
    present = [o for o in OUTCOME_ORDER if o in df["outcome"].values]
    fig, ax = plt.subplots(figsize=(6, 4))
    violin_by_outcome(ax, df, "agreement", present,
                      "Annotator agreement by outcome (MHIST)\n(lower = more ambiguous patch)",
                      "Annotator agreement (0.5–1.0)")
    if "corrected" in present and "corrupted" in present:
        a = df.loc[df["outcome"] == "corrected",  "agreement"].dropna()
        b = df.loc[df["outcome"] == "corrupted",  "agreement"].dropna()
        if len(a) > 0 and len(b) > 0:
            p = mw_pval(a, b)
            ax.text(0.5, 0.97, f"corrected vs corrupted: p={p:.2e}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7, color="dimgray")
    fig.tight_layout()
    save(fig, Path(out_dir) / "tta_case_mhist_agreement.png")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(df, dataset, model):
    df = df.copy()
    df["H_base"] = shannon_entropy(df["baseline_softmax"].tolist())
    df["H_tta"]  = shannon_entropy(df["tta_softmax"].tolist())
    df["dH"]     = df["H_tta"] - df["H_base"]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}   Model: {model}")
    print(f"{'='*60}")
    print(df["outcome"].value_counts().to_string())
    print()

    rows = []
    for o in OUTCOME_ORDER:
        sub = df[df["outcome"] == o]
        if sub.empty:
            continue
        rows.append({
            "outcome":      o,
            "n":            len(sub),
            "mean_H_base":  sub["H_base"].mean(),
            "mean_conf":    sub["baseline_conf"].mean(),
            "mean_dH":      sub["dH"].mean(),
        })
    tbl = pd.DataFrame(rows).set_index("outcome")
    print(tbl.to_string(float_format="{:.4f}".format))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    print(f"Dataset: {args.dataset}   Model: {args.model}   Seed: {args.seed}")

    df = load_per_sample(args)
    df.attrs["model"] = args.model

    print_summary(df, args.dataset, args.model)

    print("\n[A+B] Entropy & confidence analysis...")
    analysis_entropy(df, args.out_dir, args.dataset)

    print("[B]   Entropy scatter...")
    analysis_entropy_scatter(df, args.out_dir, args.dataset)

    print("[C]   Per-view agreement analysis...")
    analysis_view_agreement(df, args.logits_dir, args.out_dir,
                            args.dataset, args.model, args.seed)

    print("[D]   Class support analysis...")
    analysis_class_support(df, args.per_class_csv, args.out_dir,
                           args.dataset, args.model)

    print("[E]   Whitespace / tissue-edge analysis...")
    analysis_whitespace(df, args.out_dir, args.dataset, args.whitespace_threshold)

    print("[F]   Annotator disagreement analysis (MHIST only)...")
    analysis_mhist_agreement(df, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
