#!/usr/bin/env python3
"""
Plot TTA experiment results from JSON files produced by evaluate_tta.py.

Usage:
    # All JSON files in current directory
    python utils/plot_results.py

    # Specific files or glob pattern
    python utils/plot_results.py tta_results_*.json

    # Save figures to a directory
    python utils/plot_results.py --out_dir figures/

Figures produced:
    1. acc_by_strategy_agg   — grouped bar chart: acc per (strategy, agg) for
                               every (dataset, model, backbone_mode) combination
    2. balanced_acc_heatmap  — heatmap: balanced_acc vs strategy × model
    3. correction_chart      — stacked bar: n_corrected vs n_corrupted per strategy
    4. per_class_f1          — heatmap of per-class F1 for the best strategy
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # no display needed on HPC nodes
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_json_files(paths: list[str]) -> pd.DataFrame:
    """Load all result JSON files into a single flat DataFrame."""
    records = []
    for p in paths:
        with open(p) as f:
            rows = json.load(f)
        for row in rows:
            flat = {k: v for k, v in row.items() if k != "per_class"}
            flat["_source"] = os.path.basename(p)
            records.append(flat)
    df = pd.DataFrame(records)
    # Ensure consistent types
    for col in ("acc", "balanced_acc"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_per_class(paths: list[str]) -> pd.DataFrame:
    """Load per-class metrics into a long DataFrame."""
    records = []
    for p in paths:
        with open(p) as f:
            rows = json.load(f)
        for row in rows:
            meta = {k: row[k] for k in
                    ("model", "dataset", "backbone_mode", "strategy", "aggregation")
                    if k in row}
            for pc in (row.get("per_class") or []):
                records.append({**meta, **pc})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STRATEGY_ORDER = ["none", "flips", "d4", "d4_color"]
AGG_ORDER      = ["mean", "vote", "confidence"]

def _strategy_label(s: str) -> str:
    return {"none": "No TTA", "flips": "Flips", "d4": "D4",
            "d4_color": "D4+Color"}.get(s, s)

def _exp_label(row) -> str:
    return f"{row['model']} / {row['backbone_mode']}"


def _ordered(col: pd.Series, order: list[str]) -> list:
    present = col.unique().tolist()
    return [x for x in order if x in present] + [x for x in present if x not in order]


# ---------------------------------------------------------------------------
# Plot 1 — Accuracy by strategy & aggregation (grouped bars per experiment)
# ---------------------------------------------------------------------------

def plot_acc_by_strategy(df: pd.DataFrame, out_dir: Path, metric: str = "balanced_acc"):
    """One figure per dataset; each panel = one (model, backbone_mode)."""
    datasets = df["dataset"].unique()

    for ds in sorted(datasets):
        sub = df[df["dataset"] == ds].copy()
        experiments = sub.apply(_exp_label, axis=1).unique()

        n_exp = len(experiments)
        fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp + 1, 4.5), sharey=True)
        if n_exp == 1:
            axes = [axes]

        strats  = _ordered(sub["strategy"], STRATEGY_ORDER)
        aggs    = _ordered(sub["aggregation"], AGG_ORDER)
        x       = np.arange(len(strats))
        width   = 0.8 / max(len(aggs), 1)
        colors  = plt.cm.tab10(np.linspace(0, 0.5, len(aggs)))

        for ax, exp in zip(axes, sorted(experiments)):
            mod, bm = exp.split(" / ")
            edf = sub[(sub["model"] == mod) & (sub["backbone_mode"] == bm)]

            for j, agg in enumerate(aggs):
                vals = []
                for strat in strats:
                    row = edf[(edf["strategy"] == strat) & (edf["aggregation"] == agg)]
                    vals.append(float(row[metric].values[0]) if len(row) else float("nan"))
                offset = (j - len(aggs) / 2 + 0.5) * width
                bars = ax.bar(x + offset, vals, width, label=agg, color=colors[j])
                for bar, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.003,
                                f"{v:.3f}", ha="center", va="bottom",
                                fontsize=6, rotation=45)

            ax.set_title(exp, fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([_strategy_label(s) for s in strats], fontsize=8)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            if ax is axes[0]:
                ax.set_ylabel(metric.replace("_", " ").title())

        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, title="Aggregation",
                   loc="lower center", ncol=len(aggs), fontsize=8,
                   bbox_to_anchor=(0.5, -0.05))
        fig.suptitle(f"Dataset: {ds}  |  {metric}", fontsize=13, y=1.01)
        fig.tight_layout()

        fname = out_dir / f"acc_by_strategy_{ds}_{metric}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Plot 2 — Balanced accuracy heatmap: model × strategy (best agg per cell)
# ---------------------------------------------------------------------------

def plot_heatmap(df: pd.DataFrame, out_dir: Path, metric: str = "balanced_acc"):
    datasets = df["dataset"].unique()

    for ds in sorted(datasets):
        sub = df[df["dataset"] == ds].copy()
        sub["exp"] = sub.apply(_exp_label, axis=1)
        strats = _ordered(sub["strategy"], STRATEGY_ORDER)

        # Best value across aggregations for each (exp, strategy)
        pivot = (sub.groupby(["exp", "strategy"])[metric]
                    .max()
                    .unstack("strategy")
                    .reindex(columns=strats))

        fig, ax = plt.subplots(figsize=(max(4, len(strats) * 1.5),
                                        max(3, len(pivot) * 0.7 + 1)))
        im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1,
                       cmap="RdYlGn")
        plt.colorbar(im, ax=ax, label=metric)

        ax.set_xticks(range(len(strats)))
        ax.set_xticklabels([_strategy_label(s) for s in strats], fontsize=9)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=9)

        for i in range(len(pivot)):
            for j in range(len(strats)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=8, color="black")

        ax.set_title(f"Dataset: {ds}  |  {metric} (best aggregation)", fontsize=11)
        fig.tight_layout()

        fname = out_dir / f"heatmap_{ds}_{metric}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Plot 3 — Corrected vs Corrupted stacked bar (TTA gain/loss)
# ---------------------------------------------------------------------------

def plot_correction(df: pd.DataFrame, out_dir: Path):
    """Show how many baseline-wrong samples TTA fixes, and how many it breaks."""
    tta_df = df[df["strategy"] != "none"].copy()
    if tta_df.empty:
        return

    tta_df["exp"] = tta_df.apply(_exp_label, axis=1)
    datasets = tta_df["dataset"].unique()

    for ds in sorted(datasets):
        sub = tta_df[tta_df["dataset"] == ds].copy()
        experiments = sorted(sub["exp"].unique())

        n_exp = len(experiments)
        fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp + 1, 4.5), sharey=False)
        if n_exp == 1:
            axes = [axes]

        strats = _ordered(sub["strategy"], [s for s in STRATEGY_ORDER if s != "none"])

        for ax, exp in zip(axes, experiments):
            mod, bm = exp.split(" / ")
            edf = sub[(sub["exp"] == exp)]

            # Best agg per strategy by balanced_acc
            best_rows = (edf.sort_values("balanced_acc", ascending=False)
                           .groupby("strategy")
                           .first()
                           .reindex(strats))

            corrected = best_rows["n_corrected"].fillna(0).astype(float).values
            corrupted = best_rows["n_corrupted"].fillna(0).astype(float).values
            x = np.arange(len(strats))

            ax.bar(x, corrected, label="Corrected", color="steelblue")
            ax.bar(x, -corrupted, label="Corrupted", color="tomato")
            ax.axhline(0, color="black", linewidth=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels([_strategy_label(s) for s in strats], fontsize=8)
            ax.set_title(exp, fontsize=10)
            ax.set_ylabel("# Samples")
            ax.grid(axis="y", linestyle="--", alpha=0.4)

        handles = [
            plt.Rectangle((0, 0), 1, 1, color="steelblue"),
            plt.Rectangle((0, 0), 1, 1, color="tomato"),
        ]
        fig.legend(handles, ["Corrected by TTA", "Corrupted by TTA"],
                   loc="lower center", ncol=2, fontsize=9,
                   bbox_to_anchor=(0.5, -0.05))
        fig.suptitle(f"Dataset: {ds}  |  TTA Corrections (best agg)", fontsize=13, y=1.01)
        fig.tight_layout()

        fname = out_dir / f"correction_{ds}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Plot 4 — Per-class F1 heatmap for best strategy per experiment
# ---------------------------------------------------------------------------

def plot_per_class_f1(pc_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path):
    if pc_df.empty:
        return

    datasets = pc_df["dataset"].unique()

    for ds in sorted(datasets):
        pc_sub  = pc_df[pc_df["dataset"] == ds].copy()
        sum_sub = summary_df[summary_df["dataset"] == ds].copy()

        if sum_sub.empty:
            continue

        # Identify best (strategy, agg) per experiment by balanced_acc
        sum_sub["exp"] = sum_sub.apply(_exp_label, axis=1)
        best_per_exp = (sum_sub.sort_values("balanced_acc", ascending=False)
                               .groupby("exp")
                               .first()
                               .reset_index()[["exp", "model", "backbone_mode",
                                               "strategy", "aggregation"]])

        classes = pc_sub["class"].unique().tolist()
        n_exp   = len(best_per_exp)
        if n_exp == 0:
            continue

        fig, axes = plt.subplots(1, n_exp,
                                  figsize=(max(6, len(classes) * 0.35) * n_exp + 1,
                                           max(4, n_exp * 0.5 + 2)),
                                  sharey=True)
        if n_exp == 1:
            axes = [axes]

        for ax, (_, brow) in zip(axes, best_per_exp.iterrows()):
            mask = (
                (pc_sub["model"]         == brow["model"]) &
                (pc_sub["backbone_mode"] == brow["backbone_mode"]) &
                (pc_sub["strategy"]      == brow["strategy"]) &
                (pc_sub["aggregation"]   == brow["aggregation"])
            )
            sub = pc_sub[mask].set_index("class")["f1"].reindex(classes).values
            sub = sub.reshape(-1, 1)

            im = ax.imshow(sub.T, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
            ax.set_yticks([0])
            ax.set_yticklabels([f"{brow['model']}\n{brow['backbone_mode']}\n"
                                 f"{_strategy_label(brow['strategy'])}/"
                                 f"{brow['aggregation']}"], fontsize=7)
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)

            for j, v in enumerate(sub.flatten()):
                if not np.isnan(v):
                    ax.text(j, 0, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="black")

        fig.colorbar(im, ax=axes[-1], label="F1")
        fig.suptitle(f"Dataset: {ds}  |  Per-class F1 (best strategy)", fontsize=12, y=1.02)
        fig.tight_layout()

        fname = out_dir / f"per_class_f1_{ds}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Plot TTA results from JSON files")
    p.add_argument("files", nargs="*",
                   help="JSON result files (default: tta_results_*.json in cwd)")
    p.add_argument("--out_dir", default="figures",
                   help="Directory to write PNG figures (default: figures/)")
    p.add_argument("--metric", default="balanced_acc",
                   choices=["acc", "balanced_acc"],
                   help="Primary metric for bar chart and heatmap")
    return p.parse_args()


def main():
    args = parse_args()

    paths = args.files or sorted(glob.glob("tta_results_*.json"))
    if not paths:
        print("No JSON files found. Pass file paths or run from the project root.")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(paths)} file(s)...")
    df    = load_json_files(paths)
    pc_df = load_per_class(paths)
    print(f"  {len(df)} result rows across "
          f"{df['dataset'].nunique()} dataset(s), "
          f"{df['model'].nunique()} model(s)")

    print("Generating figures...")
    plot_acc_by_strategy(df, out_dir, metric=args.metric)
    plot_heatmap(df, out_dir, metric=args.metric)
    plot_correction(df, out_dir)
    plot_per_class_f1(pc_df, df, out_dir)

    print(f"\nDone. Figures written to '{out_dir}/'")


if __name__ == "__main__":
    main()
