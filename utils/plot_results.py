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
    4. model_comparison      — side-by-side bars: No TTA vs best TTA (mean agg) per model
    5. per_class_f1          — heatmap of per-class F1 for the best strategy
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


_GROUP_COLS = ["dataset", "model", "backbone_mode", "strategy", "aggregation"]
_NUMERIC_COLS = ["acc", "balanced_acc", "macro_f1",
                 "n_correct", "n_wrong", "n_total", "n_views",
                 "n_corrected", "n_corrupted",
                 "total_unc", "aleatoric_unc", "epistemic_unc", "agreement_rate"]


def _aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Average numeric metrics across seeds, adding *_std columns.

    If only one seed is present per group the std columns will be 0.
    Works transparently when there is no 'seed' column at all.
    """
    num_cols = [c for c in _NUMERIC_COLS if c in df.columns]
    grp = df.groupby(_GROUP_COLS, dropna=False)
    agg_mean = grp[num_cols].mean()
    agg_std  = grp[num_cols].std(ddof=1).fillna(0)
    agg_std.columns = [f"{c}_std" for c in agg_std.columns]
    n_seeds = grp.size().rename("n_seeds")
    out = agg_mean.join(agg_std).join(n_seeds).reset_index()
    out["exp"] = out.apply(_exp_label, axis=1)
    return out


# ---------------------------------------------------------------------------
# Plot 1 — Accuracy by strategy & aggregation (grouped bars per experiment)
# ---------------------------------------------------------------------------

def plot_acc_by_strategy(adf: pd.DataFrame, out_dir: Path, metric: str = "balanced_acc"):
    """One figure per dataset; each panel = one (model, backbone_mode).

    Uses seed-aggregated DataFrame (mean ± std).
    """
    std_col = f"{metric}_std"
    datasets = adf["dataset"].unique()

    for ds in sorted(datasets):
        sub = adf[adf["dataset"] == ds]
        experiments = sorted(sub["exp"].unique())

        n_exp = len(experiments)
        fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp + 1, 4.5), sharey=True)
        if n_exp == 1:
            axes = [axes]

        strats  = _ordered(sub["strategy"], STRATEGY_ORDER)
        aggs    = _ordered(sub["aggregation"], AGG_ORDER)
        x       = np.arange(len(strats))
        width   = 0.8 / max(len(aggs), 1)
        colors  = plt.cm.tab10(np.linspace(0, 0.5, len(aggs)))
        multi_seed = sub["n_seeds"].max() > 1

        for ax, exp in zip(axes, experiments):
            edf = sub[sub["exp"] == exp]

            for j, agg in enumerate(aggs):
                vals, errs = [], []
                for strat in strats:
                    row = edf[(edf["strategy"] == strat) & (edf["aggregation"] == agg)]
                    if len(row):
                        vals.append(float(row[metric].values[0]))
                        errs.append(float(row[std_col].values[0]))
                    else:
                        vals.append(float("nan"))
                        errs.append(0.0)
                offset = (j - len(aggs) / 2 + 0.5) * width
                bars = ax.bar(x + offset, vals, width, label=agg, color=colors[j],
                              yerr=errs if multi_seed else None,
                              capsize=3, error_kw={"linewidth": 0.8})
                for bar, v, e in zip(bars, vals, errs):
                    if not np.isnan(v):
                        lbl = f"{v:.3f}±{e:.3f}" if multi_seed else f"{v:.3f}"
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + (e if multi_seed else 0) + 0.003,
                                lbl, ha="center", va="bottom",
                                fontsize=5 if multi_seed else 6, rotation=45)

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
        seed_note = f"  (n_seeds={int(sub['n_seeds'].max())})" if multi_seed else ""
        fig.suptitle(f"Dataset: {ds}  |  {metric}{seed_note}", fontsize=13, y=1.01)
        fig.tight_layout()

        fname = out_dir / f"acc_by_strategy_{ds}_{metric}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Plot 2 — Balanced accuracy heatmap: model × strategy (best agg per cell)
# ---------------------------------------------------------------------------

def plot_heatmap(adf: pd.DataFrame, out_dir: Path, metric: str = "balanced_acc"):
    std_col = f"{metric}_std"
    datasets = adf["dataset"].unique()

    for ds in sorted(datasets):
        sub = adf[adf["dataset"] == ds]
        strats = _ordered(sub["strategy"], STRATEGY_ORDER)
        multi_seed = sub["n_seeds"].max() > 1

        # Best mean value across aggregations for each (exp, strategy)
        pivot_mean = (sub.groupby(["exp", "strategy"])[metric]
                         .max()
                         .unstack("strategy")
                         .reindex(columns=strats))

        # Matching std: pick the row that achieved the max mean
        pivot_std = pd.DataFrame(index=pivot_mean.index, columns=pivot_mean.columns,
                                 dtype=float)
        for exp in pivot_mean.index:
            for strat in strats:
                rows = sub[(sub["exp"] == exp) & (sub["strategy"] == strat)]
                if rows.empty:
                    continue
                best = rows.loc[rows[metric].idxmax()]
                pivot_std.loc[exp, strat] = best[std_col]

        fig, ax = plt.subplots(figsize=(max(4, len(strats) * 1.5),
                                        max(3, len(pivot_mean) * 0.7 + 1)))
        im = ax.imshow(pivot_mean.values, aspect="auto", vmin=0, vmax=1,
                       cmap="RdYlGn")
        plt.colorbar(im, ax=ax, label=metric)

        ax.set_xticks(range(len(strats)))
        ax.set_xticklabels([_strategy_label(s) for s in strats], fontsize=9)
        ax.set_yticks(range(len(pivot_mean)))
        ax.set_yticklabels(pivot_mean.index, fontsize=9)

        for i in range(len(pivot_mean)):
            for j in range(len(strats)):
                v = pivot_mean.values[i, j]
                if np.isnan(v):
                    continue
                s = pivot_std.values[i, j]
                lbl = f"{v:.3f}±{s:.3f}" if multi_seed and not np.isnan(s) else f"{v:.3f}"
                ax.text(j, i, lbl, ha="center", va="center",
                        fontsize=7 if multi_seed else 8, color="black")

        seed_note = f"  (n_seeds={int(sub['n_seeds'].max())})" if multi_seed else ""
        ax.set_title(f"Dataset: {ds}  |  {metric} (best agg){seed_note}", fontsize=11)
        fig.tight_layout()

        fname = out_dir / f"heatmap_{ds}_{metric}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Plot 3 — Corrected vs Corrupted stacked bar (TTA gain/loss)
# ---------------------------------------------------------------------------

def plot_correction(adf: pd.DataFrame, out_dir: Path):
    """Show how many baseline-wrong samples TTA fixes, and how many it breaks."""
    tta_df = adf[adf["strategy"] != "none"]
    if tta_df.empty:
        return

    datasets = tta_df["dataset"].unique()

    for ds in sorted(datasets):
        sub = tta_df[tta_df["dataset"] == ds]
        experiments = sorted(sub["exp"].unique())
        multi_seed = sub["n_seeds"].max() > 1

        n_exp = len(experiments)
        fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp + 1, 4.5), sharey=False)
        if n_exp == 1:
            axes = [axes]

        strats = _ordered(sub["strategy"], [s for s in STRATEGY_ORDER if s != "none"])

        for ax, exp in zip(axes, experiments):
            edf = sub[sub["exp"] == exp]

            # Best agg per strategy by balanced_acc
            best_rows = (edf.sort_values("balanced_acc", ascending=False)
                           .groupby("strategy")
                           .first()
                           .reindex(strats))

            corrected     = best_rows["n_corrected"].fillna(0).astype(float).values
            corrupted     = best_rows["n_corrupted"].fillna(0).astype(float).values
            corrected_err = best_rows["n_corrected_std"].fillna(0).astype(float).values if multi_seed else None
            corrupted_err = best_rows["n_corrupted_std"].fillna(0).astype(float).values if multi_seed else None
            x = np.arange(len(strats))

            ax.bar(x, corrected, label="Corrected", color="steelblue",
                   yerr=corrected_err, capsize=3, error_kw={"linewidth": 0.8})
            ax.bar(x, -corrupted, label="Corrupted", color="tomato",
                   yerr=corrupted_err, capsize=3, error_kw={"linewidth": 0.8})
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
        seed_note = f"  (n_seeds={int(sub['n_seeds'].max())})" if multi_seed else ""
        fig.suptitle(f"Dataset: {ds}  |  TTA Corrections (best agg){seed_note}",
                     fontsize=13, y=1.01)
        fig.tight_layout()

        fname = out_dir / f"correction_{ds}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Plot 4 — Model comparison: No TTA vs best TTA (mean agg), side-by-side bars
# ---------------------------------------------------------------------------

def plot_model_comparison(adf: pd.DataFrame, out_dir: Path, metric: str = "balanced_acc"):
    """Side-by-side bars comparing no-TTA baseline vs best TTA (mean agg) per model."""
    std_col = f"{metric}_std"
    datasets = adf["dataset"].unique()

    for ds in sorted(datasets):
        sub = adf[adf["dataset"] == ds]
        experiments = sorted(sub["exp"].unique())
        multi_seed = sub["n_seeds"].max() > 1

        # Collect baseline (none/mean) and best TTA with mean agg per experiment
        labels = []
        base_means, base_stds = [], []
        tta_means, tta_stds, tta_labels = [], [], []
        for exp in experiments:
            edf = sub[sub["exp"] == exp]
            base = edf[(edf["strategy"] == "none") & (edf["aggregation"] == "mean")]
            if base.empty:
                continue

            tta = edf[(edf["strategy"] != "none") & (edf["aggregation"] == "mean")]
            if tta.empty:
                continue
            best_tta = tta.loc[tta[metric].idxmax()]

            labels.append(exp)
            base_means.append(float(base[metric].values[0]))
            base_stds.append(float(base[std_col].values[0]))
            tta_means.append(float(best_tta[metric]))
            tta_stds.append(float(best_tta[std_col]))
            tta_labels.append(_strategy_label(best_tta["strategy"]))

        if not labels:
            continue

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.8 + 1), 5))
        bars1 = ax.bar(x - width / 2, base_means, width, label="No TTA",
                        color="lightcoral",
                        yerr=base_stds if multi_seed else None,
                        capsize=3, error_kw={"linewidth": 0.8})
        bars2 = ax.bar(x + width / 2, tta_means, width, label="Best TTA (mean agg)",
                        color="steelblue",
                        yerr=tta_stds if multi_seed else None,
                        capsize=3, error_kw={"linewidth": 0.8})

        for bar, e in zip(bars1, base_stds):
            v = bar.get_height()
            lbl = f"{v:.3f}±{e:.3f}" if multi_seed else f"{v:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (e if multi_seed else 0) + 0.005,
                    lbl, ha="center", va="bottom", fontsize=7)
        for bar, e, strat_lbl in zip(bars2, tta_stds, tta_labels):
            v = bar.get_height()
            val_str = f"{v:.3f}±{e:.3f}" if multi_seed else f"{v:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (e if multi_seed else 0) + 0.005,
                    f"{val_str}\n({strat_lbl})", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)
        seed_note = f"  (n_seeds={int(sub['n_seeds'].max())})" if multi_seed else ""
        ax.set_title(f"Dataset: {ds}  |  No TTA vs Best TTA (mean agg){seed_note}",
                     fontsize=12)
        fig.tight_layout()

        fname = out_dir / f"model_comparison_{ds}_{metric}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Plot 5 — Per-class F1 heatmap for best strategy per experiment
# ---------------------------------------------------------------------------

def plot_per_class_f1(pc_df: pd.DataFrame, adf: pd.DataFrame, out_dir: Path):
    """Per-class F1 heatmap for the best strategy per experiment.

    When multiple seeds are present, per-class F1 is averaged across seeds.
    """
    if pc_df.empty:
        return

    # Average per-class F1 across seeds
    pc_group_cols = ["dataset", "model", "backbone_mode", "strategy", "aggregation", "class"]
    pc_agg = (pc_df.groupby(pc_group_cols, dropna=False)["f1"]
                   .agg(["mean", "std"])
                   .reset_index()
                   .rename(columns={"mean": "f1", "std": "f1_std"}))
    pc_agg["f1_std"] = pc_agg["f1_std"].fillna(0)

    datasets = pc_agg["dataset"].unique()

    for ds in sorted(datasets):
        pc_sub  = pc_agg[pc_agg["dataset"] == ds]
        sum_sub = adf[adf["dataset"] == ds]

        if sum_sub.empty:
            continue

        multi_seed = sum_sub["n_seeds"].max() > 1

        # Identify best (strategy, agg) per experiment by balanced_acc
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
            matched = pc_sub[mask].set_index("class").reindex(classes)
            vals = matched["f1"].values.reshape(-1, 1)
            stds = matched["f1_std"].values if multi_seed else None

            im = ax.imshow(vals.T, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
            ax.set_yticks([0])
            ax.set_yticklabels([f"{brow['model']}\n{brow['backbone_mode']}\n"
                                 f"{_strategy_label(brow['strategy'])}/"
                                 f"{brow['aggregation']}"], fontsize=7)
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)

            for j, v in enumerate(vals.flatten()):
                if np.isnan(v):
                    continue
                if multi_seed and stds is not None:
                    lbl = f"{v:.2f}±{stds[j]:.2f}"
                else:
                    lbl = f"{v:.2f}"
                ax.text(j, 0, lbl, ha="center", va="center",
                        fontsize=5 if multi_seed else 6, color="black")

        fig.colorbar(im, ax=axes[-1], label="F1")
        seed_note = f"  (n_seeds={int(sum_sub['n_seeds'].max())})" if multi_seed else ""
        fig.suptitle(f"Dataset: {ds}  |  Per-class F1 (best strategy){seed_note}",
                     fontsize=12, y=1.02)
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

    n_seeds = df["seed"].nunique() if "seed" in df.columns else 1
    seeds_str = f", {n_seeds} seed(s)" if n_seeds > 1 else ""
    print(f"  {len(df)} result rows across "
          f"{df['dataset'].nunique()} dataset(s), "
          f"{df['model'].nunique()} model(s){seeds_str}")

    # Aggregate across seeds: mean ± std
    adf = _aggregate_seeds(df)
    if n_seeds > 1:
        print(f"  Aggregated over seeds → {len(adf)} unique (dataset, model, mode, strategy, agg) combos")

    print("Generating figures...")
    plot_acc_by_strategy(adf, out_dir, metric=args.metric)
    plot_heatmap(adf, out_dir, metric=args.metric)
    plot_correction(adf, out_dir)
    plot_model_comparison(adf, out_dir, metric=args.metric)
    plot_per_class_f1(pc_df, adf, out_dir)

    print(f"\nDone. Figures written to '{out_dir}/'")


if __name__ == "__main__":
    main()
