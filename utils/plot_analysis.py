#!/usr/bin/env python3
"""
Publication-quality analysis plots for TTA experiments.

Reads from results/tta_results.csv and results/tta_per_class.csv.
Aggregates across seeds (mean +/- SEM) and uses paired t-tests for significance.

Usage:
    python utils/plot_analysis.py
    python utils/plot_analysis.py --out_dir figures/analysis
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Style & constants
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", font_scale=1.1)

MODEL_TYPE_COLORS = {"histology": "#4C72B0", "general": "#DD8452", "equivariant": "#55A868"}
MODEL_TYPE_ORDER = ["histology", "general", "equivariant"]

STRATEGY_ORDER = ["none", "flips", "d4"]
STRATEGY_LABELS = {"none": "No TTA", "flips": "Flips (4v)", "d4": "D4 (8v)"}

DATASET_LABELS = {
    "tcga-ut": "TCGA-UT (31 cls)",
    "nct-crc-100k": "NCT-CRC-HE-100K (9 cls)",
    "nct-crc-nonorm": "NCT-CRC-HE-NoNorm (9 cls)",
    "mhist": "MHIST (2 cls)",
}
DATASET_ORDER = ["tcga-ut", "nct-crc-100k", "nct-crc-nonorm", "mhist"]

MODEL_ORDER = [
    # histology
    "gigapath", "hoptimus", "phikon", "phikon2", "uni", "uni2", "virchow", "virchow2",
    # general
    "convnextv2_tiny", "convnextv2_base", "dinov2_s", "dinov2_base",
    # equivariant
    "d4wrn",
]

MODEL_SHORT = {
    "gigapath": "GP", "hoptimus": "HO1", "phikon": "PH", "phikon2": "PH2",
    "uni": "UNI", "uni2": "UNI2", "virchow": "VR", "virchow2": "VR2",
    "convnextv2_tiny": "CNT", "convnextv2_base": "CNB", "dinov2_s": "D2S", 
    "dinov2_base": "D2B", "d4wrn": "D4W",
}

FM_META = {
    "phikon":   {"params_m": 86,   "pretrain_tiles_m": 40},
    "phikon2":  {"params_m": 300,  "pretrain_tiles_m": 456},
    "uni":      {"params_m": 300,  "pretrain_tiles_m": 100},
    "uni2":     {"params_m": 681,  "pretrain_tiles_m": 200},
    "virchow":  {"params_m": 632,  "pretrain_tiles_m": 1500},
    "virchow2": {"params_m": 632,  "pretrain_tiles_m": 3100},
    "gigapath": {"params_m": 1100, "pretrain_tiles_m": 1300},
    "hoptimus": {"params_m": 1100, "pretrain_tiles_m": 1000},
}

DS_COLORS = {"tcga-ut": "#E24A33", "nct-crc-100k": "#348ABD", "nct-crc-nonorm": "#988ED5", "mhist": "#8EBA42"}
MODE_MARKERS = {"frozen": "o", "finetuned": "s"}

SAVEKW = dict(dpi=300, bbox_inches="tight", facecolor="white")

# Columns that identify a unique experiment (excluding seed)
CONFIG_COLS = ["model", "dataset", "backbone_mode", "train_augment"]


# ---------------------------------------------------------------------------
# Data loading & helpers
# ---------------------------------------------------------------------------

def load_data(results_csv: str = "results/tta_results.csv",
              per_class_csv: str = "results/tta_per_class.csv"):
    df = pd.read_csv(results_csv)
    pc = pd.read_csv(per_class_csv)
    # Filter to augmented training only (noaug experiments excluded from analysis)
    df = df[(df["train_augment"] == True)&(df["train_subset"].isna())&(df["dataset"] != "mhist")].copy()
    pc = pc[(pc["train_augment"] == True)].copy()
    return df, pc


def load_ablation_data(results_csv: str = "results/tta_results.csv"):
    """Load dataset-size ablation rows (non-null train_subset) plus matching full runs."""
    df = pd.read_csv(results_csv)
    df = df[df["train_augment"] == True].copy()
    subset_rows = df[df["train_subset"].notna()]
    # Identify which (model, dataset, backbone_mode) combos have ablation data
    ablation_configs = subset_rows[["model", "dataset", "backbone_mode"]].drop_duplicates()
    # Also grab matching full runs (train_subset is NaN) with same seeds
    ablation_seeds = subset_rows["seed"].unique()
    full_rows = df[df["train_subset"].isna()]
    full_matched = full_rows.merge(ablation_configs, on=["model", "dataset", "backbone_mode"])
    full_matched = full_matched[full_matched["seed"].isin(ablation_seeds)]
    full_matched["train_subset"] = full_matched["train_subset"].fillna(-1)  # sentinel for "full"
    subset_rows = subset_rows.copy()
    return pd.concat([subset_rows, full_matched], ignore_index=True)


def _get_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Get baseline (none/mean) rows with renamed metric columns."""
    base = df[(df["strategy"] == "none") & (df["aggregation"] == "mean")].copy()
    return base


def _get_tta(df: pd.DataFrame, strategy: str = "d4", agg: str = "mean") -> pd.DataFrame:
    """Get TTA rows for a specific strategy and aggregation."""
    tta = df[(df["strategy"] == strategy) & (df["aggregation"] == agg)].copy()
    return tta


def _compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute TTA delta for each experiment per seed (d4/mean vs none/mean)."""
    base = _get_baseline(df)
    best = _get_tta(df, strategy="d4", agg="mean")

    merge_cols = CONFIG_COLS + ["seed"]
    merged = base[merge_cols + ["balanced_acc", "acc", "model_type"]].merge(
        best[merge_cols + ["balanced_acc", "acc", "strategy",
                           "epistemic_unc", "agreement_rate",
                           "n_corrected", "n_corrupted"]],
        on=merge_cols, suffixes=("_base", "_tta"),
    )
    merged["delta_bacc"] = merged["balanced_acc_tta"] - merged["balanced_acc_base"]
    merged["delta_acc"] = merged["acc_tta"] - merged["acc_base"]
    return merged


def _seed_agg(data: pd.DataFrame, group_cols: list, val_cols: list) -> pd.DataFrame:
    """Aggregate across seeds: compute mean, SEM, and count for value columns."""
    agg_dict = {}
    for col in val_cols:
        agg_dict[col] = ["mean", "sem", "count"]
    result = data.groupby(group_cols, observed=True).agg(agg_dict)
    # Flatten multi-level columns
    result.columns = [f"{col}_{stat}" for col, stat in result.columns]
    return result.reset_index()


def _ttest_pval(values):
    """One-sample t-test on values (e.g. per-seed deltas). Returns p-value."""
    values = values.dropna()
    if len(values) < 2:
        return np.nan
    _, p = stats.ttest_1samp(values, 0)
    return p


def _sig_stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def _order_models(data, col="model"):
    """Return data with model column ordered by MODEL_ORDER."""
    present = [m for m in MODEL_ORDER if m in data[col].values]
    cat = pd.CategoricalDtype(categories=present, ordered=True)
    data = data.copy()
    data[col] = data[col].astype(cat)
    return data


def _ds_label(ds):
    return DATASET_LABELS.get(ds, ds)


def _strat_label(s):
    return STRATEGY_LABELS.get(s, s)


def _primary_config(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to one canonical config per model.

    frozen + aug for all models, except d4wrn which only has finetuned + aug.
    """
    frozen_aug = df[(df["backbone_mode"] == "frozen") & (df["train_augment"] == True)]
    ft_aug = df[(df["backbone_mode"] == "finetuned") & (df["train_augment"] == True)]
    # Use frozen+aug where available, fall back to finetuned+aug (e.g. d4wrn)
    frozen_models = frozen_aug["model"].unique()
    ft_only = ft_aug[~ft_aug["model"].isin(frozen_models)]
    return pd.concat([frozen_aug, ft_only], ignore_index=True)


def _ds_mode_legend(ax, deltas_df):
    """Add standard dataset color + frozen/finetuned marker legend."""
    from matplotlib.lines import Line2D
    handles = []
    for ds in DATASET_ORDER:
        if ds in deltas_df["dataset"].values:
            handles.append(Line2D([0], [0], marker="o", color=DS_COLORS[ds],
                                  markerfacecolor=DS_COLORS[ds], markersize=7,
                                  linestyle="None", label=_ds_label(ds)))
    handles.append(Line2D([0], [0], marker="o", color="gray", markerfacecolor="gray",
                          markersize=7, linestyle="None", label="Frozen"))
    handles.append(Line2D([0], [0], marker="s", color="gray", markerfacecolor="gray",
                          markersize=7, linestyle="None", label="Finetuned"))
    ax.legend(handles=handles, fontsize=8)


# ---------------------------------------------------------------------------
# Plot 1 — TTA consistently improves all model families
# ---------------------------------------------------------------------------

def _plot1_row(axes, base, best, datasets, model_list, row_label):
    """Helper: draw one row of plot 1 (baseline vs D4 TTA bars, mean +/- SEM across seeds)."""
    for col, (ax, ds) in enumerate(zip(axes, datasets)):
        b = base[(base["dataset"] == ds) & (base["model"].isin(model_list))]
        t = best[(best["dataset"] == ds) & (best["model"].isin(model_list))]
        b = _order_models(b)
        t = _order_models(t)

        models = [m for m in MODEL_ORDER if m in b["model"].values]
        if not models:
            ax.set_visible(False)
            continue

        x = np.arange(len(models))
        w = 0.35

        base_means, base_errs = [], []
        tta_means, tta_errs = [], []
        colors, pvals = [], []

        for m in models:
            brows = b[b["model"] == m]
            trows = t[t["model"] == m]
            base_means.append(brows["balanced_acc"].mean())
            base_errs.append(brows["balanced_acc"].sem() if len(brows) > 1 else 0)
            tta_means.append(trows["balanced_acc"].mean())
            tta_errs.append(trows["balanced_acc"].sem() if len(trows) > 1 else 0)
            colors.append(MODEL_TYPE_COLORS.get(
                brows["model_type"].values[0], "gray") if not brows.empty else "gray")

            # Per-seed deltas for t-test
            if not brows.empty and not trows.empty:
                merged = brows[["seed", "balanced_acc"]].merge(
                    trows[["seed", "balanced_acc"]], on="seed", suffixes=("_b", "_t"))
                seed_deltas = merged["balanced_acc_t"] - merged["balanced_acc_b"]
                pvals.append(_ttest_pval(seed_deltas))
            else:
                pvals.append(np.nan)

        ax.bar(x - w / 2, base_means, w, yerr=base_errs,
               color=[c + "80" for c in colors], edgecolor=colors, linewidth=0.8,
               label="No TTA", capsize=3, error_kw={"linewidth": 1})
        ax.bar(x + w / 2, tta_means, w, yerr=tta_errs,
               color=colors, edgecolor=colors, linewidth=0.8,
               label="D4 TTA", capsize=3, error_kw={"linewidth": 1})

        for i, (bv, tv, p) in enumerate(zip(base_means, tta_means, pvals)):
            if not np.isnan(bv) and not np.isnan(tv):
                delta = (tv - bv) * 100
                stars = _sig_stars(p)
                label = f"{delta:+.2f}%{stars}"
                ax.text(x[i] + w / 2, tv + tta_errs[i] + 0.003, label,
                        ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Balanced Accuracy" if col == 0 else "")
        ax.legend(fontsize=7, loc="lower left")

        all_vals = [v for v in base_means + tta_means if not np.isnan(v)]
        all_errs = base_errs + tta_errs
        if all_vals:
            lo = min(all_vals) - max(all_errs) - 0.03
            hi = max(all_vals) + max(all_errs) + 0.04
            ax.set_ylim(max(0, lo), min(1, hi))

    axes[0].set_ylabel(f"{row_label}\nBalanced Accuracy", fontsize=10)


def plot1_tta_helps_all(df: pd.DataFrame, out_dir: Path):
    """Grouped bar: baseline vs D4 TTA for histology FMs (frozen) + d4wrn."""
    primary = _primary_config(df)
    base = _get_baseline(primary)
    best = _get_tta(primary, strategy="d4", agg="mean")

    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    n_ds = len(datasets)

    hist_models = [m for m in MODEL_ORDER if m in
                   df[df["model_type"] == "histology"]["model"].unique()] + ["d4wrn"]

    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5), sharey=False)
    if n_ds == 1:
        axes = [axes]

    _plot1_row(axes, base, best, datasets, hist_models, "Histology FMs")
    for ax, ds in zip(axes, datasets):
        ax.set_title(_ds_label(ds), fontsize=11)

    fig.suptitle("TTA Consistently Improves Histology Foundation Models", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "plot1_tta_helps_all.png", **SAVEKW)
    plt.close(fig)
    print("  Saved plot1_tta_helps_all.png")


def plot1b_general_models(df: pd.DataFrame, out_dir: Path):
    """Grouped bar: baseline vs D4 TTA for general models, frozen vs finetuned rows."""
    # Exclude MHIST (no finetuned results for general models)
    datasets = [d for d in DATASET_ORDER
                if d in df["dataset"].unique() and d != "mhist"]
    n_ds = len(datasets)

    general_models = [m for m in MODEL_ORDER if m in
                      df[df["model_type"] == "general"]["model"].unique()]

    frozen = df[df["backbone_mode"] == "frozen"]
    finetuned = df[df["backbone_mode"] == "finetuned"]

    fig, axes = plt.subplots(2, n_ds, figsize=(6 * n_ds, 10), sharey=False)
    if n_ds == 1:
        axes = axes.reshape(2, 1)

    for col, ds in enumerate(datasets):
        axes[0, col].set_title(_ds_label(ds), fontsize=11)

    base_ft = _get_baseline(finetuned)
    best_ft = _get_tta(finetuned, strategy="d4", agg="mean")
    _plot1_row(axes[0], base_ft, best_ft, datasets, general_models, "Finetuned")

    base_fr = _get_baseline(frozen)
    best_fr = _get_tta(frozen, strategy="d4", agg="mean")
    _plot1_row(axes[1], base_fr, best_fr, datasets, general_models, "Frozen")

    fig.suptitle("TTA on General Purpose Models: Finetuned vs Frozen", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "plot1b_general_models.png", **SAVEKW)
    plt.close(fig)
    print("  Saved plot1b_general_models.png")


# ---------------------------------------------------------------------------
# Plot 2 — Histology FMs leave performance on the table
# ---------------------------------------------------------------------------

def plot2_histology_fm_delta(df: pd.DataFrame, out_dir: Path):
    """Grouped box+strip: datasets on x-axis, side-by-side boxes for each model type."""
    from matplotlib.patches import Patch

    # Use global MODEL_TYPE_COLORS

    deltas = _compute_deltas(df)
    # Aggregate across seeds (each model×dataset×backbone_mode = one point)
    group = CONFIG_COLS + ["model_type"]
    agg = deltas.groupby(group, observed=True).agg(
        delta_pct_mean=("delta_bacc", lambda x: x.mean() * 100),
    ).reset_index()

    datasets_present = [ds for ds in DATASET_ORDER if ds in agg["dataset"].values]
    box_types = [mt for mt in ["histology", "general"] if mt in agg["model_type"].values]
    has_equivariant = "equivariant" in agg["model_type"].values

    box_width = 0.35
    # Offsets: two boxes centered around integer x positions
    offsets = {mt: (i - (len(box_types) - 1) / 2) * (box_width + 0.08)
               for i, mt in enumerate(box_types)}

    fig, ax = plt.subplots(figsize=(max(7, len(datasets_present) * 2.5), 5))
    rng = np.random.default_rng(42)

    for mt in box_types:
        color = MODEL_TYPE_COLORS[mt]
        positions = []
        box_data = []
        for di, ds in enumerate(datasets_present):
            vals = agg[(agg["dataset"] == ds) & (agg["model_type"] == mt)]["delta_pct_mean"].values
            if len(vals) == 0:
                continue
            pos = di + offsets[mt]
            positions.append(pos)
            box_data.append(vals)

        if not box_data:
            continue

        bp = ax.boxplot(box_data, positions=positions, widths=box_width,
                        patch_artist=True, showfliers=False, zorder=2,
                        medianprops=dict(color="black", linewidth=1.5),
                        whiskerprops=dict(color=color, alpha=0.7),
                        capprops=dict(color=color, alpha=0.7))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
            patch.set_edgecolor(color)

        # Overlay individual points with jitter
        for pos, vals in zip(positions, box_data):
            jitter = rng.uniform(-box_width * 0.3, box_width * 0.3, size=len(vals))
            ax.scatter(pos + jitter, vals, color=color, s=25, alpha=0.8,
                       edgecolors="white", linewidths=0.5, zorder=3)

    # Equivariant: single diamond marker per dataset
    if has_equivariant:
        eq_color = MODEL_TYPE_COLORS["equivariant"]
        eq = agg[agg["model_type"] == "equivariant"]
        for di, ds in enumerate(datasets_present):
            vals = eq[eq["dataset"] == ds]["delta_pct_mean"].values
            for v in vals:
                ax.scatter(di, v, color=eq_color, s=80, marker="D", alpha=0.9,
                           edgecolors="white", linewidths=0.8, zorder=4)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(list(range(len(datasets_present))))
    ax.set_xticklabels([_ds_label(ds) for ds in datasets_present], fontsize=10)
    ax.set_ylabel("TTA Δ Balanced Accuracy (pp)")
    ax.set_title("TTA Benefit by Dataset and Model Type", fontsize=13)

    # Legend
    from matplotlib.lines import Line2D
    handles = [Patch(facecolor=MODEL_TYPE_COLORS[mt], alpha=0.4,
                     edgecolor=MODEL_TYPE_COLORS[mt], label=mt.title())
               for mt in box_types]
    if has_equivariant:
        handles.append(Line2D([0], [0], marker="D", color=MODEL_TYPE_COLORS["equivariant"],
                              markerfacecolor=MODEL_TYPE_COLORS["equivariant"], markersize=8,
                              linestyle="None", label="Equivariant (D4WRN)"))
    ax.legend(handles=handles, loc="best", fontsize=9, title="Model Type",
              title_fontsize=10)

    fig.tight_layout()
    fig.savefig(out_dir / "plot2_histology_fm_delta.png", **SAVEKW)
    plt.close(fig)
    print("  Saved plot2_histology_fm_delta.png")


# ---------------------------------------------------------------------------
# Plot 3 — Frozen backbones benefit more than finetuned
# ---------------------------------------------------------------------------

def plot3_frozen_vs_finetuned(df: pd.DataFrame, out_dir: Path):
    """Paired dot plot: frozen vs finetuned TTA delta for dinov2_s (mean across seeds)."""
    dinov2 = df[df["model"] == "dinov2_s"]
    deltas = _compute_deltas(dinov2)

    # Aggregate across seeds
    group = CONFIG_COLS + ["model_type"]
    agg = deltas.groupby(group, observed=True).agg(
        delta_pct_mean=("delta_bacc", lambda x: x.mean() * 100),
        delta_pct_sem=("delta_bacc", lambda x: x.sem() * 100 if len(x) > 1 else 0),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))

    for _, row in agg.iterrows():
        ds = row["dataset"]
        mode = row["backbone_mode"]
        xi = 0 if mode == "frozen" else 1
        ax.errorbar(xi, row["delta_pct_mean"], yerr=row["delta_pct_sem"],
                     fmt="o", markersize=8, color=MODEL_TYPE_COLORS["general"],
                     capsize=3, elinewidth=1, zorder=5)
        ax.annotate(f"{ds}", (xi, row["delta_pct_mean"]),
                    fontsize=7, xytext=(-10, 5), textcoords="offset points", ha="right")

    # Draw connecting lines between frozen/finetuned pairs
    for ds in agg["dataset"].unique():
        frozen = agg[(agg["dataset"] == ds) & (agg["backbone_mode"] == "frozen")]
        finetuned = agg[(agg["dataset"] == ds) & (agg["backbone_mode"] == "finetuned")]
        if not frozen.empty and not finetuned.empty:
            ax.plot([0, 1],
                    [frozen["delta_pct_mean"].values[0], finetuned["delta_pct_mean"].values[0]],
                    "-", color="gray", alpha=0.4, linewidth=1)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Frozen (linear probe)", "Finetuned"], fontsize=11)
    ax.set_ylabel("TTA Δ Balanced Accuracy (%)")
    ax.set_title("Frozen Backbones Benefit More from TTA\n(DINOv2-S)", fontsize=13)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlim(-0.5, 1.5)

    fig.tight_layout()
    fig.savefig(out_dir / "plot3_frozen_vs_finetuned.png", **SAVEKW)
    plt.close(fig)
    print("  Saved plot3_frozen_vs_finetuned.png")


# ---------------------------------------------------------------------------
# Plot 4 — (removed: aug vs noaug experiment excluded from analysis)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Plot 5 — D4 is the sweet spot strategy
# ---------------------------------------------------------------------------

def plot5_strategy_progression(df: pd.DataFrame, out_dir: Path):
    """Line plot: balanced_acc across TTA strategies (mean agg), averaged across seeds."""
    sub = df[df["aggregation"] == "mean"].copy()
    sub["strategy"] = pd.Categorical(sub["strategy"], categories=STRATEGY_ORDER, ordered=True)

    datasets = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        dsub = sub[sub["dataset"] == ds]
        for model in MODEL_ORDER:
            mdata = dsub[dsub["model"] == model]
            if mdata.empty:
                continue
            mtype = mdata["model_type"].values[0]
            # Average over seeds + backbone_mode/train_augment variants
            avg = mdata.groupby("strategy", observed=True)["balanced_acc"].agg(["mean", "sem"])
            color = MODEL_TYPE_COLORS.get(mtype, "gray")
            ls = "-" if mtype == "histology" else ("--" if mtype == "general" else ":")
            ax.errorbar(range(len(avg)), avg["mean"].values, yerr=avg["sem"].values,
                        marker="o", markersize=5, label=model, color=color,
                        linestyle=ls, linewidth=1.5, alpha=0.8, capsize=2)

        ax.set_xticks(range(len(STRATEGY_ORDER)))
        ax.set_xticklabels([_strat_label(s) for s in STRATEGY_ORDER], fontsize=9)
        ax.set_title(_ds_label(ds), fontsize=11)
        ax.set_ylabel("Balanced Accuracy" if ax is axes[0] else "")

    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, title="Model")
    fig.suptitle("D4 Is the Sweet Spot — Diminishing Returns Beyond 8 Views", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "plot5_strategy_progression.png", **SAVEKW)
    plt.close(fig)
    print("  Saved plot5_strategy_progression.png")


# ---------------------------------------------------------------------------
# Plot 6 — TTA corrects more than it corrupts
# ---------------------------------------------------------------------------

def plot6_correction_scatter(df: pd.DataFrame, out_dir: Path):
    """Dual panel: (A) correction rate vs corruption rate, (B) net correction ratio by model."""
    from matplotlib.lines import Line2D

    base = df[(df["strategy"] == "none") & (df["aggregation"] == "mean")].copy()
    tta = df[(df["strategy"] == "d4") & (df["aggregation"] == "mean")].copy()
    tta = tta.dropna(subset=["n_corrected", "n_corrupted"])

    merge_cols = CONFIG_COLS + ["seed"]
    tta = tta.merge(
        base[merge_cols + ["n_wrong", "n_correct"]].rename(
            columns={"n_wrong": "base_wrong", "n_correct": "base_correct"}),
        on=merge_cols,
    )
    tta["correction_rate"] = tta["n_corrected"] / tta["base_wrong"]
    tta["corruption_rate"] = tta["n_corrupted"] / tta["base_correct"]
    tta["net_ratio"] = tta["n_corrected"] / tta["n_corrupted"].clip(lower=1)

    # Aggregate across seeds for plotting
    agg_cols = CONFIG_COLS + ["model_type"]
    tta_agg = tta.groupby(agg_cols, observed=True).agg(
        corr_rate_mean=("correction_rate", "mean"),
        corr_rate_sem=("correction_rate", lambda x: x.sem() if len(x) > 1 else 0),
        corrupt_rate_mean=("corruption_rate", "mean"),
        corrupt_rate_sem=("corruption_rate", lambda x: x.sem() if len(x) > 1 else 0),
        net_ratio_mean=("net_ratio", "mean"),
        net_ratio_sem=("net_ratio", lambda x: x.sem() if len(x) > 1 else 0),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel A: correction rate vs corruption rate scatter ---
    legend_handles = []
    for mtype in MODEL_TYPE_ORDER:
        sub = tta_agg[tta_agg["model_type"] == mtype]
        if sub.empty:
            continue
        h = ax1.scatter([], [], c=MODEL_TYPE_COLORS[mtype], s=50, label=mtype.title())
        legend_handles.append(h)

    for mtype in MODEL_TYPE_ORDER:
        color = MODEL_TYPE_COLORS[mtype]
        for mode, marker in MODE_MARKERS.items():
            sub = tta_agg[(tta_agg["model_type"] == mtype) & (tta_agg["backbone_mode"] == mode)]
            if sub.empty:
                continue
            ax1.errorbar(sub["corrupt_rate_mean"] * 100, sub["corr_rate_mean"] * 100,
                         xerr=sub["corrupt_rate_sem"] * 100, yerr=sub["corr_rate_sem"] * 100,
                         fmt=marker, markersize=7, alpha=0.7, color=color,
                         markerfacecolor=color, markeredgecolor=color,
                         capsize=2, elinewidth=1, zorder=3)

    legend_handles.append(Line2D([0], [0], marker="o", color="gray", markerfacecolor="gray",
                                 markersize=7, linestyle="None", label="Frozen"))
    legend_handles.append(Line2D([0], [0], marker="s", color="gray", markerfacecolor="gray",
                                 markersize=7, linestyle="None", label="Finetuned"))

    lim = max(tta_agg["corr_rate_mean"].max(), tta_agg["corrupt_rate_mean"].max()) * 100 * 1.1
    ax1.plot([0, lim], [0, lim], "k--", alpha=0.4, linewidth=1)

    ax1.set_xlabel("Corruption Rate (% of correct predictions broken)")
    ax1.set_ylabel("Correction Rate (% of wrong predictions fixed)")
    ax1.set_title("(A) TTA Fixes More Than It Breaks", fontsize=11)
    ax1.legend(handles=legend_handles, fontsize=7, loc="upper left")

    # --- Panel B: net correction ratio by model (canonical config) ---
    primary = _primary_config(tta_agg)
    primary = _order_models(primary)

    models = [m for m in MODEL_ORDER if m in primary["model"].values]
    x = np.arange(len(models))

    n_ds = len(DS_COLORS)
    w = 0.8 / max(n_ds, 1)

    for i, (ds, color) in enumerate(DS_COLORS.items()):
        vals, errs = [], []
        for m in models:
            row = primary[(primary["model"] == m) & (primary["dataset"] == ds)]
            vals.append(float(row["net_ratio_mean"].mean()) if not row.empty else 0)
            errs.append(float(row["net_ratio_sem"].mean()) if not row.empty else 0)
        offset = (i - n_ds / 2 + 0.5) * w
        bars = ax2.bar(x + offset, vals, w, yerr=errs, label=_ds_label(ds),
                       color=color, alpha=0.8, edgecolor="white", capsize=2)


    ax2.axhline(1, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Net Correction Ratio (corrected / corrupted)")
    ax2.set_title("(B) Net Correction Ratio by Model", fontsize=11)
    ax2.legend(fontsize=7, loc="upper right")

    fig.suptitle("TTA Corrects More Predictions Than It Corrupts", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "plot6_correction_scatter.png", **SAVEKW)
    plt.close(fig)
    print("  Saved plot6_correction_scatter.png")


# ---------------------------------------------------------------------------
# Plot 7 — Weaker baselines gain more from TTA
# ---------------------------------------------------------------------------

def plot7_baseline_vs_delta(df: pd.DataFrame, out_dir: Path):
    """Scatter: baseline balanced_acc vs TTA delta (mean across seeds per config)."""
    deltas = _compute_deltas(df)

    # Aggregate across seeds
    group = CONFIG_COLS + ["model_type"]
    agg = deltas.groupby(group, observed=True).agg(
        bacc_base_mean=("balanced_acc_base", "mean"),
        delta_pct_mean=("delta_bacc", lambda x: x.mean() * 100),
        delta_pct_sem=("delta_bacc", lambda x: x.sem() * 100 if len(x) > 1 else 0),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))

    for ds, color in DS_COLORS.items():
        for mode, marker in MODE_MARKERS.items():
            sub = agg[(agg["dataset"] == ds) & (agg["backbone_mode"] == mode)]
            if sub.empty:
                continue
            ax.errorbar(sub["bacc_base_mean"], sub["delta_pct_mean"],
                        yerr=sub["delta_pct_sem"],
                        fmt=marker, markersize=7, alpha=0.8, color=color,
                        markerfacecolor=color, markeredgecolor=color,
                        capsize=2, elinewidth=1, zorder=3)
            for _, row in sub.iterrows():
                ax.annotate(MODEL_SHORT.get(row["model"], row["model"]),
                            (row["bacc_base_mean"], row["delta_pct_mean"]),
                            fontsize=5, alpha=0.6, ha="left",
                            xytext=(5, 0), textcoords="offset points")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
    ax.set_xlabel("Baseline Balanced Accuracy (No TTA)")
    ax.set_ylabel("TTA Δ Balanced Accuracy (%)")
    ax.set_title("Weaker Baselines Gain More from TTA", fontsize=13)
    _ds_mode_legend(ax, agg)

    fig.tight_layout()
    fig.savefig(out_dir / "plot7_baseline_vs_delta.png", **SAVEKW)
    plt.close(fig)
    print("  Saved plot7_baseline_vs_delta.png")


# ---------------------------------------------------------------------------
# Plot 8 — TTA disproportionately helps hard classes
# ---------------------------------------------------------------------------

def plot8_per_class_delta(pc: pd.DataFrame, out_dir: Path):
    """Two-row analysis: (top) baseline F1 quartiles vs F1 delta,
    (bottom) class frequency vs F1 delta.  Per dataset, excluding MHIST.

    Filters to canonical config. All seeds contribute to the binned statistics,
    giving more robust SEM estimates.
    """
    primary_models = _primary_config(
        pc[["model", "dataset", "backbone_mode", "train_augment", "model_type"]]
        .drop_duplicates()
    )
    pc_filt = pc.merge(
        primary_models[["model", "dataset", "backbone_mode", "train_augment"]],
        on=["model", "dataset", "backbone_mode", "train_augment"],
    )

    base_pc = pc_filt[(pc_filt["strategy"] == "none") & (pc_filt["aggregation"] == "mean")].copy()
    tta_pc = pc_filt[(pc_filt["strategy"] == "d4") & (pc_filt["aggregation"] == "mean")].copy()

    merge_cols = ["model", "dataset", "backbone_mode", "train_augment", "seed", "class_name"]
    merged = base_pc[merge_cols + ["f1", "support"]].merge(
        tta_pc[merge_cols + ["f1"]],
        on=merge_cols, suffixes=("_base", "_tta"),
    )
    merged["delta_f1"] = merged["f1_tta"] - merged["f1_base"]
    merged = merged.dropna(subset=["f1_base", "delta_f1"])

    # Compute class frequency as % of test set per dataset×seed×model
    group_cols = ["model", "dataset", "backbone_mode", "train_augment", "seed"]
    merged["total_support"] = merged.groupby(group_cols)["support"].transform("sum")
    merged["class_pct"] = merged["support"] / merged["total_support"] * 100

    # Skip datasets with too few classes for meaningful quartile binning
    datasets = [d for d in DATASET_ORDER
                if d in merged["dataset"].unique() and d != "mhist"]

    n_ds = len(datasets)
    fig, axes = plt.subplots(2, n_ds, figsize=(6 * n_ds, 9), sharey="row")
    if n_ds == 1:
        axes = axes.reshape(2, 1)

    for col, ds in enumerate(datasets):
        sub = merged[merged["dataset"] == ds].copy()
        if sub.empty:
            continue

        color = DS_COLORS.get(ds, "gray")

        # --- Row 0: Baseline F1 quartiles ---
        ax = axes[0, col]
        sub["bin_f1"] = pd.qcut(sub["f1_base"], q=4, duplicates="drop")
        binned = sub.groupby("bin_f1", observed=True)["delta_f1"].agg(["mean", "sem", "count"])
        binned["mid"] = [interval.mid for interval in binned.index]
        binned = binned.sort_values("mid")

        x = np.arange(len(binned))
        labels = [f"{interval.left:.2f}\u2013{interval.right:.2f}" for interval in binned.index]
        ax.bar(x, binned["mean"], yerr=binned["sem"], width=0.6,
               color=color, alpha=0.7, edgecolor=color, linewidth=0.8,
               capsize=3, error_kw={"linewidth": 1})
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("Baseline F1 Quartile")
        if col == 0:
            ax.set_ylabel("Mean Δ F1 (D4 − No TTA)")
        ax.set_title(_ds_label(ds), fontsize=11)
        for i, n in enumerate(binned["count"]):
            ax.text(i, binned["mean"].iloc[i] + binned["sem"].iloc[i] + 0.001,
                    f"n={int(n)}", ha="center", va="bottom", fontsize=6, alpha=0.6)

        # --- Row 1: Class frequency quartiles ---
        ax2 = axes[1, col]
        sub["bin_pct"] = pd.qcut(sub["class_pct"], q=4, duplicates="drop")
        binned2 = sub.groupby("bin_pct", observed=True)["delta_f1"].agg(["mean", "sem", "count"])
        binned2["mid"] = [interval.mid for interval in binned2.index]
        binned2 = binned2.sort_values("mid")

        x2 = np.arange(len(binned2))
        labels2 = [f"{interval.left:.1f}\u2013{interval.right:.1f}%" for interval in binned2.index]
        ax2.bar(x2, binned2["mean"], yerr=binned2["sem"], width=0.6,
                color=color, alpha=0.7, edgecolor=color, linewidth=0.8,
                capsize=3, error_kw={"linewidth": 1})
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(labels2, rotation=30, ha="right", fontsize=8)
        ax2.set_xlabel("Class Frequency (% of test set)")
        if col == 0:
            ax2.set_ylabel("Mean Δ F1 (D4 − No TTA)")
        for i, n in enumerate(binned2["count"]):
            ax2.text(i, binned2["mean"].iloc[i] + binned2["sem"].iloc[i] + 0.001,
                     f"n={int(n)}", ha="center", va="bottom", fontsize=6, alpha=0.6)

    fig.suptitle("TTA Benefit by Class Difficulty and Frequency", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "plot8_per_class_delta.png", **SAVEKW)
    plt.close(fig)
    print("  Saved plot8_per_class_delta.png")


# ---------------------------------------------------------------------------
# Plot 9 — Epistemic uncertainty predicts TTA benefit
# ---------------------------------------------------------------------------

def plot9_uncertainty(df: pd.DataFrame, out_dir: Path):
    """Scatter: epistemic uncertainty vs TTA delta (mean across seeds per config)."""
    deltas = _compute_deltas(df)

    group = CONFIG_COLS + ["model_type"]
    agg = deltas.groupby(group, observed=True).agg(
        epi_unc_mean=("epistemic_unc", "mean"),
        delta_pct_mean=("delta_bacc", lambda x: x.mean() * 100),
        delta_pct_sem=("delta_bacc", lambda x: x.sem() * 100 if len(x) > 1 else 0),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))

    for ds, color in DS_COLORS.items():
        for mode, marker in MODE_MARKERS.items():
            sub = agg[(agg["dataset"] == ds) & (agg["backbone_mode"] == mode)]
            if sub.empty:
                continue
            ax.errorbar(sub["epi_unc_mean"], sub["delta_pct_mean"],
                        yerr=sub["delta_pct_sem"],
                        fmt=marker, markersize=7, alpha=0.8, color=color,
                        markerfacecolor=color, markeredgecolor=color,
                        capsize=2, elinewidth=1, zorder=3)
            for _, row in sub.iterrows():
                ax.annotate(MODEL_SHORT.get(row["model"], row["model"]),
                            (row["epi_unc_mean"], row["delta_pct_mean"]),
                            fontsize=5, alpha=0.6, ha="left",
                            xytext=(5, 0), textcoords="offset points")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
    ax.set_xlabel("Epistemic Uncertainty (D4 strategy, nats)")
    ax.set_ylabel("TTA Δ Balanced Accuracy (%)")
    ax.set_title("View Disagreement Correlates with TTA Benefit", fontsize=13)
    _ds_mode_legend(ax, agg)

    fig.tight_layout()
    fig.savefig(out_dir / "plot9_uncertainty.png", **SAVEKW)
    plt.close(fig)
    print("  Saved plot9_uncertainty.png")


# ---------------------------------------------------------------------------
# Plot 10 — Agreement rate vs TTA delta
# ---------------------------------------------------------------------------

def plot10_agreement_rate(df: pd.DataFrame, out_dir: Path):
    """Scatter: agreement rate vs TTA delta (mean across seeds per config)."""
    deltas = _compute_deltas(df)

    group = CONFIG_COLS + ["model_type"]
    agg = deltas.groupby(group, observed=True).agg(
        agree_mean=("agreement_rate", "mean"),
        delta_pct_mean=("delta_bacc", lambda x: x.mean() * 100),
        delta_pct_sem=("delta_bacc", lambda x: x.sem() * 100 if len(x) > 1 else 0),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))

    for ds, color in DS_COLORS.items():
        for mode, marker in MODE_MARKERS.items():
            sub = agg[(agg["dataset"] == ds) & (agg["backbone_mode"] == mode)]
            if sub.empty:
                continue
            ax.errorbar(sub["agree_mean"], sub["delta_pct_mean"],
                        yerr=sub["delta_pct_sem"],
                        fmt=marker, markersize=7, alpha=0.8, color=color,
                        markerfacecolor=color, markeredgecolor=color,
                        capsize=2, elinewidth=1, zorder=3)
            for _, row in sub.iterrows():
                ax.annotate(MODEL_SHORT.get(row["model"], row["model"]),
                            (row["agree_mean"], row["delta_pct_mean"]),
                            fontsize=5, alpha=0.6, ha="left",
                            xytext=(5, 0), textcoords="offset points")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
    ax.set_xlabel("Agreement Rate Among TTA Views (D4)")
    ax.set_ylabel("TTA Δ Balanced Accuracy (%)")
    ax.set_title("Low Agreement → Bigger TTA Gains", fontsize=13)
    _ds_mode_legend(ax, agg)

    fig.tight_layout()
    fig.savefig(out_dir / "plot10_agreement_rate.png", **SAVEKW)
    plt.close(fig)
    print("  Saved plot10_agreement_rate.png")


# ---------------------------------------------------------------------------
# Plot 11 — Does model scale reduce TTA benefit?
# ---------------------------------------------------------------------------

def plot11_scale_vs_delta(df: pd.DataFrame, out_dir: Path):
    """Two-panel scatter: (A) parameter count vs TTA delta, (B) pretrain data vs TTA delta.

    Each panel shows histology FMs on TCGA-UT (frozen backbone) with Pearson r.
    Log-scale x-axis; points labeled with model short names; error bars = SEM across seeds.
    """
    # Filter to histology FMs on TCGA-UT, canonical frozen config
    histo = df[(df["model_type"] == "histology") & (df["dataset"] == "tcga-ut")]
    primary = _primary_config(histo)
    deltas = _compute_deltas(primary)

    # Keep only models with metadata
    deltas = deltas[deltas["model"].isin(FM_META)]

    group = CONFIG_COLS + ["model_type"]
    agg = deltas.groupby(group, observed=True).agg(
        delta_pct_mean=("delta_bacc", lambda x: x.mean() * 100),
        delta_pct_sem=("delta_bacc", lambda x: x.sem() * 100 if len(x) > 1 else 0),
    ).reset_index()

    agg["params_m"] = agg["model"].map(lambda m: FM_META[m]["params_m"])
    agg["pretrain_tiles_m"] = agg["model"].map(lambda m: FM_META[m]["pretrain_tiles_m"])

    color = DS_COLORS["tcga-ut"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    panels = [
        (ax1, "params_m", "Parameter Count (M)", "(A) Parameter Count"),
        (ax2, "pretrain_tiles_m", "Pre-training Tiles (M)", "(B) Pre-training Data"),
    ]

    for ax, xcol, xlabel, title in panels:
        xvals = agg[xcol].values
        yvals = agg["delta_pct_mean"].values
        yerrs = agg["delta_pct_sem"].values

        ax.errorbar(xvals, yvals, yerr=yerrs,
                    fmt="o", markersize=9, alpha=0.8, color=color,
                    markerfacecolor=color, markeredgecolor=color,
                    capsize=3, elinewidth=1, zorder=3)

        # Label each point
        for _, row in agg.iterrows():
            ax.annotate(MODEL_SHORT.get(row["model"], row["model"]),
                        (row[xcol], row["delta_pct_mean"]),
                        fontsize=8, alpha=0.8, ha="left",
                        xytext=(7, 3), textcoords="offset points")

        # Pearson r on log-transformed x
        mask = ~np.isnan(yvals)
        if mask.sum() >= 3:
            r, p = stats.pearsonr(np.log10(xvals[mask]), yvals[mask])
            p_str = f"p={p:.3f}" if p >= 0.001 else f"p={p:.1e}"
            ax.text(0.05, 0.95, f"r={r:.2f}, {p_str}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

        ax.set_xscale("log")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=11)

    ax1.set_ylabel("TTA Δ Balanced Accuracy (%)")

    fig.suptitle("Does Model Scale Reduce TTA Benefit?\n(TCGA-UT, frozen backbone)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "plot11_scale_vs_delta.png", **SAVEKW)
    plt.close(fig)
    print("  Saved plot11_scale_vs_delta.png")


# ---------------------------------------------------------------------------
# Plot 12 — Dataset size ablation: TTA benefit vs training set size
# ---------------------------------------------------------------------------

SUBSET_ORDER = [1000, 5000, 10000, 50000, -1]
SUBSET_LABELS = {1000: "1K", 5000: "5K", 10000: "10K", 50000: "50K", -1: "Full"}

def plot12_datasize_ablation(abl: pd.DataFrame, out_dir: Path):
    """Two-panel plot: (A) absolute balanced acc with/without TTA, (B) TTA delta vs training size.

    TCGA-UT only. Shows No TTA vs TTA (D4/mean), averaged across seeds with SEM error bars.
    """
    abl_mean = abl[(abl["aggregation"] == "mean") & (abl["dataset"] == "tcga-ut")].copy()

    present = abl_mean["train_subset"].unique()
    subset_vals = [v for v in SUBSET_ORDER if v in present]
    xlabels = [SUBSET_LABELS.get(v, str(int(v))) for v in subset_vals]
    x = np.arange(len(subset_vals))

    color_base = "#999999"
    color_tta = "#348ABD"

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel A: absolute balanced accuracy ---
    for strat, color, marker, label in [
        ("none", color_base, "o", "No TTA"),
        ("d4", color_tta, "D", "TTA"),
    ]:
        ss = abl_mean[abl_mean["strategy"] == strat]
        grouped = ss.groupby("train_subset").agg(
            bacc_mean=("balanced_acc", "mean"),
            bacc_sem=("balanced_acc", lambda x: x.sem() if len(x) > 1 else 0),
        ).reindex(subset_vals)

        ax0.errorbar(x, grouped["bacc_mean"] * 100, yerr=grouped["bacc_sem"] * 100,
                     fmt=f"-{marker}", color=color, markerfacecolor=color,
                     markersize=7, capsize=4, elinewidth=1.2, linewidth=1.5,
                     label=label, alpha=0.85)

    ax0.set_xticks(x)
    ax0.set_xticklabels(xlabels)
    ax0.set_xlabel("Training Set Size")
    ax0.set_ylabel("Balanced Accuracy (%)")
    ax0.set_title("(A) Accuracy vs Training Data", fontsize=11)
    ax0.legend(fontsize=9)

    # --- Panel B: TTA delta ---
    base = abl_mean[abl_mean["strategy"] == "none"]
    tta = abl_mean[abl_mean["strategy"] == "d4"]
    merge_on = ["train_subset", "seed"]
    merged = base[merge_on + ["balanced_acc"]].merge(
        tta[merge_on + ["balanced_acc"]],
        on=merge_on, suffixes=("_base", "_tta"),
    )
    merged["delta"] = (merged["balanced_acc_tta"] - merged["balanced_acc_base"]) * 100

    grouped_d = merged.groupby("train_subset").agg(
        delta_mean=("delta", "mean"),
        delta_sem=("delta", lambda x: x.sem() if len(x) > 1 else 0),
    ).reindex(subset_vals)

    ax1.errorbar(x, grouped_d["delta_mean"], yerr=grouped_d["delta_sem"],
                 fmt="-D", color=color_tta, markerfacecolor=color_tta,
                 markersize=7, capsize=4, elinewidth=1.2, linewidth=1.5, alpha=0.85)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels)
    ax1.set_xlabel("Training Set Size")
    ax1.set_ylabel("TTA Δ Balanced Accuracy (pp)")
    ax1.set_title("(B) TTA Improvement vs Training Data", fontsize=11)

    fig.suptitle("Dataset Size Ablation (Phikon, TCGA-UT, frozen backbone)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "plot12_datasize_ablation.png", **SAVEKW)
    plt.close(fig)
    print("  Saved plot12_datasize_ablation.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PLOT_REGISTRY = {
    "1":   ("plot1_tta_helps_all",      "df"),
    "1b":  ("plot1b_general_models",    "df"),
    "2":   ("plot2_histology_fm_delta", "df"),
    "3":   ("plot3_frozen_vs_finetuned","df"),
    "5":   ("plot5_strategy_progression","df"),
    "6":   ("plot6_correction_scatter", "df"),
    "7":   ("plot7_baseline_vs_delta",  "df"),
    "8":   ("plot8_per_class_delta",    "pc"),
    "9":   ("plot9_uncertainty",        "df"),
    "10":  ("plot10_agreement_rate",    "df"),
    "11":  ("plot11_scale_vs_delta",    "df"),
    "12":  ("plot12_datasize_ablation", "abl"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Publication-quality TTA analysis plots")
    p.add_argument("--out_dir", default="figures", help="Output directory (default: figures/)")
    p.add_argument("--results_csv", default="results/tta_results.csv")
    p.add_argument("--per_class_csv", default="results/tta_per_class.csv")
    p.add_argument("--plot", nargs="*", default=None,
                   help=f"Plot(s) to generate (default: all). Choices: {list(PLOT_REGISTRY.keys())}")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, pc = load_data(args.results_csv, args.per_class_csv)
    n_seeds = df["seed"].nunique()
    print(f"  {len(df)} summary rows, {len(pc)} per-class rows, {n_seeds} seeds")

    selected = args.plot if args.plot else list(PLOT_REGISTRY.keys())

    # Lazy-load ablation data only when needed
    abl = None
    if any(PLOT_REGISTRY.get(k, (None, None))[1] == "abl" for k in selected):
        abl = load_ablation_data(args.results_csv)
        print(f"  {len(abl)} ablation rows loaded")

    print("Generating plots...")
    for key in selected:
        if key not in PLOT_REGISTRY:
            print(f"  WARNING: unknown plot '{key}', skipping. Choices: {list(PLOT_REGISTRY.keys())}")
            continue
        func_name, data_key = PLOT_REGISTRY[key]
        func = globals()[func_name]
        data = {"df": df, "pc": pc, "abl": abl}[data_key]
        func(data, out_dir)

    print(f"\nDone. All plots saved to '{out_dir}/'")


if __name__ == "__main__":
    main()
