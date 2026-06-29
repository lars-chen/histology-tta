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

sns.set_theme(style="whitegrid", font_scale=1.4)
matplotlib.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

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
DATASET_NCLASSES = {"tcga-ut": 31, "nct-crc-100k": 9, "nct-crc-nonorm": 9, "mhist": 2}

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
    "resnet18": "R18", "resnet50": "R50",
    "convnextv2_tiny": "CNT", "convnextv2_base": "CNB", "dinov2_s": "D2S",
    "dinov2_b": "D2B", "d4wrn": "D4W",
}

FM_META = {
    # month encoded as year + (month-1)/12 for fractional year plotting
    "ctranspath": {"params_m": 28,   "pretrain_tiles_m": 15,   "year_frac": 2021 + 11/12},
    "phikon":     {"params_m": 86,   "pretrain_tiles_m": 40,   "year_frac": 2023 +  6/12},
    "virchow":    {"params_m": 632,  "pretrain_tiles_m": 1500, "year_frac": 2023 +  8/12},
    "uni":        {"params_m": 300,  "pretrain_tiles_m": 100,  "year_frac": 2024 +  2/12},
    "gigapath":   {"params_m": 1100, "pretrain_tiles_m": 1300, "year_frac": 2024 +  4/12},
    "hoptimus":   {"params_m": 1100, "pretrain_tiles_m": 1000, "year_frac": 2025 +  2/12},
    "virchow2":   {"params_m": 632,  "pretrain_tiles_m": 3100, "year_frac": 2024 +  7/12},
    "phikon2":    {"params_m": 300,  "pretrain_tiles_m": 456,  "year_frac": 2024 +  8/12},
    "uni2":       {"params_m": 681,  "pretrain_tiles_m": 200,  "year_frac": 2025 +  0/12},
}

DS_COLORS = {"tcga-ut": "#6B9EC7", "nct-crc-100k": "#8B5E3C", "nct-crc-nonorm": "#56B4E9", "mhist": "#CC79A7"}
MODE_MARKERS = {"frozen": "o", "finetuned": "s"}

SAVEKW = dict(dpi=300, bbox_inches="tight", facecolor="white")

# Publication-quality rcParams — apply with `with plt.rc_context(PUB_RCPARAMS):`
PUB_RCPARAMS = {
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# Colorblind-safe palette (Wong 2011)
CB_PALETTE = {
    "histology":   "#0072B2",  # blue
    "general":     "#D4772A",  # burnt orange
    "equivariant": "#009E73",  # green
}

# Full model display names (for figure labels)
MODEL_FULL_NAMES = {
    "gigapath":       "GigaPath",
    "hoptimus":       "H-optimus",
    "phikon":         "Phikon",
    "phikon2":        "Phikon-2",
    "uni":            "UNI",
    "uni2":           "UNI-2",
    "virchow":        "Virchow",
    "virchow2":       "Virchow2",
    "resnet18":        "ResNet-18",
    "resnet50":        "ResNet-50",
    "convnextv2_tiny": "ConvNeXtV2-T",
    "convnextv2_base": "ConvNeXtV2-B",
    "dinov2_s":       "DINOv2-S",
    "dinov2_b":       "DINOv2-B",
    "d4wrn":          "D4-WRN",
}

# Model → type mapping (for datasets loaded without model_type column)
MODEL_TYPE_MAP = {
    "gigapath": "histology", "hoptimus": "histology",
    "phikon": "histology", "phikon2": "histology",
    "uni": "histology", "uni2": "histology",
    "virchow": "histology", "virchow2": "histology",
    "resnet18": "general", "resnet50": "general",
    "convnextv2_tiny": "general", "convnextv2_base": "general",
    "dinov2_s": "general", "dinov2_b": "general", "dinov2_base": "general",
    "d4wrn": "equivariant",
}

# Columns that identify a unique experiment (excluding seed)
CONFIG_COLS = ["model", "dataset", "backbone_mode", "train_augment"]


# ---------------------------------------------------------------------------
# Data loading & helpers
# ---------------------------------------------------------------------------

def load_data(results_csv: str = "results/tta_results.csv",
              per_class_csv: str = "results/tta_per_class.csv",
              mlp_hidden: str | None = None):
    """
    Load results. mlp_hidden=None (default) → linear head only.
    Pass a float (e.g. 768.0) to load a specific MLP head size instead.
    """
    df = pd.read_csv(results_csv)
    pc = pd.read_csv(per_class_csv)
    df = df[(df["train_augment"] == True) & (df["train_subset"].isna())].copy()
    if mlp_hidden is None:
        df = df[df["mlp_hidden"] == "linear"]
    else:
        df = df[df["mlp_hidden"] == mlp_hidden]
    pc = pc[(pc["train_augment"] == True)].copy()
    return df, pc


def load_canonical(canonical_csv: str = "results/canonical_results.csv",
                   head_type: str = "linear"):
    """Load the unified canonical results (new frozen-probe pipeline) in a schema
    compatible with the figure helpers (_compute_deltas / panels).

    canonical_results.csv holds frozen probe rows (all head types) for
    histology/general models plus finetuned linear rows (e.g. the d4wrn
    equivariant control). We keep one head type (default 'linear', the canonical
    probe) and synthesize the columns the figure code expects but the new schema
    doesn't carry (train_augment is always True here; the optional uncertainty
    columns are filled with NaN — the figure panels don't consume them).
    """
    df = pd.read_csv(canonical_csv)
    df = df[df["head_type"] == head_type].copy()
    df["train_augment"] = True
    for opt in ("epistemic_unc", "agreement_rate"):
        if opt not in df.columns:
            df[opt] = np.nan
    return df


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
    # Optional columns only present in the finetuned-pipeline schema.
    optional = [c for c in ("epistemic_unc", "agreement_rate") if c in best.columns]
    merged = base[merge_cols + ["balanced_acc", "acc", "model_type"]].merge(
        best[merge_cols + ["balanced_acc", "acc", "strategy",
                           "n_corrected", "n_corrupted"] + optional],
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
    ctranspath is excluded from mhist (skill filter is unreliable on that dataset).
    """
    df = df[~((df["model"] == "ctranspath") & (df["dataset"] == "mhist"))]
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
    ax.legend(handles=handles, fontsize=11)


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
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=11)
        ax.set_ylabel("Balanced Accuracy" if col == 0 else "")
        ax.legend(fontsize=10, loc="lower left")

        all_vals = [v for v in base_means + tta_means if not np.isnan(v)]
        all_errs = base_errs + tta_errs
        if all_vals:
            lo = min(all_vals) - max(all_errs) - 0.03
            hi = max(all_vals) + max(all_errs) + 0.04
            ax.set_ylim(max(0, lo), min(1, hi))

    axes[0].set_ylabel(f"{row_label}\nBalanced Accuracy", fontsize=13)


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
        ax.set_title(_ds_label(ds), fontsize=14)

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
        axes[0, col].set_title(_ds_label(ds), fontsize=14)

    base_ft = _get_baseline(finetuned)
    best_ft = _get_tta(finetuned, strategy="d4", agg="mean")
    _plot1_row(axes[0], base_ft, best_ft, datasets, general_models, "Finetuned")

    base_fr = _get_baseline(frozen)
    best_fr = _get_tta(frozen, strategy="d4", agg="mean")
    _plot1_row(axes[1], base_fr, best_fr, datasets, general_models, "Frozen")

    fig.tight_layout()
    fig.savefig(out_dir / "plot1b_general_models.png", **SAVEKW)
    plt.close(fig)
    print("  Saved plot1b_general_models.png")


# ---------------------------------------------------------------------------
# Plot 2 — Histology FMs leave performance on the table
# ---------------------------------------------------------------------------

def _draw_panel_a(ax, df, xtick_fontsize=12):
    """Draw panel (a): per-dataset boxplot of TTA Δ balanced accuracy by model type.

    Returns legend handles (caller decides placement).
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    deltas = _compute_deltas(df)
    group = CONFIG_COLS + ["model_type"]
    agg = deltas.groupby(group, observed=True).agg(
        delta_pct_mean=("delta_bacc", lambda x: x.mean() * 100),
    ).reset_index()

    datasets_present = [ds for ds in DATASET_ORDER if ds in agg["dataset"].values]
    box_types = [mt for mt in ["histology", "general"] if mt in agg["model_type"].values]
    has_equivariant = "equivariant" in agg["model_type"].values

    box_width = 0.32
    offsets = {mt: (i - (len(box_types) - 1) / 2) * (box_width + 0.10)
               for i, mt in enumerate(box_types)}

    rng = np.random.default_rng(42)
    ann_tops = []

    for mt in box_types:
        color = CB_PALETTE[mt]
        positions, box_data = [], []
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
                        medianprops=dict(color="black", linewidth=2.0),
                        whiskerprops=dict(color=color, alpha=0.7, linewidth=1.2),
                        capprops=dict(color=color, alpha=0.7, linewidth=1.2))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.5)

        for pos, vals in zip(positions, box_data):
            jitter = rng.uniform(-box_width * 0.28, box_width * 0.28, size=len(vals))
            ax.scatter(pos + jitter, vals, color=color, s=30, alpha=0.85,
                       edgecolors="white", linewidths=0.5, zorder=3)
            ann_y = min(
                max(np.max(vals),
                    np.percentile(vals, 75) + 1.5 * (np.percentile(vals, 75) - np.percentile(vals, 25))
                    ) + 0.08,
                2.8,
            )
            ax.text(pos, ann_y, f"{np.mean(vals):+.2f}",
                    ha="center", va="bottom", fontsize=8.5, color=color, fontweight="bold")
            ann_tops.append(ann_y)

    # Equivariant control: diamond per dataset
    if has_equivariant:
        eq_color = CB_PALETTE["equivariant"]
        eq = agg[agg["model_type"] == "equivariant"]
        for di, ds in enumerate(datasets_present):
            vals = eq[eq["dataset"] == ds]["delta_pct_mean"].values
            for v in vals:
                ax.scatter(di, v, color=eq_color, s=60, marker="o", alpha=0.95,
                           edgecolors="white", linewidths=0.8, zorder=5)

    # Zero reference line: light gray dashed
    ax.axhline(0, color="#888888", linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)

    _DS_SHORT = {
        "tcga-ut": "TCGA-UT",
        "nct-crc-100k": "NCT-CRC-100K",
        "nct-crc-nonorm": "NCT-CRC-NoNorm",
        "mhist": "MHIST",
    }
    ax.set_xticks(list(range(len(datasets_present))))
    ax.set_xticklabels([_DS_SHORT.get(ds, ds) for ds in datasets_present], fontsize=xtick_fontsize)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", which="both", length=0)
    ax.set_ylabel("TTA Δ Bal. Accuracy (pp)", fontsize=9)

    all_agg_vals = agg["delta_pct_mean"].values
    ylo = np.nanmin(all_agg_vals) - 0.15
    yhi = 3.6
    ax.set_ylim(ylo, yhi)

    handles = [Patch(facecolor=CB_PALETTE[mt], alpha=0.4,
                     edgecolor=CB_PALETTE[mt], label=mt.title())
               for mt in box_types]
    if has_equivariant:
        handles.append(Patch(facecolor=CB_PALETTE["equivariant"], alpha=0.4,
                             edgecolor=CB_PALETTE["equivariant"], label="Equivariant"))
    for _gl in ax.get_xgridlines() + ax.get_ygridlines():
        _gl.set_alpha(0.15)
    return handles


def plot2_histology_fm_delta(df: pd.DataFrame, out_dir: Path):
    """Panel (a): per-dataset boxplot of TTA Δ balanced accuracy by model type.

    Colorblind-safe palette. Mean ± SD annotations, n per box, legend outside right.
    """
    with plt.rc_context(PUB_RCPARAMS):
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        handles = _draw_panel_a(ax, df)
        ax.legend(handles=handles, loc="upper left", fontsize=10,
                  title="Model Type", title_fontsize=10,
                  bbox_to_anchor=(1.01, 1.0), borderaxespad=0, framealpha=0.95)
        fig.tight_layout()
        stem = out_dir / "plot2_histology_fm_delta"
        fig.savefig(str(stem) + ".png", **SAVEKW)
        fig.savefig(str(stem) + ".pdf", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print("  Saved plot2_histology_fm_delta.png/.pdf")


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
                    fontsize=10, xytext=(-10, 5), textcoords="offset points", ha="right")

    # Draw connecting lines between frozen/finetuned pairs
    for ds in agg["dataset"].unique():
        frozen = agg[(agg["dataset"] == ds) & (agg["backbone_mode"] == "frozen")]
        finetuned = agg[(agg["dataset"] == ds) & (agg["backbone_mode"] == "finetuned")]
        if not frozen.empty and not finetuned.empty:
            ax.plot([0, 1],
                    [frozen["delta_pct_mean"].values[0], finetuned["delta_pct_mean"].values[0]],
                    "-", color="gray", alpha=0.4, linewidth=1)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Frozen (linear probe)", "Finetuned"], fontsize=14)
    ax.set_ylabel("TTA Δ Balanced Accuracy (%)")
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

    ax1.set_xlabel("Corruption Rate (%)")
    ax1.set_ylabel("Correction Rate (%)")
    ax1.set_title("(A) TTA Fixes More Than It Breaks", fontsize=14)
    ax1.legend(handles=legend_handles, fontsize=10, loc="upper right")

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
    ax2.set_xticklabels(models, rotation=45, ha="right", fontsize=11)
    ax2.set_ylabel("Net Correction Ratio (corrected / corrupted)")
    ax2.set_title("(B) Net Correction Ratio by Model", fontsize=14)
    ax2.legend(fontsize=10, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "plot6_correction_scatter.png", **SAVEKW)
    plt.close(fig)
    print("  Saved plot6_correction_scatter.png")


# ---------------------------------------------------------------------------
# Plot 7 — Weaker baselines gain more from TTA
# ---------------------------------------------------------------------------

def _draw_panel_b(axes, df, show_protocol: bool = True):
    """Draw panel (b): 1×N faceted scatter of baseline acc vs TTA Δ, one subplot per dataset.

    Color = model family (CB_PALETTE). Shape = training protocol (frozen ○, finetuned □)
    when show_protocol=True; single marker when False.
    Returns legend handles (caller places the legend).
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    deltas = _compute_deltas(df)
    group = CONFIG_COLS + ["model_type"]
    agg = deltas.groupby(group, observed=True).agg(
        bacc_base_mean=("balanced_acc_base", "mean"),
        delta_pct_mean=("delta_bacc", lambda x: x.mean() * 100),
        delta_pct_sem=("delta_bacc", lambda x: x.sem() * 100 if len(x) > 1 else 0),
    ).reset_index()

    datasets_present = [ds for ds in DATASET_ORDER if ds in agg["dataset"].values]
    _DS_SHORT = {
        "tcga-ut": "TCGA-UT",
        "nct-crc-100k": "NCT-CRC-100K",
        "nct-crc-nonorm": "NCT-CRC-NoNorm",
        "mhist": "MHIST",
    }

    # Shared y-axis range across all facets (includes ± SEM)
    all_lo = (agg["delta_pct_mean"] - agg["delta_pct_sem"]).values
    all_hi = (agg["delta_pct_mean"] + agg["delta_pct_sem"]).values
    ylo = np.nanmin(all_lo) - 0.12
    yhi = np.nanmax(all_hi) + 0.18

    for ax, ds in zip(axes, datasets_present):
        sub = agg[agg["dataset"] == ds]
        ax.axhline(0, color="#888888", linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)

        for mtype in ["histology", "general", "equivariant"]:
            color = CB_PALETTE[mtype]
            marker = "o"
            msize = 6.5
            if show_protocol:
                for mode, pmarker in [("frozen", marker), ("finetuned", "s")]:
                    if mtype == "equivariant":
                        pmarker = marker
                    pts = sub[(sub["model_type"] == mtype) & (sub["backbone_mode"] == mode)]
                    if pts.empty:
                        continue
                    ax.errorbar(
                        pts["bacc_base_mean"], pts["delta_pct_mean"],
                        yerr=pts["delta_pct_sem"],
                        fmt=pmarker, markersize=msize, alpha=0.85,
                        color=color, markerfacecolor=color,
                        markeredgecolor="white", markeredgewidth=0.7,
                        capsize=2, elinewidth=1, zorder=3,
                    )
            else:
                pts = sub[sub["model_type"] == mtype]
                if pts.empty:
                    continue
                ax.errorbar(
                    pts["bacc_base_mean"], pts["delta_pct_mean"],
                    yerr=pts["delta_pct_sem"],
                    fmt=marker, markersize=msize, alpha=0.85,
                    color=color, markerfacecolor=color,
                    markeredgecolor="white", markeredgewidth=0.7,
                    capsize=2, elinewidth=1, zorder=3,
                )

        # Regression line + std-error fill + Spearman ρ (exclude equivariant/d4wrn)
        reg_sub = sub[sub["model_type"] != "equivariant"]
        x_all = reg_sub["bacc_base_mean"].values
        y_all = reg_sub["delta_pct_mean"].values
        if len(x_all) >= 2:
            from scipy.stats import spearmanr
            coef = np.polyfit(x_all, y_all, 1)
            x_line = np.linspace(x_all.min(), x_all.max(), 200)
            y_line = np.polyval(coef, x_line)
            residuals = y_all - np.polyval(coef, x_all)
            std_res = residuals.std()
            ax.plot(x_line, y_line, color="#444444", linewidth=1.2, zorder=2)
            ax.fill_between(x_line, y_line - std_res, y_line + std_res,
                            color="#444444", alpha=0.12, zorder=1)
            rho, _ = spearmanr(x_all, y_all)
            ax.text(0.97, 0.97, f"ρ = {rho:.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8)

        ax.set_title("", pad=3)
        ax.set_xlabel("Baseline Bal. Acc.", fontsize=9)
        ax.set_ylim(ylo, yhi)
        x_vals = sub["bacc_base_mean"].values
        pad = (x_vals.max() - x_vals.min()) * 0.08
        ax.set_xticks(np.linspace(x_vals.min() + pad, x_vals.max() - pad, 3))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))

    axes[0].set_ylabel("TTA Δ Bal. Accuracy (pp)", fontsize=9)
    axes[0].set_yticks([0, 1, 2, 3])
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    for ax in axes:
        for _gl in ax.get_xgridlines() + ax.get_ygridlines():
            _gl.set_alpha(0.15)

    type_handles = [
        Patch(facecolor=CB_PALETTE["histology"], edgecolor=CB_PALETTE["histology"],
              alpha=0.85, label="Histology FM"),
        Patch(facecolor=CB_PALETTE["general"], edgecolor=CB_PALETTE["general"],
              alpha=0.85, label="General"),
        Patch(facecolor=CB_PALETTE["equivariant"], edgecolor=CB_PALETTE["equivariant"],
              alpha=0.85, label="Equivariant"),
    ]
    if show_protocol:
        proto_handles = [
            Line2D([0], [0], marker="o", color="gray", markerfacecolor="gray",
                   markersize=7, linestyle="None", label="Frozen"),
            Line2D([0], [0], marker="s", color="gray", markerfacecolor="gray",
                   markersize=7, linestyle="None", label="Finetuned"),
        ]
        return type_handles + proto_handles
    return type_handles


def plot7_baseline_vs_delta(df: pd.DataFrame, out_dir: Path):
    """Panel (b): 1×4 faceted scatter of baseline balanced acc vs TTA Δ.

    Color = model family; shape = training protocol. Shared y-axis across facets.
    """
    datasets_present = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    n_ds = len(datasets_present)

    with plt.rc_context(PUB_RCPARAMS):
        fig, axes = plt.subplots(1, n_ds, figsize=(7, 3.6), sharey=True)
        if n_ds == 1:
            axes = [axes]
        legend_handles = _draw_panel_b(list(axes), df)
        fig.subplots_adjust(bottom=0.26)
        fig.legend(handles=legend_handles, loc="lower center",
                   bbox_to_anchor=(0.5, 0.01), ncol=len(legend_handles),
                   fontsize=9, frameon=True, framealpha=0.95)
        stem = out_dir / "plot7_baseline_vs_delta"
        fig.savefig(str(stem) + ".png", **SAVEKW)
        fig.savefig(str(stem) + ".pdf", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print("  Saved plot7_baseline_vs_delta.png/.pdf")


# ---------------------------------------------------------------------------
# Plot 8 — TTA disproportionately helps hard classes
# ---------------------------------------------------------------------------

def plot8_per_class_delta(pc: pd.DataFrame, out_dir: Path):
    """Per-class TTA Δ F1 distribution by baseline difficulty quartile.

    Each panel = one dataset. Violin + strip shows the full distribution of
    per-class Δ F1 (including negatives). A secondary y-axis overlays the
    fraction of classes hurt (Δ F1 < 0) per quartile as a line.
    """
    pc_filt = pc[(pc["model_type"] == "histology") &
                 (pc["backbone_mode"] == "frozen")].copy()

    base_pc = pc_filt[(pc_filt["strategy"] == "none") & (pc_filt["aggregation"] == "mean")].copy()
    tta_pc  = pc_filt[(pc_filt["strategy"] == "d4")   & (pc_filt["aggregation"] == "mean")].copy()

    merge_cols = ["model", "dataset", "backbone_mode", "train_augment", "seed", "class_name"]
    merged = base_pc[merge_cols + ["f1", "support"]].merge(
        tta_pc[merge_cols + ["f1"]],
        on=merge_cols, suffixes=("_base", "_tta"),
    )
    merged["delta_f1"] = merged["f1_tta"] - merged["f1_base"]
    merged = merged.dropna(subset=["f1_base", "delta_f1"])

    datasets = [d for d in DATASET_ORDER
                if d in merged["dataset"].unique() and d != "mhist"]

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 6))
    if n_ds == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = merged[merged["dataset"] == ds].copy()
        if sub.empty:
            continue

        color = DS_COLORS.get(ds, "gray")

        lo, hi = sub["delta_f1"].quantile([0.02, 0.98])
        sub = sub[(sub["delta_f1"] >= lo) & (sub["delta_f1"] <= hi)].copy()

        sub["bin_f1"] = pd.qcut(sub["f1_base"], q=4, duplicates="drop")
        # Sort bins by midpoint
        bin_order = sorted(sub["bin_f1"].unique(), key=lambda iv: iv.mid)
        sub["bin_label"] = sub["bin_f1"].map(
            {iv: f"Q{i+1}\n{iv.left:.2f}–{iv.right:.2f}" for i, iv in enumerate(bin_order)}
        )
        label_order = [f"Q{i+1}\n{iv.left:.2f}–{iv.right:.2f}"
                       for i, iv in enumerate(bin_order)]

        sns.violinplot(data=sub, x="bin_label", y="delta_f1", order=label_order,
                       color=color, alpha=0.55, inner=None, linewidth=0.8, ax=ax)
        sns.stripplot(data=sub, x="bin_label", y="delta_f1", order=label_order,
                      color=color, alpha=0.25, size=2.5, jitter=True, ax=ax)

        # Mean marker per bin
        means = sub.groupby("bin_label", observed=True)["delta_f1"].mean()
        for i, lbl in enumerate(label_order):
            if lbl in means:
                ax.scatter(i, means[lbl], color="white", edgecolors="black",
                           s=40, zorder=5, linewidths=1.2)

        ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
        ax.set_xlabel("Baseline F1 Quartile")
        ax.set_ylabel("Δ F1 (D4 TTA − No TTA)" if ax is axes[0] else "")
        ax.set_title(_ds_label(ds), fontsize=14)

        # Secondary axis: % of classes hurt per bin
        ax2 = ax.twinx()
        pct_hurt = sub.groupby("bin_label", observed=True)["delta_f1"].apply(
            lambda x: (x < 0).mean() * 100
        ).reindex(label_order)
        ax2.plot(range(len(label_order)), pct_hurt.values,
                 color="crimson", linewidth=1.5, marker="o", markersize=5,
                 linestyle="-", alpha=0.8, zorder=6)
        ax2.set_ylabel("% classes hurt by TTA", color="crimson", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="crimson", labelsize=7)
        ax2.set_ylim(0, 100)

        # Counts
        counts = sub.groupby("bin_label", observed=True)["delta_f1"].count().reindex(label_order)
        for i, (lbl, n) in enumerate(counts.items()):
            ax.text(i, ax.get_ylim()[0], f"n={int(n)}", ha="center", va="bottom",
                    fontsize=9, alpha=0.6)

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

    # Normalize epistemic uncertainty by log(num_classes) so cross-dataset
    # comparison is valid (raw entropy is bounded by log(C)).
    agg["epi_unc_norm"] = agg.apply(
        lambda r: r["epi_unc_mean"] / np.log(DATASET_NCLASSES[r["dataset"]]),
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    for ds, color in DS_COLORS.items():
        for mode, marker in MODE_MARKERS.items():
            sub = agg[(agg["dataset"] == ds) & (agg["backbone_mode"] == mode)]
            if sub.empty:
                continue
            ax.errorbar(sub["epi_unc_norm"], sub["delta_pct_mean"],
                        yerr=sub["delta_pct_sem"],
                        fmt=marker, markersize=7, alpha=0.8, color=color,
                        markerfacecolor=color, markeredgecolor=color,
                        capsize=2, elinewidth=1, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
    ax.set_xlabel("Normalized Epistemic Uncertainty (D4 strategy)")
    ax.set_ylabel("TTA Δ Balanced Accuracy (%)")
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

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
    ax.set_xlabel("Agreement Rate Among TTA Views (D4)")
    ax.set_ylabel("TTA Δ Balanced Accuracy (%)")
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
                        fontsize=11, alpha=0.8, ha="left",
                        xytext=(7, 3), textcoords="offset points")

        # Pearson r on log-transformed x
        mask = ~np.isnan(yvals)
        if mask.sum() >= 3:
            r, p = stats.pearsonr(np.log10(xvals[mask]), yvals[mask])
            p_str = f"p={p:.3f}" if p >= 0.001 else f"p={p:.1e}"
            ax.text(0.05, 0.95, f"r={r:.2f}, {p_str}",
                    transform=ax.transAxes, fontsize=12, va="top",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

        ax.set_xscale("log")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=14)

    ax1.set_ylabel("TTA Δ Balanced Accuracy (%)")

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
    ax0.set_title("(A) Accuracy vs Training Data", fontsize=14)
    ax0.legend(fontsize=12)

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
    ax1.set_title("(B) TTA Improvement vs Training Data", fontsize=14)

    fig.tight_layout()
    fig.savefig(out_dir / "plot12_datasize_ablation.png", **SAVEKW)
    plt.close(fig)
    print("  Saved plot12_datasize_ablation.png")



# 8-colour palette: one per D4 transform, in canonical order
_D4_TRANSFORM_NAMES = [
    "orig", "rot90", "rot180", "rot270",
    "hflip", "hflip+rot90", "hflip+rot180", "hflip+rot270",
]
_D4_COLORS = [
    "#E24A33", "#348ABD", "#988ED5", "#8EBA42",
    "#FBC15E", "#FFB5C8", "#777777", "#56B4E9",
]
_D4_DISPLAY = [
    "0° (orig)", "90°", "180°", "270°",
    "HFlip", "HFlip+90°", "HFlip+180°", "HFlip+270°",
]


# ---------------------------------------------------------------------------
# Plot 16 — Per-class scatter: hard classes benefit most from TTA
# ---------------------------------------------------------------------------

def plot_perclass_scatter(pc: pd.DataFrame, out_dir: Path):
    """Figure 2 (pub): per-class baseline F1 vs TTA delta F1.

    Main panel: TCGA-UT (31 classes), point size ∝ log(support).
    Inset panel: NCT-CRC-100K (9 classes), same axes.
    Averaged over histology FM models × frozen backbone × seeds.
    """
    from matplotlib.lines import Line2D
    import matplotlib.gridspec as gridspec

    def _compute_class_deltas(pc_in, dataset, model_type="histology", mode="frozen"):
        sub = pc_in[
            (pc_in["dataset"] == dataset) &
            (pc_in["model_type"] == model_type) &
            (pc_in["backbone_mode"] == mode) &
            (pc_in["train_augment"] == True)
        ].copy()
        base = sub[sub["strategy"] == "none"][["model", "seed", "class_name", "f1", "support"]].copy()
        tta  = sub[sub["strategy"] == "d4"][["model", "seed", "class_name", "f1"]].copy()
        merged = base.merge(tta, on=["model", "seed", "class_name"], suffixes=("_base", "_tta"))
        merged["delta_f1"] = merged["f1_tta"] - merged["f1_base"]
        # Average across models and seeds
        agg = merged.groupby("class_name", observed=True).agg(
            f1_base=("f1_base", "mean"),
            delta_f1=("delta_f1", "mean"),
            support=("support", "first"),
        ).reset_index()
        return agg

    def _reg_line_ci(x, y, ax, color="gray"):
        """Draw OLS line + 95% CI band."""
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = np.array(x)[mask], np.array(y)[mask]
        if len(x) < 3:
            return
        slope, intercept, r, p, se = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = slope * x_line + intercept
        n = len(x)
        t_val = stats.t.ppf(0.975, n - 2)
        resid_std = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n - 2))
        ci = t_val * resid_std * np.sqrt(1 / n + (x_line - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
        ax.plot(x_line, y_line, color=color, linewidth=1.5, linestyle="--", alpha=0.7, zorder=1)
        ax.fill_between(x_line, y_line - ci, y_line + ci, color=color, alpha=0.12, zorder=0)
        return r, p

    def _draw_scatter(ax, agg, title, label_top=6, label_bottom=3, show_legend=True):
        sizes = np.log1p(agg["support"].values) * 12 + 15
        colors = np.where(agg["delta_f1"] >= 0, CB_PALETTE["histology"], "#D55E00")
        sc = ax.scatter(agg["f1_base"], agg["delta_f1"],
                        s=sizes, c=colors, alpha=0.75,
                        edgecolors="white", linewidths=0.5, zorder=3)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.45)
        ax.axvline(0.7, color="gray", linewidth=0.7, linestyle=":", alpha=0.5)

        out = _reg_line_ci(agg["f1_base"].values, agg["delta_f1"].values, ax, color="gray")
        if out:
            r, p = out
            ax.text(0.04, 0.97, f"r = {r:.2f}, p = {p:.3f}",
                    transform=ax.transAxes, fontsize=10, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))

        # Label extremes
        top_idx = agg["delta_f1"].nlargest(label_top).index
        bot_idx = agg["delta_f1"].nsmallest(label_bottom).index
        for idx in top_idx.union(bot_idx):
            row = agg.loc[idx]
            name = row["class_name"].replace("_", " ")
            # Shorten long TCGA names
            if len(name) > 22:
                name = name[:20] + "…"
            ax.annotate(name, (row["f1_base"], row["delta_f1"]),
                        fontsize=7.5, xytext=(5, 3), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5), zorder=4)

        ax.set_xlabel("Baseline F1 (no TTA)")
        ax.set_ylabel("TTA Δ F1 (D4 − baseline)")

        if show_legend:
            handles = [
                Line2D([0], [0], marker="o", color=CB_PALETTE["histology"],
                       markerfacecolor=CB_PALETTE["histology"], markersize=8,
                       linestyle="None", label="TTA helps (Δ F1 ≥ 0)"),
                Line2D([0], [0], marker="o", color="#D55E00",
                       markerfacecolor="#D55E00", markersize=8,
                       linestyle="None", label="TTA hurts (Δ F1 < 0)"),
                Line2D([0], [0], marker="o", color="gray", markersize=6,
                       markerfacecolor="gray", linestyle="None",
                       label="Size ∝ log(class support)"),
            ]
            ax.legend(handles=handles, fontsize=9, loc="upper right")

    with plt.rc_context(PUB_RCPARAMS):
        fig = plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], wspace=0.35)
        ax_main = fig.add_subplot(gs[0])
        ax_inset = fig.add_subplot(gs[1])

        tcga = _compute_class_deltas(pc, "tcga-ut")
        nct  = _compute_class_deltas(pc, "nct-crc-100k")

        _draw_scatter(ax_main, tcga, "TCGA-UT (31 classes)", label_top=6, label_bottom=3)
        _draw_scatter(ax_inset, nct,  "NCT-CRC-100K (9 classes)", label_top=5, label_bottom=2,
                      show_legend=False)

        fig.tight_layout()
        fig.savefig(out_dir / "plot16_perclass_scatter.png", **SAVEKW)
        plt.close(fig)
    print("  Saved plot16_perclass_scatter.png")


# ---------------------------------------------------------------------------
# Plot 17 — Selective TTA: pareto curves + cross-dataset summary
# ---------------------------------------------------------------------------

def _load_selective_data(raw_dir: str = "results/raw") -> pd.DataFrame:
    """Load all selective TTA JSON result files into a single DataFrame."""
    import json, glob
    records = []
    for fpath in glob.glob(f"{raw_dir}/selective_tta_*.json"):
        try:
            with open(fpath) as f:
                records.extend(json.load(f))
        except Exception:
            continue
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["model_type"] = df["model"].map(MODEL_TYPE_MAP).fillna("unknown")
    return df


def plot_selective_pareto_clean(df_sel: pd.DataFrame, out_dir: Path):
    """Figure 3 (pub): selective TTA pareto + cross-dataset summary.

    Panel A: TCGA-UT pareto curves (coverage % vs Δ balanced accuracy),
             histology FM vs general, mean ± band across models × seeds.
             Aggregates by threshold_frac to get clean monotonic curves.
    Panel B: At t=0.6 threshold (the elbow: ~35% coverage, ~84% benefit for
             histology FMs), shows % of full-D4 benefit and % passes saved
             per dataset.
    """
    if df_sel.empty:
        print("  WARNING: no selective TTA data found, skipping plot17")
        return

    import matplotlib.gridspec as gridspec

    key = ["model", "dataset", "backbone_mode", "seed"]

    # Per-(model, dataset, seed) baseline and full-D4 reference
    base = df_sel[df_sel["strategy"] == "none"][
        key + ["balanced_acc"]
    ].rename(columns={"balanced_acc": "bacc_base"})

    d4_full = df_sel[df_sel["strategy"] == "d4"][
        key + ["balanced_acc"]
    ].rename(columns={"balanced_acc": "bacc_d4"})

    selective = df_sel[df_sel["strategy"].str.startswith("selective_", na=False)].copy()
    selective["threshold_frac"] = selective["threshold_frac"].astype(float)
    selective = selective.merge(base, on=key).merge(d4_full, on=key)
    selective["delta_pp"] = (selective["balanced_acc"] - selective["bacc_base"]) * 100
    selective["d4_delta_pp"] = (selective["bacc_d4"] - selective["bacc_base"]) * 100

    # Aggregate by threshold_frac (the true control variable) — gives clean curves
    def _pareto_agg_by_thresh(data, dataset, model_type, backbone_mode="frozen"):
        sub = data[
            (data["dataset"] == dataset) &
            (data["model_type"] == model_type) &
            (data["backbone_mode"] == backbone_mode)
        ]
        g = sub.groupby("threshold_frac", observed=True).agg(
            coverage=("coverage", "mean"),
            delta_mean=("delta_pp", "mean"),
            delta_sem=("delta_pp", lambda x: x.sem() if len(x) > 1 else 0),
            d4_mean=("d4_delta_pp", "mean"),
        ).reset_index().sort_values("coverage")
        # Add full-D4 endpoint
        d4_rows = d4_full[
            (d4_full["dataset"] == dataset) &
            (d4_full["backbone_mode"] == backbone_mode)
        ].merge(base, on=key)
        d4_rows = d4_rows[d4_rows["model"].isin(sub["model"].unique())]
        if not d4_rows.empty:
            d4_delta = (d4_rows["bacc_d4"] - d4_rows["bacc_base"]).mean() * 100
            d4_sem   = (d4_rows["bacc_d4"] - d4_rows["bacc_base"]).sem() * 100
            endpoint = pd.DataFrame({"threshold_frac": [np.nan], "coverage": [1.0],
                                     "delta_mean": [d4_delta], "delta_sem": [d4_sem],
                                     "d4_mean": [d4_delta]})
            g = pd.concat([g, endpoint], ignore_index=True).sort_values("coverage")
        return g

    # Panel B: at threshold t=0.6, group-level frac of D4 benefit (not per-row)
    t_elbow = 0.6
    t_elbow_data = selective[
        (np.abs(selective["threshold_frac"] - t_elbow) < 0.01) &
        (selective["backbone_mode"] == "frozen") &
        (selective["model_type"] == "histology")
    ]
    summary_rows = []
    for ds in DATASET_ORDER:
        sub = t_elbow_data[t_elbow_data["dataset"] == ds]
        if sub.empty:
            continue
        frac_benefit = sub["delta_pp"].mean() / max(sub["d4_delta_pp"].mean(), 1e-6) * 100
        passes_saved = sub["pct_passes_saved"].mean()   # already in %
        passes_saved_sem = sub["pct_passes_saved"].sem() if len(sub) > 1 else 0
        summary_rows.append({
            "dataset": ds,
            "frac_benefit": frac_benefit,
            "passes_saved_mean": passes_saved,
            "passes_saved_sem": passes_saved_sem,
            "coverage_mean": sub["coverage"].mean(),
        })
    summary = pd.DataFrame(summary_rows)

    with plt.rc_context(PUB_RCPARAMS):
        fig = plt.figure(figsize=(13, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], wspace=0.38)
        ax_a = fig.add_subplot(gs[0])
        ax_b = fig.add_subplot(gs[1])

        # Panel A: pareto curves for TCGA-UT
        for mt in ["histology", "general"]:
            color = CB_PALETTE[mt]
            agg = _pareto_agg_by_thresh(selective, "tcga-ut", mt)
            if agg.empty:
                continue
            label = "Histology FMs" if mt == "histology" else "General models"
            ax_a.plot(agg["coverage"] * 100, agg["delta_mean"],
                      color=color, linewidth=2.2, marker="o", markersize=5,
                      label=label, zorder=3)
            ax_a.fill_between(
                agg["coverage"] * 100,
                agg["delta_mean"] - agg["delta_sem"],
                agg["delta_mean"] + agg["delta_sem"],
                color=color, alpha=0.18, zorder=2
            )

        # Elbow annotation at ~35% coverage (t=0.6)
        elbow_x = selective[
            (selective["dataset"] == "tcga-ut") &
            (selective["backbone_mode"] == "frozen") &
            (selective["model_type"] == "histology") &
            (np.abs(selective["threshold_frac"] - t_elbow) < 0.01)
        ]["coverage"].mean() * 100
        ax_a.axvline(elbow_x, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
        ax_a.text(elbow_x + 1.5, 0.02,
                  f"~{elbow_x:.0f}% coverage\n84% of D4 benefit\n(hist. FMs, t=0.6)",
                  fontsize=9, color="gray", va="bottom")

        ax_a.set_xlabel("Samples Receiving TTA (%)")
        ax_a.set_ylabel("D4 TTA Δ Balanced Accuracy (pp)")
        ax_a.legend(fontsize=11, loc="lower right")

        # Panel B
        ds_present = [ds for ds in DATASET_ORDER if ds in summary["dataset"].values]
        x = np.arange(len(ds_present))
        w = 0.36
        c_benefit = CB_PALETTE["histology"]
        c_saved   = CB_PALETTE["general"]

        ax_b.bar(x - w / 2,
                 [summary[summary["dataset"] == ds]["frac_benefit"].values[0] for ds in ds_present],
                 w, color=c_benefit, alpha=0.8, capsize=3, label="% of full-D4 benefit",
                 edgecolor="white")
        ax_b.bar(x + w / 2,
                 [summary[summary["dataset"] == ds]["passes_saved_mean"].values[0] for ds in ds_present],
                 w,
                 yerr=[summary[summary["dataset"] == ds]["passes_saved_sem"].values[0]
                       for ds in ds_present],
                 color=c_saved, alpha=0.8, capsize=3, label="% inference passes saved",
                 edgecolor="white")

        ax_b.axhline(80, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
        ax_b.set_xticks(x)
        _DS_SHORT_B = {"tcga-ut": "TCGA-UT", "nct-crc-100k": "NCT-100K",
                       "nct-crc-nonorm": "NCT-NoNorm", "mhist": "MHIST"}
        ax_b.set_xticklabels([_DS_SHORT_B.get(ds, ds) for ds in ds_present],
                             fontsize=10, rotation=25, ha="right")
        ax_b.set_ylabel("Percentage (%)")
        ax_b.legend(fontsize=9, loc="lower right")
        ax_b.set_ylim(0, 115)

        fig.tight_layout()
        fig.savefig(out_dir / "plot17_selective_pareto.png", **SAVEKW)
        plt.close(fig)
    print("  Saved plot17_selective_pareto.png")


# ---------------------------------------------------------------------------
# Plot 18 — Qualitative examples: TTA corrects uncertain patches
# ---------------------------------------------------------------------------

def plot_qualitative_examples(per_sample_dir, out_dir: Path):
    """Figure 5 (pub): patch grid showing TTA outcomes vs annotator ambiguity.

    Reads per-sample CSVs from eval_per_sample.py output.
    Row 1: MHIST patches (HP vs SSA), grouped by TTA outcome × agreement.
    Row 2: NCT-CRC-100K patches grouped by TTA outcome × baseline confidence.

    Requires:
        results/per_sample/per_sample_mhist_phikon_seed42.csv
        results/per_sample/per_sample_nct-crc-100k_phikon_seed42.csv
    Both produced by: python eval_per_sample.py --model phikon --seed 42 ...
    """
    import matplotlib.gridspec as gridspec
    import json

    per_sample_dir = Path(per_sample_dir)
    mhist_csv  = per_sample_dir / "per_sample_mhist_phikon_seed42.csv"
    nct_csv    = per_sample_dir / "per_sample_nct-crc-100k_phikon_seed42.csv"

    if not mhist_csv.exists():
        print(f"  WARNING: {mhist_csv} not found. "
              "Run: sbatch scripts/run_per_sample_inference.sh  (GPU required)")
        return
    if not nct_csv.exists():
        print(f"  WARNING: {nct_csv} not found. Skipping NCT-CRC row.")

    mhist = pd.read_csv(mhist_csv)
    mhist["baseline_softmax"] = mhist["baseline_softmax"].apply(json.loads)
    mhist["tta_softmax"]      = mhist["tta_softmax"].apply(json.loads)
    classes_mhist = ["HP", "SSA"]

    def _select_examples(df, outcome, sort_col, ascending, n=2):
        sub = df[df["outcome"] == outcome].copy()
        if sub.empty:
            return sub
        return sub.sort_values(sort_col, ascending=ascending).head(n)

    # MHIST examples: 2 corrected (lowest agreement) + 2 both_correct (highest agreement)
    corrected    = _select_examples(mhist, "corrected",    "agreement", ascending=True,  n=2)
    both_correct = _select_examples(mhist, "both_correct", "agreement", ascending=False, n=2)
    mhist_examples = pd.concat([corrected, both_correct], ignore_index=True)

    MHIST_IMG_DIR = Path("data/mhist/images")
    AGREE_COLORS = {True: "#2ca02c", False: "#d62728"}  # green=TTA correct, red=wrong

    def _agreement_label(a):
        n_agree = round(a * 7)
        return f"{max(n_agree, 7-n_agree)}/7 agree"

    with plt.rc_context(PUB_RCPARAMS):
        fig = plt.figure(figsize=(14, 6))
        n_mhist = len(mhist_examples)
        has_nct = nct_csv.exists()
        nrows = 2 if has_nct else 1
        gs = gridspec.GridSpec(nrows, n_mhist, hspace=0.55, wspace=0.08)

        # --- Row 1: MHIST ---
        for col, (_, row) in enumerate(mhist_examples.iterrows()):
            ax_img = fig.add_subplot(gs[0, col])
            img_path = MHIST_IMG_DIR / row["image_id"]
            if img_path.exists():
                from PIL import Image as PILImage
                img = PILImage.open(img_path).convert("RGB")
                ax_img.imshow(img)

            # Color frame by TTA outcome
            fc = "#2ca02c" if row["tta_correct"] else "#d62728"
            for spine in ax_img.spines.values():
                spine.set_edgecolor(fc)
                spine.set_linewidth(3)
            ax_img.set_xticks([])
            ax_img.set_yticks([])

            # Title: true label + agreement
            agree_str = _agreement_label(row["agreement"])
            outcome_str = "TTA corrected" if row["outcome"] == "corrected" else "Both correct"
            ax_img.set_title(
                f"{row['true_label']}  ({agree_str})\n{outcome_str}",
                fontsize=9, pad=3
            )

            # Softmax bars below
            ax_bar_pos = ax_img.get_position()
            ax_bar = fig.add_axes([
                ax_bar_pos.x0, ax_bar_pos.y0 - 0.12,
                ax_bar_pos.width, 0.09
            ])
            x = np.arange(len(classes_mhist))
            w = 0.35
            base_s = row["baseline_softmax"]
            tta_s  = row["tta_softmax"]
            ax_bar.bar(x - w/2, base_s, w, color="#aec7e8", label="Baseline", edgecolor="white")
            ax_bar.bar(x + w/2, tta_s,  w, color=CB_PALETTE["histology"],
                       label="D4 TTA", edgecolor="white")
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(classes_mhist, fontsize=8)
            ax_bar.set_ylim(0, 1.05)
            ax_bar.set_yticks([0, 0.5, 1.0])
            ax_bar.tick_params(axis="y", labelsize=7)
            ax_bar.axhline(0.5, color="gray", linewidth=0.6, linestyle=":")
            if col == 0:
                ax_bar.set_ylabel("Softmax", fontsize=8)
                ax_bar.legend(fontsize=7, loc="upper right")

        # --- Row 2: NCT-CRC ---
        if has_nct:
            nct = pd.read_csv(nct_csv)
            nct["baseline_softmax"] = nct["baseline_softmax"].apply(json.loads)
            nct["tta_softmax"]      = nct["tta_softmax"].apply(json.loads)
            classes_nct = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

            corrected_nct    = _select_examples(nct, "corrected",    "baseline_conf", ascending=True, n=2)
            both_correct_nct = _select_examples(nct, "both_correct", "baseline_conf", ascending=False, n=2)
            nct_examples = pd.concat([corrected_nct, both_correct_nct], ignore_index=True)

            for col, (_, row) in enumerate(nct_examples.iterrows()):
                ax_img = fig.add_subplot(gs[1, col])
                ax_img.set_facecolor("#f0f0f0")
                ax_img.text(0.5, 0.5, f"{row['true_label']}\n(no local image)",
                            ha="center", va="center", fontsize=8, transform=ax_img.transAxes)
                fc = "#2ca02c" if row["tta_correct"] else "#d62728"
                for spine in ax_img.spines.values():
                    spine.set_edgecolor(fc)
                    spine.set_linewidth(3)
                ax_img.set_xticks([])
                ax_img.set_yticks([])
                conf_str = f"conf={row['baseline_conf']:.2f}"
                outcome_str = "TTA corrected" if row["outcome"] == "corrected" else "Both correct"
                ax_img.set_title(f"{row['true_label']}  ({conf_str})\n{outcome_str}", fontsize=9, pad=3)

        fig.savefig(out_dir / "plot18_qualitative_examples.png", **SAVEKW)
        plt.close(fig)
    print("  Saved plot18_qualitative_examples.png")


# ---------------------------------------------------------------------------
# Plot 19 — Correction vs corruption rate: diverging lollipop
# ---------------------------------------------------------------------------

def plot_correction_rates(df: pd.DataFrame, out_dir: Path):
    """Figure pub: diverging lollipop showing per-model correction and corruption rates.

    Correction rate = fraction of baseline errors fixed by D4 TTA.
    Corruption rate = fraction of baseline correct predictions broken by D4 TTA.
    Models sorted by correction rate within model-type groups.
    Individual dataset points overlaid to show cross-dataset consistency.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    with plt.rc_context(PUB_RCPARAMS):
        # --- Build rates ---
        base = df[(df["strategy"] == "none") & (df["aggregation"] == "mean") &
                  (df["backbone_mode"] == "frozen")].copy()
        tta  = df[(df["strategy"] == "d4")  & (df["aggregation"] == "mean") &
                  (df["backbone_mode"] == "frozen")].copy()
        tta  = tta.dropna(subset=["n_corrected", "n_corrupted"])

        merge_cols = CONFIG_COLS + ["seed"]
        tta = tta.merge(
            base[merge_cols + ["n_wrong", "n_correct"]].rename(
                columns={"n_wrong": "base_wrong", "n_correct": "base_correct"}),
            on=merge_cols,
        )
        tta["correction_rate"] = tta["n_corrected"] / tta["base_wrong"].clip(lower=1) * 100
        tta["corruption_rate"] = tta["n_corrupted"] / tta["base_correct"].clip(lower=1) * 100

        # Per-model × dataset aggregate (mean across seeds)
        agg_ds = tta.groupby(["model", "model_type", "dataset"], observed=True).agg(
            corr_r=("correction_rate", "mean"),
            corrupt_r=("corruption_rate", "mean"),
        ).reset_index()

        # Overall per-model aggregate (mean across seeds × datasets)
        agg = tta.groupby(["model", "model_type"], observed=True).agg(
            corr_mean=("correction_rate", "mean"),
            corr_sem=("correction_rate", lambda x: x.sem() if len(x) > 1 else 0),
            corrupt_mean=("corruption_rate", "mean"),
            corrupt_sem=("corruption_rate", lambda x: x.sem() if len(x) > 1 else 0),
        ).reset_index()

        # Build ordered y-axis: within each type group, sort by corr_mean descending
        present_types = [t for t in ["histology", "general", "equivariant"]
                         if not agg[agg["model_type"] == t].empty]
        groups = []
        for mtype in present_types:
            sub = agg[agg["model_type"] == mtype].sort_values("corr_mean", ascending=False)
            groups.append(sub)
        agg_sorted = pd.concat(groups, ignore_index=True)
        n_models = len(agg_sorted)

        # Y positions: assign consecutive integers bottom-up, then reverse for top-down display
        type_sizes = [len(agg[agg["model_type"] == t]) for t in present_types]
        total_rows = sum(type_sizes) + max(len(type_sizes) - 1, 0)   # gaps between groups
        y_pos_bottom_up = []
        y = 0
        for ts in type_sizes:
            y_pos_bottom_up.extend(range(y, y + ts))
            y += ts + 1
        # Reverse so top model has highest y (matplotlib default: y=0 at bottom)
        y_max = max(y_pos_bottom_up)
        y_pos = [y_max - yy for yy in y_pos_bottom_up]

        # Group midpoints in display coordinates
        group_mids = []
        y = 0
        for ts in type_sizes:
            block = [y_max - (y + k) for k in range(ts)]
            group_mids.append(float(np.mean(block)) if block else 0.0)
            y += ts + 1

        # Compute axis limits before drawing so annotation positions are stable
        xlim_val = max(
            agg_sorted["corr_mean"].max() + agg_sorted["corr_sem"].max() + 3,
            agg_sorted["corrupt_mean"].max() + agg_sorted["corrupt_sem"].max() + 3,
        )
        xlim_val = max(xlim_val, 20)
        ratio_x = xlim_val * 1.06   # fixed x position for ratio annotations and group labels

        fig, ax = plt.subplots(figsize=(10, max(7, n_models * 0.55 + 1.5)))

        # Colors
        COR_COLOR    = "#0072B2"   # blue  — correction
        CORRUPT_COLOR = "#D55E00"  # vermillion — corruption

        # Draw lollipops and individual dataset points
        for idx, (_, row) in enumerate(agg_sorted.iterrows()):
            yp = y_pos[idx]
            model = row["model"]

            # Correction lollipop (right, blue)
            ax.plot([0, row["corr_mean"]], [yp, yp], color=COR_COLOR, lw=2, zorder=2)
            ax.plot(row["corr_mean"], yp, "o", color=COR_COLOR, ms=8, zorder=3)
            ax.errorbar(row["corr_mean"], yp, xerr=row["corr_sem"],
                        fmt="none", color=COR_COLOR, capsize=3, elinewidth=1.5, zorder=4)

            # Corruption lollipop (left, vermillion)
            ax.plot([0, -row["corrupt_mean"]], [yp, yp], color=CORRUPT_COLOR, lw=2, zorder=2)
            ax.plot(-row["corrupt_mean"], yp, "o", color=CORRUPT_COLOR, ms=8, zorder=3)
            ax.errorbar(-row["corrupt_mean"], yp, xerr=row["corrupt_sem"],
                        fmt="none", color=CORRUPT_COLOR, capsize=3, elinewidth=1.5, zorder=4)

            # Individual dataset scatter (small dots)
            ds_sub = agg_ds[agg_ds["model"] == model]
            jitter = np.linspace(-0.25, 0.25, len(ds_sub))
            for j, (_, ds_row) in enumerate(ds_sub.iterrows()):
                dc = DS_COLORS.get(ds_row["dataset"], "#888888")
                ax.scatter(ds_row["corr_r"],    yp + jitter[j], color=dc, s=18, alpha=0.7, zorder=5)
                ax.scatter(-ds_row["corrupt_r"], yp + jitter[j], color=dc, s=18, alpha=0.7, zorder=5)

            # Ratio annotation on the right margin
            ratio = row["corr_mean"] / max(row["corrupt_mean"], 0.01)
            ax.text(ratio_x, yp, f"{ratio:.1f}×", va="center", ha="left",
                    fontsize=9, color="#333333")

        # Y tick labels (full model names)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [MODEL_FULL_NAMES.get(r["model"], r["model"]) for _, r in agg_sorted.iterrows()],
            fontsize=11
        )

        # Group bracket labels on the right
        type_labels = {"histology": "Histology FMs", "general": "General Models",
                       "equivariant": "Equivariant"}
        for mid, mtype in zip(group_mids, ["histology", "general", "equivariant"]):
            ax.text(ratio_x + 6, mid, type_labels[mtype], va="center", ha="left",
                    fontsize=10, color=CB_PALETTE[mtype], fontweight="bold", rotation=0)

        # Axes
        ax.axvline(0, color="black", lw=1.0, zorder=1)
        ax.set_xlabel("Rate (%) of baseline predictions changed by D4 TTA", fontsize=14)

        # Format x-axis symmetrically with absolute labels
        ax.set_xlim(-xlim_val * 0.55, xlim_val * 1.55)
        ticks = np.arange(0, xlim_val + 1, 5)
        ax.set_xticks(np.concatenate([-ticks[1:][::-1], ticks]))
        ax.set_xticklabels([f"{abs(t):.0f}" for t in np.concatenate([-ticks[1:][::-1], ticks])],
                           fontsize=11)

        # Direction labels (use data-space y at top of plot)
        y_top = max(y_pos) + 1.0
        ax.text(-xlim_val * 0.5, y_top, "← Corruption rate",
                ha="left", va="bottom", fontsize=10, color=CORRUPT_COLOR, style="italic")
        ax.text(xlim_val * 0.02, y_top, "Correction rate →",
                ha="left", va="bottom", fontsize=10, color=COR_COLOR, style="italic")

        # Legend: correction/corruption + datasets
        legend_elements = [
            Patch(facecolor=COR_COLOR,     label="Correction rate (errors fixed)"),
            Patch(facecolor=CORRUPT_COLOR, label="Corruption rate (correct → wrong)"),
        ]
        for ds, dc in DS_COLORS.items():
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor=dc,
                       markersize=6, label=DATASET_LABELS.get(ds, ds))
            )
        ax.legend(handles=legend_elements, fontsize=9, loc="lower right",
                  framealpha=0.85, ncol=2)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / "plot19_correction_rates.png", **SAVEKW)
        plt.close(fig)
    print("  Saved plot19_correction_rates.png")


# ---------------------------------------------------------------------------
# Figure 2 combined (pub) — panel (a) + panel (b) stacked, MELBA two-column
# ---------------------------------------------------------------------------

def _draw_panel_c(ax, df_canon: pd.DataFrame):
    """Panel (c): pretraining scale (params) vs TTA Δ for histology FMs.

    Shows TCGA-UT and NCT-CRC-100K only (most informative datasets).
    One regression line per dataset; Spearman ρ annotated per dataset.
    """
    from scipy.stats import spearmanr as _spearmanr

    FM_MODELS = set(FM_META.keys())
    SHOW_DS = ["tcga-ut", "nct-crc-100k"]

    sub = df_canon[df_canon["model"].isin(FM_MODELS) & df_canon["dataset"].isin(SHOW_DS)]
    base = sub[sub["strategy"] == "none"][["model", "dataset", "seed", "balanced_acc"]].rename(
        columns={"balanced_acc": "base_acc"})
    best = sub[sub["strategy"] == "d4"][["model", "dataset", "seed", "balanced_acc"]].rename(
        columns={"balanced_acc": "d4_acc"})
    merged = base.merge(best, on=["model", "dataset", "seed"])
    merged["delta"] = (merged["d4_acc"] - merged["base_acc"]) * 100
    agg = merged.groupby(["model", "dataset"])["delta"].mean().reset_index()
    agg["params_m"] = agg["model"].map(lambda m: FM_META[m]["params_m"])

    for i, ds in enumerate(SHOW_DS):
        sub_ds = agg[agg["dataset"] == ds]
        x = sub_ds["params_m"].values
        y = sub_ds["delta"].values
        color = DS_COLORS[ds]

        ax.scatter(x, y, color=color, s=40, zorder=3, alpha=0.9)

        if len(x) >= 3:
            xp = np.log10(x)
            coef = np.polyfit(xp, y, 1)
            xs = np.linspace(xp.min(), xp.max(), 200)
            ax.plot(10**xs, np.polyval(coef, xs),
                    color=color, linewidth=1.2, linestyle="--", alpha=0.7, zorder=2)
            rho, _ = _spearmanr(x, y)
            ax.text(0.97, 0.97 - i * 0.12, f"ρ = {rho:.2f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (M)", fontsize=10)
    ax.set_ylabel("TTA Δ Bal. Acc. (pp)", fontsize=10)
    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle=":")
    ax.tick_params(axis="both", labelsize=9)


def plot_figure2_combined(df: pd.DataFrame, out_dir: Path):
    """Figure 2 (pub): panel (a) top-left, panel (b) bottom-left, panel (c) right.

    Panel (a): per-dataset boxplot of TTA Δ balanced accuracy by model type.
    Panel (b): 1×4 faceted scatter of baseline acc vs TTA Δ per dataset.
    Panel (c): pretraining scale (params) vs TTA Δ for histology FMs.
    """
    datasets_present = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    n_ds = len(datasets_present)

    with plt.rc_context(PUB_RCPARAMS):
        fig = plt.figure(figsize=(10, 5.0))
        gs_outer = fig.add_gridspec(
            2, 1, height_ratios=[0.75, 0.85], hspace=0.08,
            top=0.96, bottom=0.22, left=0.08, right=0.99,
        )

        from matplotlib.transforms import blended_transform_factory

        df_canon = _primary_config(df)

        # --- Panel (a) ---
        ax_a = fig.add_subplot(gs_outer[0])
        _draw_panel_a(ax_a, df_canon, xtick_fontsize=11)
        trans_a = blended_transform_factory(fig.transFigure, ax_a.transAxes)
        fig.text(0.04, 1.14, "(a)", transform=trans_a,
                 fontsize=13, fontweight="bold", va="bottom")

        # --- Panel (b) ---
        gs_b = gs_outer[1].subgridspec(1, n_ds, wspace=0.08)
        first_ax = fig.add_subplot(gs_b[0, 0])
        axes_b = [first_ax]
        for i in range(1, n_ds):
            axes_b.append(fig.add_subplot(gs_b[0, i], sharey=first_ax))
        handles_b = _draw_panel_b(axes_b, df_canon, show_protocol=False)
        trans_b = blended_transform_factory(fig.transFigure, axes_b[0].transAxes)
        fig.text(0.04, 0.96, "(b)", transform=trans_b,
                 fontsize=13, fontweight="bold", va="bottom")

        # Align panel (a) dataset midlines to panel (b) facet centres.
        fig.canvas.draw()
        pos_a = ax_a.get_position()
        la, wa = pos_a.x0, pos_a.x1 - pos_a.x0
        facet_cx = [(ax_b.get_position().x0 + ax_b.get_position().x1) / 2
                    for ax_b in axes_b]
        r = [(c - la) / wa for c in facet_cx]
        A = np.array([[1 - r[0], r[0]], [1 - r[-1], r[-1]]])
        x0_ds, x1_ds = np.linalg.solve(A, np.array([0.0, float(n_ds - 1)]))
        ax_a.set_xlim(x0_ds, x1_ds)

        # Single shared legend for both panels, centred below figure
        fig.legend(
            handles=handles_b,
            loc="lower center",
            bbox_to_anchor=(0.48, -0.02),
            ncol=len(handles_b),
            fontsize=13,
            frameon=True,
            framealpha=0.95,
        )

        stem = out_dir / "figure2"
        fig.savefig(str(stem) + ".png", dpi=300, facecolor="white")
        fig.savefig(str(stem) + ".pdf", dpi=300, facecolor="white")
        plt.close(fig)
    print("  Saved figure2.png/pdf")


# ---------------------------------------------------------------------------
# Figure 4 — Selective TTA Pareto curves (per-model + family average)
# ---------------------------------------------------------------------------

def plot_figure4_selective_pareto(df, out_dir: Path):
    """Figure 4: selective TTA efficiency–accuracy Pareto curves.

    Two panels side-by-side: histology FMs (left) and general-purpose models (right).
    Loads per-sample multi-view logits from logits/ and sweeps 50 entropy thresholds.
    """
    import torch
    import torch.nn.functional as F
    from collections import defaultdict
    from sklearn.metrics import balanced_accuracy_score

    LOGITS_DIR = Path(__file__).parent.parent / "logits"
    if not LOGITS_DIR.exists():
        print(f"  WARNING: logits dir not found at {LOGITS_DIR}, skipping fig4")
        return

    THRESHOLDS = np.linspace(0, 1, 50)

    # dataset -> model -> list of (fracs_array, deltas_array)
    all_model_curves: dict = defaultdict(lambda: defaultdict(list))

    pt_files = sorted(LOGITS_DIR.rglob("*.pt"))
    if not pt_files:
        print(f"  WARNING: no .pt files found in {LOGITS_DIR}, skipping fig4")
        return

    print(f"  Loading {len(pt_files)} logit files for Figure 4...")
    for pt_file in pt_files:
        try:
            ck = torch.load(pt_file, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"    WARNING: skipping {pt_file.name}: {e}")
            continue

        if ck.get("backbone_mode") != "frozen":
            continue
        if ck.get("aug_tag", "aug") != "aug":
            continue

        dataset = ck.get("dataset", "")
        model = ck.get("model", "")
        if MODEL_TYPE_MAP.get(model) not in ("histology", "general"):
            continue

        logits = ck["logits"].float()   # (n_views, N, C)
        labels = ck["labels"]           # (N,)
        _, N, C = logits.shape

        base_probs = F.softmax(logits[0], dim=-1)
        eps = 1e-12
        base_ent = (
            -(base_probs * (base_probs + eps).log()).sum(-1) / np.log(C)
        ).numpy()
        base_preds = base_probs.argmax(-1).numpy()
        d4_preds = F.softmax(logits, dim=-1).mean(0).argmax(-1).numpy()
        labels_np = labels.numpy()

        baseline_ba = balanced_accuracy_score(labels_np, base_preds) * 100

        fracs = np.empty(len(THRESHOLDS))
        deltas = np.empty(len(THRESHOLDS))
        for i, t in enumerate(THRESHOLDS):
            mask = base_ent > t
            sel_preds = np.where(mask, d4_preds, base_preds)
            fracs[i] = mask.mean() * 100
            deltas[i] = balanced_accuracy_score(labels_np, sel_preds) * 100 - baseline_ba

        all_model_curves[dataset][model].append((fracs, deltas))

    if not all_model_curves:
        print("  WARNING: no usable logit files, skipping fig4")
        return

    def _model_avg(model_curves, model):
        curves = model_curves[model]
        return (np.array([c[0] for c in curves]).mean(0),
                np.array([c[1] for c in curves]).mean(0))

    def _family_avg(model_curves, models):
        all_f, all_d = zip(*[_model_avg(model_curves, m) for m in models])
        return np.array(all_f).mean(0), np.array(all_d).mean(0)

    ANN_THRESHOLDS = [0.3, 0.5, 0.7]

    def _draw_panel(ax, model_curves, models, colors, avg_color, title,
                    label_x, ha, arrow_relpos, legend_loc, y_min, y_max,
                    ann_y_top=None, legend_bbox=None):
        y_range = y_max - y_min
        label_y_top   = ann_y_top if ann_y_top is not None else y_min + y_range * 0.38
        label_spacing = y_range * 0.065

        for m, color in zip(models, colors):
            f, d = _model_avg(model_curves, m)
            order = np.argsort(f)
            ax.plot(f[order], d[order], color=color, lw=0.8, alpha=0.25,
                    label=MODEL_FULL_NAMES.get(m, m))

        if models:
            fam_f, fam_d = _family_avg(model_curves, models)
            order = np.argsort(fam_f)
            ax.plot(fam_f[order], fam_d[order], color=avg_color, lw=2.5,
                    alpha=1.0, label="Family avg", zorder=4)

            for i, t_val in enumerate(ANN_THRESHOLDS):
                frac_pt  = float(np.interp(t_val, THRESHOLDS, fam_f))
                delta_pt = float(np.interp(t_val, THRESHOLDS, fam_d))
                pct_saved = 100.0 - frac_pt
                label_y = label_y_top - i * label_spacing
                ax.plot(frac_pt, delta_pt, "o", color=avg_color, ms=5, zorder=5)
                ax.annotate(
                    f"t={t_val} ({pct_saved:.0f}% saved)",
                    xy=(frac_pt, delta_pt),
                    xytext=(label_x, label_y),
                    xycoords="data", textcoords="data",
                    fontsize=7, ha=ha, va="center", color=avg_color,
                    arrowprops=dict(arrowstyle="-", color=avg_color, lw=0.5,
                                   relpos=arrow_relpos, shrinkA=2, shrinkB=3),
                    zorder=6,
                )

        ax.axhline(0, color="black", lw=1.2, zorder=2)
        ax.set_xlim(-2, 100)
        ax.set_ylim(y_min, y_max)
        ax.set_title(title, fontsize=13)
        legend_kw = dict(fontsize=8, loc=legend_loc, ncol=1, framealpha=0.85,
                         handlelength=1.5, labelspacing=0.25)
        if legend_bbox is not None:
            legend_kw["bbox_to_anchor"] = legend_bbox
        ax.legend(**legend_kw)
        for _gl in ax.get_xgridlines() + ax.get_ygridlines():
            _gl.set_alpha(0.15)

    DS_TITLES = {
        "tcga-ut": "TCGA-UT (31 cls)",
        "nct-crc-100k": "NCT-CRC-100K (9 cls)",
        "nct-crc-nonorm": "NCT-CRC-NoNorm (9 cls)",
        "mhist": "MHIST (2 cls)",
    }

    for dataset, model_curves in all_model_curves.items():
        histo_models = [m for m in MODEL_ORDER
                        if MODEL_TYPE_MAP.get(m) == "histology" and m in model_curves]
        general_models = [m for m in MODEL_ORDER
                          if MODEL_TYPE_MAP.get(m) == "general" and m in model_curves]

        all_deltas = []
        for m in histo_models + general_models:
            _, d = _model_avg(model_curves, m)
            all_deltas.extend(d.tolist())
        y_min = min(all_deltas) - 0.15
        y_max = min(max(all_deltas) + 0.45, 1.5)
        y_range = y_max - y_min
        zero_frac = (0 - y_min) / y_range

        n_histo = len(histo_models)
        n_gen = len(general_models)
        histo_colors = plt.cm.Blues(np.linspace(0.35, 0.85, max(n_histo, 1)))
        gen_colors = plt.cm.Oranges(np.linspace(0.40, 0.85, max(n_gen, 1)))

        with plt.rc_context(PUB_RCPARAMS):
            fig, (ax_l, ax_r) = plt.subplots(
                1, 2, figsize=(6.8, 3.8), sharey=True,
                constrained_layout=True,
            )

            _draw_panel(ax_l, model_curves, histo_models, histo_colors,
                        CB_PALETTE["histology"], "Histology Foundation Models",
                        label_x=2, ha="left", arrow_relpos=(1, 0.5),
                        legend_loc="upper left", y_min=y_min, y_max=y_max)

            _draw_panel(ax_r, model_curves, general_models, gen_colors,
                        CB_PALETTE["general"], "General-Purpose Models",
                        label_x=2, ha="left", arrow_relpos=(1, 0.5),
                        legend_loc="lower right", y_min=y_min, y_max=y_max,
                        legend_bbox=(1.0, zero_frac + 0.04),
                        ann_y_top=y_max - 0.25)

            ax_l.set_ylabel("Δ Balanced Accuracy (pp)")
            ds_title = DS_TITLES.get(dataset, dataset)
            fig.suptitle(ds_title, fontsize=14, fontweight="bold", y=1.01)
            fig.supxlabel("Fraction of Samples Receiving Full D4 TTA (%)", fontsize=12)

            stem = out_dir / f"figure4_selective_pareto_{dataset}"
            fig.savefig(str(stem) + ".png", **SAVEKW)
            fig.savefig(str(stem) + ".pdf", **SAVEKW)
            plt.close(fig)
        print(f"  Saved figure4_selective_pareto_{dataset}.png/pdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PLOT_REGISTRY = {
    "1":   ("plot1_tta_helps_all",           "df_no_mhist"),
    "1b":  ("plot1b_general_models",         "df_no_mhist"),
    "2":   ("plot2_histology_fm_delta",      "df"),
    "3":   ("plot3_frozen_vs_finetuned",     "df_no_mhist"),
    "6":   ("plot6_correction_scatter",      "df"),
    "7":   ("plot7_baseline_vs_delta",       "df"),
    "8":   ("plot8_per_class_delta",         "pc"),
    "9":   ("plot9_uncertainty",             "df_no_mhist"),
    "10":  ("plot10_agreement_rate",         "df_no_mhist"),
    "11":  ("plot11_scale_vs_delta",         "df_no_mhist"),
    "12":  ("plot12_datasize_ablation",      "abl"),
    # Publication figures
    "16":  ("plot_perclass_scatter",         "pc"),
    "17":  ("plot_selective_pareto_clean",   "selective"),
    "18":  ("plot_qualitative_examples",     "per_sample_dir"),
    "19":  ("plot_correction_rates",         "df"),
    # Combined Figure 2 — driven by the new canonical (frozen-probe) results
    "fig2": ("plot_figure2_combined",        "canonical"),
    # Figure 4: selective TTA Pareto
    "fig4": ("plot_figure4_selective_pareto", "df"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Publication-quality TTA analysis plots")
    p.add_argument("--out_dir", default="figures", help="Output directory (default: figures/)")
    p.add_argument("--results_csv", default="results/tta_results.csv")
    p.add_argument("--per_class_csv", default="results/tta_per_class.csv")
    p.add_argument("--canonical_csv", default="results/canonical_results.csv",
                   help="Unified frozen-probe results (drives Figure 2)")
    p.add_argument("--selective_dir", default="results/raw",
                   help="Directory containing selective_tta_*.json files")
    p.add_argument("--per_sample_dir", default="results/per_sample",
                   help="Directory containing per_sample_*.csv files from eval_per_sample.py")
    p.add_argument("--plot", nargs="*", default=None,
                   help=f"Plot(s) to generate (default: all). Choices: {list(PLOT_REGISTRY.keys())}")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = args.plot if args.plot else list(PLOT_REGISTRY.keys())
    needs = {PLOT_REGISTRY.get(k, (None, None))[1] for k in selected}

    print("Loading data...")
    df, pc, df_no_mhist = None, None, None
    if needs & {"df", "df_no_mhist", "pc"}:
        df, pc = load_data(args.results_csv, args.per_class_csv)
        df_no_mhist = df[df["dataset"] != "mhist"].copy()
        n_seeds = df["seed"].nunique()
        print(f"  {len(df)} summary rows, {len(pc)} per-class rows, {n_seeds} seeds")

    # Lazy-load ablation data only when needed
    abl = None
    if "abl" in needs:
        abl = load_ablation_data(args.results_csv)
        print(f"  {len(abl)} ablation rows loaded")

    # Lazy-load canonical (frozen-probe) data only when needed
    canonical = None
    if "canonical" in needs:
        canonical = load_canonical(args.canonical_csv)
        print(f"  {len(canonical)} canonical rows loaded (head=linear)")

    # Lazy-load selective TTA data only when needed
    selective = None
    if any(PLOT_REGISTRY.get(k, (None, None))[1] == "selective" for k in selected):
        selective = _load_selective_data(args.selective_dir)
        if selective.empty:
            print(f"  WARNING: no selective TTA JSON files found in '{args.selective_dir}'. "
                  "Skipping selective pareto plot.")
            selected = [k for k in selected
                        if PLOT_REGISTRY.get(k, (None, None))[1] != "selective"]
        else:
            print(f"  {len(selective)} selective TTA rows loaded")

    print("Generating plots...")
    for key in selected:
        if key not in PLOT_REGISTRY:
            print(f"  WARNING: unknown plot '{key}', skipping. Choices: {list(PLOT_REGISTRY.keys())}")
            continue
        func_name, data_key = PLOT_REGISTRY[key]
        func = globals()[func_name]
        data = {
            "df": df, "df_no_mhist": df_no_mhist, "pc": pc,
            "abl": abl, "selective": selective, "canonical": canonical,
            "per_sample_dir": args.per_sample_dir,
        }[data_key]
        func(data, out_dir)

    print(f"\nDone. All plots saved to '{out_dir}'")


if __name__ == "__main__":
    main()
