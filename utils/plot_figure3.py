#!/usr/bin/env python3
"""
Figure 3: Mechanism panels for TTA paper.

Panel (a): Diverging lollipop — correction vs corruption rates per model.
Panel (b): Per-class F1 scatter (TCGA-UT, 31 classes), pp units.
Panel (c): Agreement rate vs TTA Δ balanced accuracy (no MHIST).

Outputs (PNG + PDF):
    figure3_panel_a, figure3_panel_b,
    figure3_panel_c_v1 (single regression),
    figure3_panel_c_v2 (per-family regressions),
    figure3_journal
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.plot_analysis import (
    CB_PALETTE, PUB_RCPARAMS, SAVEKW,
    MODEL_TYPE_MAP, DATASET_ORDER, DATASET_LABELS, DS_COLORS,
    CONFIG_COLS, _compute_deltas,
)

# ---------------------------------------------------------------------------
# Figure 3–specific constants
# ---------------------------------------------------------------------------

HIST_ORDER = ["phikon", "uni2", "virchow", "phikon2", "gigapath", "uni", "hoptimus", "virchow2", "ctranspath"]
GEN_ORDER  = ["convnextv2_tiny", "dinov2_s", "dinov2_b", "convnextv2_base"]

FIG3_MODEL_NAMES = {
    "phikon":          "Phikon",
    "phikon2":         "Phikon-2",
    "uni":             "UNI",
    "uni2":            "UNI-v2",
    "virchow":         "Virchow",
    "virchow2":        "Virchow2",
    "gigapath":        "GigaPath",
    "hoptimus":        "H-Optimus-1",
    "ctranspath":      "CTransPath",
    "resnet18":        "ResNet-18",
    "resnet50":        "ResNet-50",
    "convnextv2_tiny": "ConvNeXtV2-T",
    "convnextv2_base": "ConvNeXtV2-B",
    "dinov2_s":        "DINOv2-S",
    "dinov2_b":        "DINOv2-B",
    "d4wrn":           "D4-WRN",
}

COR_COLOR     = "#009E73"  # Wong bluish-green — corrections
CORRUPT_COLOR = "#D55E00"  # Wong vermillion — corruptions

TCGA_ABBREV = {
    "Adrenocortical_carcinoma":                                         "ACC",
    "Bladder_Urothelial_Carcinoma":                                     "BLCA",
    "Brain_Lower_Grade_Glioma":                                         "LGG",
    "Breast_invasive_carcinoma":                                        "BRCA",
    "Cervical_squamous_cell_carcinoma_and_endocervical_adenocarcinoma": "CESC",
    "Cholangiocarcinoma":                                               "CHOL",
    "Colon_Rectum_adenocarcinoma":                                      "COAD",
    "Esophageal_carcinoma":                                             "ESCA",
    "Glioblastoma_multiforme":                                          "GBM",
    "Head_and_Neck_squamous_cell_carcinoma":                            "HNSC",
    "Kidney_Chromophobe":                                               "KICH",
    "Kidney_renal_clear_cell_carcinoma":                                "KIRC",
    "Kidney_renal_papillary_cell_carcinoma":                            "KIRP",
    "Liver_hepatocellular_carcinoma":                                   "LIHC",
    "Lung_adenocarcinoma":                                              "LUAD",
    "Lung_squamous_cell_carcinoma":                                     "LUSC",
    "Lymphoid_Neoplasm_Diffuse_Large_B-cell_Lymphoma":                  "DLBC",
    "Mesothelioma":                                                     "MESO",
    "Ovarian_serous_cystadenocarcinoma":                                "OV",
    "Pancreatic_adenocarcinoma":                                        "PAAD",
    "Pheochromocytoma_and_Paraganglioma":                               "PCPG",
    "Prostate_adenocarcinoma":                                          "PRAD",
    "Sarcoma":                                                          "SARC",
    "Skin_Cutaneous_Melanoma":                                          "SKCM",
    "Stomach_adenocarcinoma":                                           "STAD",
    "Testicular_Germ_Cell_Tumors":                                      "TGCT",
    "Thymoma":                                                          "THYM",
    "Thyroid_carcinoma":                                                "THCA",
    "Uterine_Carcinosarcoma":                                           "UCS",
    "Uterine_Corpus_Endometrial_Carcinoma":                             "UCEC",
    "Uveal_Melanoma":                                                   "UVM",
}

GROUP_LABELS = {
    "histology":   "Histology FMs",
    "general":     "General Models",
    "equivariant": "Equivariant",
}

_C_MARKERS       = {"frozen": "o", "finetuned": "s"}
_C_MARKER_LABELS = {"frozen": "Frozen", "finetuned": "Fine-tuned"}


# ---------------------------------------------------------------------------
# Shared regression helper
# ---------------------------------------------------------------------------

def _reg_line_ci(x, y, ax, color="gray", linestyle="--", linewidth=1.5, alpha_band=0.12):
    """Draw OLS line + 95% CI band. Returns (spearman_rho, p) or None."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.array(x)[mask], np.array(y)[mask]
    if len(x) < 3:
        return None
    slope, intercept, _, _, _ = stats.linregress(x, y)
    rho, p = stats.spearmanr(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept
    n = len(x)
    t_val = stats.t.ppf(0.975, n - 2)
    resid_std = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n - 2))
    ci = t_val * resid_std * np.sqrt(
        1 / n + (x_line - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)
    )
    ax.plot(x_line, y_line, color=color, linewidth=linewidth,
            linestyle=linestyle, alpha=0.8, zorder=1)
    ax.fill_between(x_line, y_line - ci, y_line + ci,
                    color=color, alpha=alpha_band, zorder=0)
    return rho, p


def _dim_grid(ax):
    for _gl in ax.get_xgridlines() + ax.get_ygridlines():
        _gl.set_alpha(0.15)


# ---------------------------------------------------------------------------
# Panel (a) — Diverging lollipop
# ---------------------------------------------------------------------------

def _build_panel_a_data(df: pd.DataFrame):
    """Compute per-model correction/corruption rates.

    Histology and general: frozen only. D4WRN: finetuned (no frozen variant).
    """
    # canonical has multiple head types — use linear as canonical probe
    if "head_type" in df.columns:
        df = df[df["head_type"].isin(["linear", float("nan")])].copy()
    base = df[(df["strategy"] == "none") & (df["aggregation"] == "mean")].copy()
    tta  = df[(df["strategy"] == "d4")   & (df["aggregation"] == "mean")].copy()
    tta  = tta.dropna(subset=["n_corrected", "n_corrupted"])

    # dataset test-set sizes from parquets (used as fallback for models without parquets)
    import glob as _glob
    _raw_dir = Path(__file__).parent.parent / "results" / "raw"
    _ds_sizes = {}
    for _f in _glob.glob(str(_raw_dir / "probe_persample_*_linear_seed0.parquet")):
        try:
            _ps = __import__("pandas").read_parquet(_f)
            _ds_sizes[_ps["dataset"].iloc[0]] = len(_ps)
        except Exception:
            pass

    merge_cols = CONFIG_COLS + ["seed"]
    tta = tta.merge(
        base[merge_cols + ["n_wrong", "n_correct"]].rename(
            columns={"n_wrong": "base_wrong", "n_correct": "base_correct"}),
        on=merge_cols, how="left",
    )
    tta["n_test"] = tta["base_wrong"] + tta["base_correct"]
    # for models without parquet data (e.g. d4wrn), fill n_test from known dataset sizes
    mask = tta["n_test"].isna()
    if mask.any() and _ds_sizes:
        tta.loc[mask, "n_test"] = tta.loc[mask, "dataset"].map(_ds_sizes)
    tta["correction_rate"] = tta["n_corrected"] / tta["n_test"].clip(lower=1) * 100
    tta["corruption_rate"] = tta["n_corrupted"] / tta["n_test"].clip(lower=1) * 100

    tta = tta[
        ((tta["model_type"].isin(["histology", "general"])) & (tta["backbone_mode"] == "frozen")) |
        (tta["model"] == "d4wrn")
    ].copy()

    agg_ds = tta.groupby(["model", "model_type", "dataset"], observed=True).agg(
        corr_r=("correction_rate", "mean"),
        corrupt_r=("corruption_rate", "mean"),
    ).reset_index()

    agg = tta.groupby(["model", "model_type"], observed=True).agg(
        corr_mean=("correction_rate", "mean"),
        corr_sem=("correction_rate", lambda x: x.sem() if len(x) > 1 else 0),
        corrupt_mean=("corruption_rate", "mean"),
        corrupt_sem=("corruption_rate", lambda x: x.sem() if len(x) > 1 else 0),
    ).reset_index()

    # mean bar from TCGA-UT (primary dataset, strongest TTA signal)
    agg_tcga = tta[tta["dataset"] == "tcga-ut"].groupby(["model", "model_type"], observed=True).agg(
        corr_mean=("correction_rate", "mean"),
        corr_sem=("correction_rate", lambda x: x.sem() if len(x) > 1 else 0),
        corrupt_mean=("corruption_rate", "mean"),
        corrupt_sem=("corruption_rate", lambda x: x.sem() if len(x) > 1 else 0),
    ).reset_index()
    agg_tcga["net_delta"] = agg_tcga["corr_mean"] - agg_tcga["corrupt_mean"]

    ordered = []
    for mtype, order in [
        ("general",     GEN_ORDER),
        ("histology",   HIST_ORDER),
        ("equivariant", ["d4wrn"]),
    ]:
        sub = agg_tcga[agg_tcga["model_type"] == mtype].copy()
        sub = sub.sort_values("net_delta", ascending=False)
        if not sub.empty:
            ordered.append(sub)
    agg_sorted = pd.concat(ordered, ignore_index=True)

    return agg_sorted, agg_ds


def _draw_panel_a(ax, df: pd.DataFrame):
    """Draw diverging lollipop into ax. Returns legend handles."""
    agg_sorted, agg_ds = _build_panel_a_data(df)

    # derive group order and sizes directly from agg_sorted to match whatever ordering was used
    present_types, type_sizes = [], []
    for mtype in agg_sorted["model_type"].unique():
        present_types.append(mtype)
        type_sizes.append(int((agg_sorted["model_type"] == mtype).sum()))

    GROUP_GAP = 0.0
    y_pos_bottom_up = []
    y = 0.0
    for ts in type_sizes:
        y_pos_bottom_up.extend([y + i for i in range(ts)])
        y += ts + GROUP_GAP
    y_max = max(y_pos_bottom_up)
    y_pos = [y_max - yy for yy in y_pos_bottom_up]

    xlim_val = min(
        max(
            agg_sorted["corr_mean"].max() + agg_sorted["corr_sem"].max() + 1,
            agg_sorted["corrupt_mean"].max() + agg_sorted["corrupt_sem"].max() + 1,
        ) * 1.15,
        10,
    )

    from matplotlib.transforms import blended_transform_factory as _btf
    trans_margin = _btf(ax.transAxes, ax.transData)

    DS_MARKERS_A = {
        "tcga-ut":        "o",
        "nct-crc-100k":   "s",
        "nct-crc-nonorm": "^",
        "mhist":          "D",
    }

    for idx, (_, row) in enumerate(agg_sorted.iterrows()):
        yp = y_pos[idx]
        model = row["model"]
        net = row["corr_mean"] - row["corrupt_mean"]

        tc = DS_COLORS["tcga-ut"]
        # thin base stems
        ax.plot([0, row["corr_mean"]],     [yp, yp], color=COR_COLOR,     lw=1.5, alpha=0.4, zorder=2)
        ax.plot([0, -row["corrupt_mean"]], [yp, yp], color=CORRUPT_COLOR, lw=1.5, alpha=0.4, zorder=2)
        # bold net-delta segment on the winning side
        if net >= 0:
            ax.plot([row["corrupt_mean"], row["corr_mean"]], [yp, yp],
                    color=tc, lw=4, solid_capstyle="butt", zorder=3)
            if idx == 0:
                brace_y  = yp + 0.52
                tick_h   = 0.18
                mid_x    = (row["corrupt_mean"] + row["corr_mean"]) / 2
                gap = 1.4
                ax.plot([row["corrupt_mean"], mid_x - gap], [brace_y, brace_y],
                        color="black", lw=1.2, zorder=5)
                ax.plot([mid_x + gap, row["corr_mean"]], [brace_y, brace_y],
                        color="black", lw=1.2, zorder=5)
                ax.plot([row["corrupt_mean"], row["corrupt_mean"]],
                        [brace_y - tick_h, brace_y], color="black", lw=1.2, zorder=5)
                ax.plot([row["corr_mean"], row["corr_mean"]],
                        [brace_y - tick_h, brace_y], color="black", lw=1.2, zorder=5)
                ax.text(mid_x, brace_y, "TTA Δ\nTCGA-UT",
                        ha="center", va="center", fontsize=8, color="black",
                        linespacing=1.3)
        else:
            ax.plot([-row["corr_mean"], -row["corrupt_mean"]], [yp, yp],
                    color=CORRUPT_COLOR, lw=4, solid_capstyle="butt", zorder=3)
        # endpoint dots
        ax.plot(row["corr_mean"],     yp, "o", color=tc, ms=7, zorder=4)
        ax.plot(-row["corrupt_mean"], yp, "o", color=tc, ms=7, zorder=4)

        # faint net-delta segment only (like the thick TCGA-UT bar) for other datasets
        ds_sub = agg_ds[(agg_ds["model"] == model) & (~agg_ds["dataset"].isin(["tcga-ut"]))]
        offsets = np.linspace(-0.15, -0.15 - 0.18 * (max(len(ds_sub), 1) - 1), max(len(ds_sub), 1))
        for j, (_, ds_row) in enumerate(ds_sub.iterrows()):
            yj = yp + offsets[j]
            cr, co = ds_row["corr_r"], ds_row["corrupt_r"]
            dc = DS_COLORS.get(ds_row["dataset"], "#888888")
            ds_net = cr - co
            if ds_net >= 0:
                ax.plot([co, cr], [yj, yj], color=dc, lw=3, alpha=0.4, solid_capstyle="butt", zorder=3)
            else:
                ax.plot([-cr, -co], [yj, yj], color=dc, lw=3, alpha=0.4, solid_capstyle="butt", zorder=3)


        ratio = row["corr_mean"] / max(row["corrupt_mean"], 0.01)
        ax.text(1.02, yp, f"{ratio:.1f}×", va="center", ha="left",
                fontsize=9, color="#333333", transform=trans_margin, clip_on=False)

    ax.text(1.02, 1.01, "Correct./\nCorrupt.", va="bottom", ha="left",
            fontsize=9, color="#333333", fontweight="bold",
            transform=ax.transAxes, clip_on=False)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [FIG3_MODEL_NAMES.get(r["model"], r["model"]) for _, r in agg_sorted.iterrows()],
        fontsize=11,
    )

    ax.axvline(0, color="black", lw=1.0, zorder=1)
    ax.set_xlabel("% of test-set predictions flipped by D4 TTA", fontsize=13)
    left_lim = -(agg_sorted["corrupt_mean"].max() + agg_sorted["corrupt_sem"].max() + 1) * 1.15
    ax.set_xlim(left_lim, xlim_val)
    ax.set_ylim(min(y_pos) - 0.5, max(y_pos) + 0.8)

    # Group separator lines and labels
    GROUP_COLORS = {"general": "#DD8452", "histology": "#4C72B0", "equivariant": "#55A868"}
    from matplotlib.transforms import blended_transform_factory as _btf2
    trans_left = _btf2(ax.transAxes, ax.transData)
    idx = 0
    for gi, (mtype, ts) in enumerate(zip(present_types, type_sizes)):
        group_ypos = y_pos[idx:idx + ts]
        mid_y = (group_ypos[0] + group_ypos[-1]) / 2
        color = GROUP_COLORS.get(mtype, "#888888")
        # horizontal rule above each group except the first, with inline label
        if gi > 0:
            rule_y = (y_pos[idx - 1] + group_ypos[0]) / 2
            ax.axhline(rule_y, color="#bbbbbb", lw=0.9, ls="--", zorder=0)
            ax.text(0.82, rule_y, f"  {GROUP_LABELS.get(mtype, mtype)}  ",
                    transform=ax.get_yaxis_transform(), fontsize=8, color=color,
                    ha="center", va="center", clip_on=False,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"))
        idx += ts

    tick_step = max(1, round(xlim_val / 5))
    ticks = np.arange(0, xlim_val + tick_step, tick_step)
    xtick_vals = np.concatenate([-ticks[1:][::-1], ticks])
    xtick_vals = xtick_vals[xtick_vals >= left_lim]
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels(
        [f"−{abs(t):.0f}" if t < 0 else f"{t:.0f}" for t in xtick_vals],
        fontsize=12,
    )

    y_top = max(y_pos) + 0.8
    tc = DS_COLORS["tcga-ut"]
    ax.text(-xlim_val * 0.05, y_top, "← Corruption rate",
            ha="right", va="bottom", fontsize=10, color=CORRUPT_COLOR, style="italic")
    ax.text( xlim_val * 0.05, y_top, "Correction rate →",
            ha="left",  va="bottom", fontsize=10, color=COR_COLOR, style="italic")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _dim_grid(ax)

    other_datasets = [ds for ds in DATASET_ORDER if ds != "tcga-ut" and ds in agg_ds["dataset"].values]
    def _short_label(ds):
        return DATASET_LABELS.get(ds, ds).split(" (")[0]
    legend_elements = [
        Line2D([0], [0], color=tc, lw=4, label="TCGA-UT"),
    ] + [
        Line2D([0], [0], color=DS_COLORS.get(ds, "#888"), lw=3, alpha=0.8,
               label=_short_label(ds))
        for ds in other_datasets
    ]
    return legend_elements


def make_panel_a(df: pd.DataFrame, out_dir: Path):
    n_models = len(HIST_ORDER) + len(GEN_ORDER) + 1
    with plt.rc_context(PUB_RCPARAMS):
        fig, ax = plt.subplots(figsize=(11, max(5.5, n_models * 0.42 + 1.8)))
        handles = _draw_panel_a(ax, df)
        ax.legend(handles=handles, fontsize=9, loc="lower right",
                  framealpha=0.85, ncol=4)
        fig.tight_layout()
        stem = out_dir / "figure3_panel_a"
        fig.savefig(str(stem) + ".png", **SAVEKW)
        fig.savefig(str(stem) + ".pdf", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print("  Saved figure3_panel_a.png/.pdf")


# ---------------------------------------------------------------------------
# Panel (b) — Entropy distribution: corrected vs corrupted samples
# ---------------------------------------------------------------------------

_FM_HIST = {'phikon','phikon2','uni','uni2','virchow','virchow2','gigapath','hoptimus','ctranspath'}

_PANEL_B_FAMILY = {
    "histology": ("Histology FMs", "--"),
    "general":   ("General Models",  "-"),
}


_PANEL_B_DATASETS = ["tcga-ut", "nct-crc-100k", "nct-crc-nonorm"]
_PANEL_B_N_BINS   = 12
_PANEL_B_XMAX     = 0.30


def _build_panel_b_data():
    """Pool all three datasets. Returns (centers, rates, hists) split by model family.

    Entropy is normalized by log(K) so datasets are on the same scale.
    rates : {'histology': {'correction': arr, 'corruption': arr}, 'general': {...}}
    hists : {'histology': hist_pct (sums to 100), 'general': hist_pct (sums to 100)}
    """
    import glob
    raw_dir = Path(__file__).parent.parent / "results" / "raw"

    bins    = np.linspace(0, _PANEL_B_XMAX, _PANEL_B_N_BINS + 1)
    centers = (bins[:-1] + bins[1:]) / 2

    corr_by      = {"histology": [], "general": []}
    corrup_by    = {"histology": [], "general": []}
    unchanged_by = {"histology": [], "general": []}

    for f in sorted(glob.glob(str(raw_dir / "probe_persample_*.parquet"))):
        ps = pd.read_parquet(f)
        if "head_type" not in ps.columns or ps["head_type"].iloc[0] != "linear":
            continue
        if ps["dataset"].iloc[0] != "tcga-ut":
            continue
        mtype = "histology" if ps["model"].iloc[0] in _FM_HIST else "general"

        n_classes = ps["true_label"].nunique()
        ps = ps.copy()
        ps["entropy_0"] = ps["entropy_0"] / np.log(n_classes)
        ps["bin_idx"]   = pd.cut(ps["entropy_0"], bins=bins, labels=False, include_lowest=True)
        ps["corrected"] = (~ps["correct_0"]) & ps["correct_d4"]
        ps["corrupted"] = ps["correct_0"]    & (~ps["correct_d4"])
        g = ps.groupby("bin_idx", observed=False)
        n_test = len(ps)

        def _pct(series_or_size):
            return series_or_size.reindex(range(_PANEL_B_N_BINS), fill_value=0).values / n_test * 100

        # everything is a % of this model's *whole test set* (not the per-bin
        # rate), so the curves show where flips actually concentrate (they peak
        # and fall) and grey + corrected + corrupted partition the test set.
        all_pct    = _pct(g.size())
        corr_pct   = _pct(g["corrected"].sum())
        corrup_pct = _pct(g["corrupted"].sum())
        corr_by[mtype].append(corr_pct)
        corrup_by[mtype].append(corrup_pct)
        unchanged_by[mtype].append(all_pct - corr_pct - corrup_pct)

    rates, hists = {}, {}
    for mtype in ("histology", "general"):
        rates[mtype] = {
            "correction": np.mean(corr_by[mtype],   axis=0) if corr_by[mtype]   else np.zeros(_PANEL_B_N_BINS),
            "corruption": np.mean(corrup_by[mtype], axis=0) if corrup_by[mtype] else np.zeros(_PANEL_B_N_BINS),
        }
        hists[mtype] = (np.mean(unchanged_by[mtype], axis=0)
                        if unchanged_by[mtype] else np.zeros(_PANEL_B_N_BINS))

    return centers, rates, hists


def _draw_panel_b_single(ax, centers, corr_rate, corrup_rate, hist_pct, title,
                         show_ylabel=True, show_legend=True):
    """Draw one panel-b subplot for a single model family."""
    tc    = DS_COLORS["tcga-ut"]
    width = centers[1] - centers[0]

    ax2 = ax.twinx()
    ax2.bar(centers, hist_pct, width=width * 0.85, color="#aaaaaa", alpha=0.28, zorder=1)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("% of samples", fontsize=8, color="#999999")
    ax2.tick_params(axis="y", labelcolor="#999999", labelsize=7)
    ax2.spines["right"].set_color("#dddddd")
    ax2.grid(False)

    ax.plot(centers, corr_rate,   color=COR_COLOR,    lw=2.0, zorder=3, label="Corrected")
    ax.plot(centers, corrup_rate, color=CORRUPT_COLOR, lw=2.0, zorder=3, label="Corrupted")

    ax.set_xlabel("Normalized prediction entropy", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("% of bin flipped by D4", fontsize=9)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xlim(0, 0.27)
    ax.set_ylim(bottom=0)
    if show_legend:
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.90)
    _dim_grid(ax)


def _draw_panel_b_ds(ax, centers, rates, hists, ds, show_ylabel=True, show_legend=True):
    """Draw correction/corruption vs normalized entropy for one dataset."""
    color = DS_COLORS.get(ds, "#555555")
    width = centers[1] - centers[0]

    corr_rate   = np.nanmean([rates["histology"]["correction"], rates["general"]["correction"]], axis=0)
    corrup_rate = np.nanmean([rates["histology"]["corruption"], rates["general"]["corruption"]], axis=0)
    hist_pct    = np.nanmean([hists["histology"], hists["general"]], axis=0)

    ax2 = ax.twinx()
    ax2.bar(centers, hist_pct, width=width * 0.85, color="#aaaaaa", alpha=0.28, zorder=1)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("% of samples", fontsize=9, color="#999999")
    ax2.tick_params(axis="y", labelcolor="#999999", labelsize=8)
    ax2.spines["right"].set_color("#dddddd")
    ax2.grid(False)

    ax.fill_between(centers, corr_rate, corrup_rate,
                    where=corr_rate >= corrup_rate, color=COR_COLOR, alpha=0.12, zorder=2)
    ax.fill_between(centers, corr_rate, corrup_rate,
                    where=corr_rate < corrup_rate, color=CORRUPT_COLOR, alpha=0.12, zorder=2)
    ax.plot(centers, corr_rate,   color=COR_COLOR,    lw=2.0, zorder=3, label="Corrected")
    ax.plot(centers, corrup_rate, color=CORRUPT_COLOR, lw=2.0, zorder=3, label="Corrupted")

    ax.set_xlabel("Normalized prediction entropy", fontsize=10)
    if show_ylabel:
        ax.set_ylabel("% of bin flipped by D4", fontsize=10)
    ax.set_title(_PANEL_B_TITLES.get(ds, ds), fontsize=10, pad=4)
    ax.set_xlim(0, _PANEL_B_XMAX)
    ax.set_ylim(bottom=0)
    if show_legend:
        ax.legend(fontsize=8, loc="upper left", framealpha=0.90)
    _dim_grid(ax)


def _draw_panel_b(ax, _unused=None):
    """Population-weighted flip contribution vs normalized entropy (TCGA-UT),
    split by model family. Shows what fraction of the whole test set is
    corrected / corrupted in each entropy bin: the gains concentrate in the
    high-entropy tail even though most samples are low-entropy (grey histogram),
    and general models (solid) drive far more of the high-entropy flips than
    histology FMs (dashed)."""
    centers, rates, hists = _build_panel_b_data()
    tc    = DS_COLORS["tcga-ut"]
    width = centers[1] - centers[0]

    # unchanged-prediction backdrop (all - corrected - corrupted), one soft
    # filled area per family (same dashed/solid convention as the curves):
    # histology FMs pile up near zero entropy, general models' mass extends to
    # higher entropy. Grey + corrected + corrupted partition the test set.
    ax2 = ax.twinx()
    _hist_edge = {"histology": dict(ls=(0, (3.5, 2.5)), lw=1.6, alpha=0.8),
                  "general":   dict(ls="-",             lw=1.1, alpha=0.5)}
    for fam, (_label, _ls) in _PANEL_B_FAMILY.items():
        ax2.fill_between(centers, hists[fam], step="mid",
                         color="#9a9a9a", alpha=0.16, lw=0, zorder=1)
        ax2.step(centers, hists[fam], where="mid", color="#777777",
                 zorder=1, **_hist_edge[fam])
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_ylabel("Unchanged (% of test set)", fontsize=11, color="#999999")
    ax2.tick_params(axis="y", labelcolor="#999999", labelsize=10)
    ax2.spines["right"].set_color("#dddddd")
    ax2.grid(False)

    for fam, (_label, ls) in _PANEL_B_FAMILY.items():
        corr   = rates[fam]["correction"]
        corrup = rates[fam]["corruption"]
        if fam == "general":          # shade the prominent net-gain region
            ax.fill_between(centers, corr, corrup, where=corr >= corrup,
                            color=COR_COLOR, alpha=0.15, zorder=2)
        ax.plot(centers, corr,   color=COR_COLOR,     lw=2.2, ls=ls, zorder=3)
        ax.plot(centers, corrup, color=CORRUPT_COLOR, lw=2.2, ls=ls, zorder=3)

    ax.set_xlabel("Prediction entropy / log(C)", fontsize=13)
    ax.set_ylabel("Flipped (% of test set)", fontsize=13)
    ax.set_xlim(0, _PANEL_B_XMAX)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3])
    ax.set_ylim(bottom=0)

    # legend: colour encodes outcome, linestyle encodes family
    handles = [
        Line2D([0], [0], color=COR_COLOR,     lw=2.2, label="Corrected"),
        Line2D([0], [0], color=CORRUPT_COLOR, lw=2.2, label="Corrupted"),
        Line2D([0], [0], color="#555555", lw=1.8, ls=(0, (3, 2)), label="Histology FMs"),
        Line2D([0], [0], color="#555555", lw=1.8, ls="-",         label="General models"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="upper left",
              framealpha=0.9, ncol=2, columnspacing=1.0, handlelength=3.0)
    ax.text(0.98, 0.98, "TCGA-UT", transform=ax.transAxes,
            fontsize=9, va="top", ha="right", color="#555555",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    _dim_grid(ax)


def make_panel_b(pc: pd.DataFrame, out_dir: Path):
    with plt.rc_context(PUB_RCPARAMS):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        _draw_panel_b(ax)
        fig.tight_layout()
        stem = out_dir / "figure3_panel_b"
        fig.savefig(str(stem) + ".png", **SAVEKW)
        fig.savefig(str(stem) + ".pdf", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print("  Saved figure3_panel_b.png/.pdf")


# ---------------------------------------------------------------------------
# Panel (c) — Agreement rate vs TTA Δ balanced accuracy
# ---------------------------------------------------------------------------

def _build_panel_c_data(df: pd.DataFrame = None) -> pd.DataFrame:
    """Per-(model × dataset) mean baseline entropy vs TTA Δ balanced accuracy.

    Built directly from per-sample parquets (frozen linear probe, MHIST excluded, entropy normalized by log(K)).
    """
    import glob
    raw_dir = Path(__file__).parent.parent / "results" / "raw"
    FM_HIST = {'phikon','phikon2','uni','uni2','virchow','virchow2','gigapath','hoptimus','ctranspath'}
    from sklearn.metrics import balanced_accuracy_score as _bacc
    rows = []
    for f in sorted(glob.glob(str(raw_dir / "probe_persample_*.parquet"))):
        ps = pd.read_parquet(f)
        if "head_type" not in ps.columns or ps["head_type"].iloc[0] != "linear":
            continue
        model   = ps["model"].iloc[0]
        dataset = ps["dataset"].iloc[0]
        seed    = int(ps["seed"].iloc[0])
        if dataset == "mhist":
            continue
        mtype = "histology" if model in FM_HIST else "general"
        base  = _bacc(ps["true_label"], ps["pred_0"])
        d4    = _bacc(ps["true_label"], ps["pred_d4"])
        n_classes = ps["true_label"].nunique()
        rows.append(dict(
            model=model, dataset=dataset, seed=seed,
            model_type=mtype, backbone_mode="frozen",
            mean_entropy=float(ps["entropy_0"].mean() / np.log(n_classes)),
            delta_pp=float((d4 - base) * 100),
        ))
    raw = pd.DataFrame(rows)
    agg = raw.groupby(["model", "dataset", "model_type", "backbone_mode"], observed=True).agg(
        entropy_mean=("mean_entropy", "mean"),
        delta_pp_mean=("delta_pp", "mean"),
        delta_pp_sem=("delta_pp", lambda x: x.sem() if len(x) > 1 else 0),
    ).reset_index()
    return agg


def _draw_panel_c(ax, df: pd.DataFrame = None, regression: str = "single",
                  show_protocol: bool = True):
    """Draw baseline entropy vs TTA delta into ax.

    regression: "single" — one overall gray line;
                "family" — gray background line + per-family colored lines.
    Returns legend handles.
    """
    agg = _build_panel_c_data(df)
    type_handles = []

    # dataset -> (marker, short label, line color) for regression == "dataset"
    DS_STYLE = {
        "tcga-ut":         ("o", "TCGA-UT",    "#2d2d2d"),
        "nct-crc-100k":    ("s", "NCT-CRC",    "#777777"),
        "nct-crc-nonorm":  ("^", "NCT-NoNorm", "#aaaaaa"),
    }
    by_dataset = regression == "dataset"
    if regression == "tcga":          # primary benchmark only, as in panel (b)
        agg = agg[agg["dataset"] == "tcga-ut"]

    for mtype in ["histology", "general"]:
        sub = agg[agg["model_type"] == mtype]
        if sub.empty:
            continue
        color = CB_PALETTE[mtype]
        if by_dataset:
            for ds, (mk, _, _) in DS_STYLE.items():
                s2 = sub[sub["dataset"] == ds]
                if s2.empty:
                    continue
                ax.scatter(s2["entropy_mean"], s2["delta_pp_mean"], marker=mk,
                           s=50, alpha=0.85, color=color, zorder=3,
                           edgecolors="white", linewidths=0.8)
        else:
            ax.scatter(sub["entropy_mean"], sub["delta_pp_mean"],
                       s=50, alpha=0.85, color=color, zorder=3,
                       edgecolors="white", linewidths=0.8)
        type_handles.append(
            Line2D([0], [0], marker="o", color=color, markerfacecolor=color,
                   markeredgecolor="white", markeredgewidth=0.8,
                   markersize=7, linestyle="None",
                   label=GROUP_LABELS.get(mtype, mtype.capitalize()))
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)

    if by_dataset:
        # One fit per dataset: pooling the three attenuates the association,
        # so the within-dataset fits are the ones the caption reports.
        # No CI bands here: three overlapping bands wash out the panel. The
        # rho values ride in the legend, so the fits stay unlabelled inline.
        for ds, (mk, lab, lc) in DS_STYLE.items():
            sub = agg[agg["dataset"] == ds]
            if len(sub) < 3:
                continue
            x = sub["entropy_mean"].values
            y = sub["delta_pp_mean"].values
            out = _reg_line_ci(x, y, ax, color=lc, linestyle="--",
                               linewidth=1.4, alpha_band=0.0)
            rho = out[0] if out else float("nan")
            type_handles.append(
                Line2D([0], [0], marker=mk, color=lc, markerfacecolor=lc,
                       markeredgecolor="white", markeredgewidth=0.8,
                       markersize=6, linestyle="--", linewidth=1.4,
                       label=f"{lab}  ρ = {rho:.2f}")
            )
    elif regression in ("single", "tcga"):
        out = _reg_line_ci(agg["entropy_mean"].values, agg["delta_pp_mean"].values,
                           ax, color="gray")
        if out:
            rho, _ = out
            ax.text(0.03, 0.97, f"ρ = {rho:.2f}",
                    transform=ax.transAxes, fontsize=10, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))
    else:
        _reg_line_ci(agg["entropy_mean"].values, agg["delta_pp_mean"].values,
                     ax, color="gray", linestyle=":", linewidth=1.0, alpha_band=0.06)
        for mtype in ["histology", "general"]:
            sub = agg[agg["model_type"] == mtype]
            if len(sub) < 3:
                continue
            out = _reg_line_ci(sub["entropy_mean"].values, sub["delta_pp_mean"].values,
                               ax, color=CB_PALETTE[mtype], linestyle="--",
                               linewidth=1.5, alpha_band=0.10)
            if out:
                rho, _ = out
                xmed = float(sub["entropy_mean"].median())
                ymed = float(sub["delta_pp_mean"].median())
                ax.text(xmed, ymed, f"ρ = {rho:.2f}", fontsize=8,
                        color=CB_PALETTE[mtype], ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6))

    ax.set_xlabel("Prediction entropy / log(C)", fontsize=13)
    ax.set_ylabel("TTA Δ Bal. Accuracy (pp)", fontsize=13)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if not by_dataset:   # in dataset mode the legend already names the datasets
        _foot = "TCGA-UT" if regression == "tcga" else "TCGA-UT · NCT-CRC · NCT-NoNorm"
        ax.text(0.02, 0.02, _foot, transform=ax.transAxes,
                fontsize=9, va="bottom", ha="left", color="#555555",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    _dim_grid(ax)

    return type_handles


def make_panel_c(df: pd.DataFrame, out_dir: Path):
    with plt.rc_context(PUB_RCPARAMS):
        for version, reg in [("v1", "single"), ("v2", "family"), ("v3", "dataset"), ("v4", "tcga")]:
            fig, ax = plt.subplots(figsize=(6, 5))
            handles = _draw_panel_c(ax, df, regression=reg)
            ax.legend(handles=handles, fontsize=9, loc="lower right", framealpha=0.85)
            fig.tight_layout()
            stem = out_dir / f"figure3_panel_c_{version}"
            fig.savefig(str(stem) + ".png", **SAVEKW)
            fig.savefig(str(stem) + ".pdf", dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
    print("  Saved figure3_panel_c_v1.png/.pdf and figure3_panel_c_v2.png/.pdf")


# ---------------------------------------------------------------------------
# Journal layout — (a) left, (b)+(c) stacked right
# ---------------------------------------------------------------------------

def make_figure3_journal(df: pd.DataFrame, pc: pd.DataFrame, out_dir: Path,
                         fig_width: float = 10.5):
    """Two-column journal layout: panel (a) left, panels (b)+(c) stacked right."""
    n_models = len(HIST_ORDER) + len(GEN_ORDER) + 1
    fig_h = max(7.8, n_models * 0.40 + 2.5)

    with plt.rc_context(PUB_RCPARAMS):
        fig = plt.figure(figsize=(fig_width, fig_h))
        gs = fig.add_gridspec(
            1, 2,
            width_ratios=[1.2, 1.0],
            wspace=0.38,
            top=0.97, bottom=0.06, left=0.09, right=0.99,
        )

        ax_a = fig.add_subplot(gs[0])
        handles_a = _draw_panel_a(ax_a, df)
        ax_a.legend(handles=handles_a, fontsize=11,
                    loc="upper center", bbox_to_anchor=(0.5, -0.11),
                    ncol=4, framealpha=0.90)
        ax_a.text(-0.13, 1.02, "(a)", transform=ax_a.transAxes,
                  fontsize=12, fontweight="bold", va="bottom")

        gs_bc = gs[1].subgridspec(2, 1, hspace=0.42)

        ax_b = fig.add_subplot(gs_bc[0, 0])
        ax_b.patch.set_visible(False)
        _draw_panel_b(ax_b)
        ax_b.text(-0.18, 1.05, "(b)", transform=ax_b.transAxes,
                  fontsize=12, fontweight="bold", va="bottom")

        ax_c = fig.add_subplot(gs_bc[1, 0])
        ax_c.patch.set_visible(False)
        handles_c = _draw_panel_c(ax_c, df, regression="single", show_protocol=False)
        ax_c.legend(handles=handles_c, fontsize=8, loc="lower right", framealpha=0.85)
        ax_c.text(-0.12, 1.05, "(c)", transform=ax_c.transAxes,
                  fontsize=12, fontweight="bold", va="bottom")

        stem = out_dir / "figure3_journal"
        fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(str(stem) + ".pdf", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"  Saved figure3_journal.png/.pdf  ({fig_width}\" × {fig_h:.1f}\")")


# ---------------------------------------------------------------------------
# Data loading & CLI
# ---------------------------------------------------------------------------

def _augment_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Add n_correct/n_wrong/n_test/agreement_rate columns to canonical_results.csv
    by reading per-sample parquets from results/raw/."""
    import glob
    raw_dir = Path(__file__).parent.parent / "results" / "raw"
    records = []
    for f in sorted(glob.glob(str(raw_dir / "probe_persample_*.parquet"))):
        ps = pd.read_parquet(f)
        if "head_type" not in ps.columns:
            continue
        model   = ps["model"].iloc[0]
        dataset = ps["dataset"].iloc[0]
        seed    = int(ps["seed"].iloc[0])
        ht      = ps["head_type"].iloc[0]
        n_test  = len(ps)
        n_correct = int((ps["pred_0"] == ps["true_label"]).sum())
        n_wrong   = n_test - n_correct
        # agreement rate: fraction of samples where all 8 D4 views agree
        view_cols = [c for c in ps.columns if c.startswith("pred_view_")]
        if view_cols:
            agree = (ps[view_cols].nunique(axis=1) == 1).mean()
        else:
            agree = float("nan")
        records.append(dict(model=model, dataset=dataset, seed=seed,
                            head_type=ht, n_test=n_test,
                            n_correct=n_correct, n_wrong=n_wrong,
                            agreement_rate=agree))
    if not records:
        df["n_correct"] = float("nan")
        df["n_wrong"]   = float("nan")
        df["n_test"]    = float("nan")
        df["agreement_rate"] = float("nan")
        return df
    aux = pd.DataFrame(records)
    return df.merge(aux, on=["model", "dataset", "seed", "head_type"], how="left")


def _compute_per_class_from_parquets() -> pd.DataFrame:
    """Compute per-class F1 for frozen linear probe from per-sample parquets."""
    import glob
    from sklearn.metrics import f1_score
    raw_dir = Path(__file__).parent.parent / "results" / "raw"
    FM_HIST = {'phikon','phikon2','uni','uni2','virchow','virchow2','gigapath','hoptimus','ctranspath'}
    rows = []
    for f in sorted(glob.glob(str(raw_dir / "probe_persample_*.parquet"))):
        ps = pd.read_parquet(f)
        if "head_type" not in ps.columns or ps["head_type"].iloc[0] != "linear":
            continue
        model   = ps["model"].iloc[0]
        dataset = ps["dataset"].iloc[0]
        seed    = int(ps["seed"].iloc[0])
        mtype   = "histology" if model in FM_HIST else "general"
        y_true  = ps["true_label"].values
        classes = sorted(np.unique(y_true))
        for strategy, pred_col in [("none", "pred_0"), ("d4", "pred_d4")]:
            if pred_col not in ps.columns:
                continue
            y_pred = ps[pred_col].values
            f1s = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
            for cls, f1v in zip(classes, f1s):
                support = int((y_true == cls).sum())
                rows.append(dict(
                    model=model, dataset=dataset, backbone_mode="frozen",
                    train_augment=True, seed=seed, model_type=mtype,
                    strategy=strategy, aggregation="mean",
                    train_subset=float("nan"),
                    class_name=str(cls), f1=float(f1v), support=support,
                ))
    return pd.DataFrame(rows)


def load_data(results_csv: str = "results/canonical_results.csv",
              per_class_csv: str = "results/tta_per_class.csv"):
    df = pd.read_csv(results_csv)
    pc_old = pd.read_csv(per_class_csv)

    # canonical_results.csv schema — add bridge columns expected by figure3
    if "train_augment" not in df.columns:
        df["train_augment"] = True
    if "train_subset" not in df.columns:
        df["train_subset"] = float("nan")

    df = df[(df["train_augment"] == True) & (df["train_subset"].isna())].copy()

    # derive n_correct / n_wrong / agreement_rate from per-sample parquets
    if "n_correct" not in df.columns:
        df = _augment_canonical(df)

    # use frozen per-class data from parquets if available; fall back to old CSV
    pc_new = _compute_per_class_from_parquets()
    if not pc_new.empty:
        pc = pc_new
    else:
        pc = pc_old[(pc_old["train_augment"] == True)].copy()

    if "model_type" not in df.columns or df["model_type"].isna().any():
        df["model_type"] = df["model"].map(MODEL_TYPE_MAP)
    if "model_type" not in pc.columns or pc["model_type"].isna().any():
        pc["model_type"] = pc["model"].map(MODEL_TYPE_MAP)
    return df, pc


def parse_args():
    p = argparse.ArgumentParser(description="Figure 3: mechanism panels for TTA paper")
    p.add_argument("--out_dir", default="figures")
    p.add_argument("--results_csv", default="results/canonical_results.csv")
    p.add_argument("--per_class_csv", default="results/tta_per_class.csv")
    p.add_argument("--panel", choices=["a", "b", "c", "journal", "all"], default="all")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, pc = load_data(args.results_csv, args.per_class_csv)

    panels = ["journal"] if args.panel == "all" else [args.panel]
    for panel in panels:
        if panel == "a":
            print("Panel (a): correction/corruption rates...")
            make_panel_a(df, out_dir)
        elif panel == "b":
            print("Panel (b): per-class F1 scatter...")
            make_panel_b(pc, out_dir)
        elif panel == "c":
            print("Panel (c): agreement rate scatter (v1 + v2)...")
            make_panel_c(df, out_dir)
        elif panel == "journal":
            print("Journal layout (a left, b+c stacked right)...")
            make_figure3_journal(df, pc, out_dir)


if __name__ == "__main__":
    main()
