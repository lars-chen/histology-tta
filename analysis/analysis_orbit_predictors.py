"""
Horse race: which orbit / entropy metrics best predict TTA delta?

Inputs:
  results/orbit_tightness_v2.csv   — OCN, MPCS, within/between class MPCS,
                                     separability_ratio (from cached embeddings)
  results/canonical_results.csv    — balanced_acc per model×dataset×strategy×seed
  results/agreement_rate_summary.csv — entropy, agreement_rate per model×dataset

Outputs:
  results/orbit_predictor_correlations.csv
  figures/orbit_predictor_scatter.png   (top predictors vs TTA delta)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
ORBIT_V2  = ROOT / "results" / "orbit_tightness_v2.csv"
CANON     = ROOT / "results" / "canonical_results.csv"
AGREE     = ROOT / "results" / "agreement_rate_summary.csv"
OUT_CORR  = ROOT / "results" / "orbit_predictor_correlations.csv"
OUT_FIG   = ROOT / "figures" / "orbit_predictor_scatter.png"

FM_MODELS = {
    "phikon", "phikon2", "uni", "uni2",
    "virchow", "virchow2", "gigapath", "hoptimus", "ctranspath",
}

DS_COLORS = {
    "tcga-ut":        "#6B9EC7",
    "nct-crc-100k":   "#8B5E3C",
    "nct-crc-nonorm": "#B5785A",
    "mhist":          "#4CAF50",
}
MT_COLORS = {"histology_fm": "#0072B2", "general": "#D4772A"}
DS_MARKERS = {"tcga-ut": "o", "nct-crc-100k": "s", "nct-crc-nonorm": "^", "mhist": "D"}


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def load_tta_delta() -> pd.DataFrame:
    """Return mean TTA delta (d4 - none) per (model, dataset) for linear head."""
    canon = pd.read_csv(CANON)
    lin = canon[canon["head_type"] == "linear"]
    none_acc = (
        lin[lin["strategy"] == "none"]
        [["model", "dataset", "seed", "balanced_acc"]]
        .rename(columns={"balanced_acc": "base_acc"})
    )
    d4_acc = (
        lin[(lin["strategy"] == "d4") & (lin["aggregation"] == "mean")]
        [["model", "dataset", "seed", "balanced_acc"]]
    )
    delta = d4_acc.merge(none_acc, on=["model", "dataset", "seed"])
    delta["delta"] = delta["balanced_acc"] - delta["base_acc"]
    return delta.groupby(["model", "dataset"])["delta"].mean().reset_index()


def load_merged() -> pd.DataFrame:
    """Merge orbit v2, TTA delta, and entropy/agreement into one table."""
    orbit  = pd.read_csv(ORBIT_V2)
    delta  = load_tta_delta()
    agree  = pd.read_csv(AGREE) if AGREE.exists() else None

    # keep only core orbit columns (drop ocn__{class} columns)
    orbit_cols = [c for c in orbit.columns if not c.startswith("ocn__")]
    orbit = orbit[orbit_cols]

    merged = orbit.merge(delta, on=["model", "dataset"], how="inner")
    if agree is not None:
        agree_cols = ["model", "dataset", "mean_entropy_0_norm", "agree_rate_base_d4"]
        agree_sub  = agree[[c for c in agree_cols if c in agree.columns]]
        merged = merged.merge(agree_sub, on=["model", "dataset"], how="left")

    merged["model_type"] = merged["model"].apply(
        lambda m: "histology_fm" if m in FM_MODELS else "general"
    )
    return merged


# ---------------------------------------------------------------------------
# Correlation horse race
# ---------------------------------------------------------------------------

CANDIDATE_METRICS = [
    ("mean_ocn",               "OCN (orbit centroid norm)",             False),
    ("mean_mpcs",              "MPCS (mean pairwise cos-sim, orbit)",   False),
    ("mean_within_class_mpcs", "Within-class MPCS",                     False),
    ("mean_between_class_mpcs","Between-class MPCS",                    True),
    ("separability_ratio",     "Separability ratio (within/between)",   True),
    ("mean_entropy_0_norm",    "Entropy (normalized, identity view)",    True),
    ("agree_rate_base_d4",     "Agreement rate (base vs D4)",           False),
]
# sign=True means higher value → higher delta expected; False → lower value → higher delta


def spearman_row(x: pd.Series, y: pd.Series, metric: str, label: str, scope: str) -> dict:
    valid = x.notna() & y.notna()
    x_, y_ = x[valid], y[valid]
    if len(x_) < 4:
        return None
    r, p = stats.spearmanr(x_, y_)
    return {
        "metric":  metric,
        "label":   label,
        "scope":   scope,
        "n":       len(x_),
        "rho":     round(r, 4),
        "p":       round(p, 4),
    }


def run_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, label, _ in CANDIDATE_METRICS:
        if metric not in df.columns:
            continue
        # Pooled (excl. MHIST — binary dataset distorts rank)
        sub = df[df["dataset"] != "mhist"]
        row = spearman_row(sub[metric], sub["delta"], metric, label, "pooled_excl_mhist")
        if row:
            rows.append(row)

        # Pooled all
        row = spearman_row(df[metric], df["delta"], metric, label, "pooled_all")
        if row:
            rows.append(row)

        # Per dataset
        for ds in df["dataset"].unique():
            sub_ds = df[df["dataset"] == ds]
            row = spearman_row(sub_ds[metric], sub_ds["delta"], metric, label, ds)
            if row:
                rows.append(row)

        # Per model family (pooled excl mhist)
        for mt in ["histology_fm", "general"]:
            sub_mt = df[(df["dataset"] != "mhist") & (df["model_type"] == mt)]
            row = spearman_row(sub_mt[metric], sub_mt["delta"], metric, label, f"{mt}_excl_mhist")
            if row:
                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(corr: pd.DataFrame):
    print("\n" + "="*72)
    print("METRIC HORSE RACE — Spearman ρ vs TTA delta")
    print("="*72)

    pooled = corr[corr["scope"] == "pooled_excl_mhist"].sort_values("rho", ascending=False)
    print(f"\n{'Metric':<36} {'ρ':>6}  {'p':>6}  n")
    print("-"*60)
    for _, r in pooled.iterrows():
        star = "*" if r["p"] < 0.05 else " "
        print(f"  {r['label']:<34} {r['rho']:+.3f}{star}  {r['p']:.3f}  {r['n']}")

    print("\n  Per-dataset breakdown (top-2 metrics):")
    top2 = pooled.head(2)["metric"].tolist()
    for metric in top2:
        sub = corr[corr["metric"] == metric]
        ds_rows = sub[sub["scope"].isin(["tcga-ut","nct-crc-100k","nct-crc-nonorm","mhist"])]
        print(f"\n  {metric}:")
        for _, r in ds_rows.iterrows():
            star = "*" if r["p"] < 0.05 else " "
            print(f"    {r['scope']:<20} ρ={r['rho']:+.3f}{star}  p={r['p']:.3f}  n={r['n']}")

    # Family breakdown for top metric
    if top2:
        top = top2[0]
        lbl = corr[corr["metric"] == top]["label"].iloc[0]
        fm_row  = corr[(corr["metric"] == top) & (corr["scope"] == "histology_fm_excl_mhist")]
        gen_row = corr[(corr["metric"] == top) & (corr["scope"] == "general_excl_mhist")]
        if len(fm_row) and len(gen_row):
            fr, gr = fm_row.iloc[0], gen_row.iloc[0]
            print(f"\n  Best metric: {lbl}")
            print(f"    Histology FMs: ρ={fr['rho']:+.3f}  p={fr['p']:.3f}  n={fr['n']}")
            print(f"    General:       ρ={gr['rho']:+.3f}  p={gr['p']:.3f}  n={gr['n']}")


# ---------------------------------------------------------------------------
# Scatter figure
# ---------------------------------------------------------------------------

def make_scatter(df: pd.DataFrame, corr: pd.DataFrame, top_n: int = 2):
    pooled = corr[corr["scope"] == "pooled_excl_mhist"].sort_values("rho", ascending=False)
    top_metrics = pooled.head(top_n)["metric"].tolist()

    fig, axes = plt.subplots(1, len(top_metrics), figsize=(5 * len(top_metrics), 4.5))
    if len(top_metrics) == 1:
        axes = [axes]

    sub = df[df["dataset"] != "mhist"].copy()

    for ax, metric in zip(axes, top_metrics):
        lbl_row = pooled[pooled["metric"] == metric].iloc[0]
        xlabel  = lbl_row["label"]
        rho     = lbl_row["rho"]
        pval    = lbl_row["p"]

        # Scatter: color by model_type, marker by dataset
        for mt, mt_color in MT_COLORS.items():
            for ds, ds_marker in DS_MARKERS.items():
                mask = (sub["model_type"] == mt) & (sub["dataset"] == ds)
                pts  = sub[mask]
                if pts.empty or metric not in pts.columns:
                    continue
                valid = pts[metric].notna() & pts["delta"].notna()
                pts   = pts[valid]
                label = f"{mt.replace('_',' ')} / {ds}" if len(pts) else None
                ax.scatter(pts[metric], pts["delta"] * 100,
                           color=mt_color, marker=ds_marker, s=55,
                           alpha=0.8, linewidths=0.5, edgecolors="white",
                           label=label)

        # OLS regression line
        valid_all = sub[metric].notna() & sub["delta"].notna()
        x_all = sub.loc[valid_all, metric].values
        y_all = sub.loc[valid_all, "delta"].values * 100
        if len(x_all) >= 4:
            m_, b_ = np.polyfit(x_all, y_all, 1)
            x_line = np.linspace(x_all.min(), x_all.max(), 100)
            ax.plot(x_line, m_ * x_line + b_, color="black", lw=1.2, ls="--")

        # Annotation
        star = "*" if pval < 0.05 else ""
        ax.annotate(f"ρ = {rho:+.2f}{star}\n(n={valid_all.sum()})",
                    xy=(0.97, 0.05), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("TTA Δ balanced acc (pp)", fontsize=10)
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.tick_params(labelsize=9)

    # Compact legend on last axis
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        seen, h2, l2 = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l); h2.append(h); l2.append(l)
        axes[-1].legend(h2, l2, fontsize=7, framealpha=0.8,
                        loc="upper left", ncol=1)

    plt.tight_layout()
    OUT_FIG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_FIG, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nFigure saved → {OUT_FIG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no_figure", action="store_true")
    p.add_argument("--top_n", type=int, default=2,
                   help="Number of top metrics to show in scatter figure")
    args = p.parse_args()

    if not ORBIT_V2.exists():
        raise FileNotFoundError(
            f"{ORBIT_V2} not found — run analysis_orbit_from_embeddings.py first"
        )

    print("Loading data...")
    df = load_merged()
    print(f"  {len(df)} model×dataset combos | "
          f"models: {df['model'].nunique()} | datasets: {df['dataset'].nunique()}")

    print("Running correlation analysis...")
    corr = run_correlations(df)
    corr.to_csv(OUT_CORR, index=False)
    print(f"  Saved → {OUT_CORR}")

    print_summary(corr)

    if not args.no_figure:
        print("\nGenerating scatter figure...")
        make_scatter(df, corr, top_n=args.top_n)


if __name__ == "__main__":
    main()
