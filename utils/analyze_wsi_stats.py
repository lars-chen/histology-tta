"""
WSI-level TTA statistics.
Outputs:
  results/boundary_proximity_stats.csv
  results/panda_slide_summary_stats.csv
  figures/boundary_proximity.pdf/.png
  figures/panda_slide_stats.pdf/.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import mannwhitneyu

OUT_DIR = Path("figures")
RES_DIR = Path("results")


# ── A. Boundary Proximity (Camelyon17) ──────────────────────────────────────

def analyze_boundary_proximity():
    df = pd.read_csv(RES_DIR / "camelyon17_boundary_dist.csv")

    # Normal patches (gt=0): these are where TTA corrections happen (FP → TN = specificity gain)
    # dist_um is positive for gt=0 patches (distance outside tumor boundary)
    normal = df[df['gt'] == 0].copy()
    outcomes = ["corrected", "both_correct", "corrupted", "both_wrong"]

    print(f"\nNormal patch counts: {normal.groupby('outcome').size().to_dict()}")

    # Per-outcome distance stats
    rows = []
    for oc in outcomes:
        sub = normal[normal.outcome == oc]["dist_um"]
        rows.append({
            "outcome": oc, "n": len(sub),
            "median_um": sub.median(), "mean_um": sub.mean(),
            "q25_um": sub.quantile(0.25), "q75_um": sub.quantile(0.75),
        })
    stats_df = pd.DataFrame(rows)
    print("\nBoundary distance stats (normal patches, gt=0):")
    print(stats_df.to_string(index=False))

    # Mann-Whitney: corrected (FP→TN) vs both_correct (TN→TN) — corrected should be closer
    corr = normal[normal.outcome == "corrected"]["dist_um"]
    both = normal[normal.outcome == "both_correct"]["dist_um"]
    stat, pval = mannwhitneyu(corr, both, alternative="less")
    n1, n2 = len(corr), len(both)
    effect = stat / (n1 * n2)
    print(f"\nMann-Whitney (corrected < both_correct dist): U={stat:.0f}, p={pval:.2e}, r={effect:.3f}")
    stats_df["mw_vs_both_correct_p"] = None
    stats_df.loc[stats_df.outcome == "corrected", "mw_vs_both_correct_p"] = pval
    stats_df.to_csv(RES_DIR / "boundary_proximity_stats.csv", index=False)

    # Binned P(corrected | distance bin) for normal patches
    bins = [0, 50, 100, 200, 500, 1000, 2000, np.inf]
    labels = ["0–50", "50–100", "100–200", "200–500", "500–1k", "1k–2k", ">2k"]
    normal["dist_bin"] = pd.cut(normal["dist_um"], bins=bins, labels=labels, right=False)
    binned = normal.groupby("dist_bin", observed=True)["outcome"].value_counts().unstack(fill_value=0)
    for oc in outcomes:
        if oc not in binned.columns:
            binned[oc] = 0
    binned["total"] = binned[outcomes].sum(axis=1)
    binned["p_corrected"] = binned["corrected"] / binned["total"]
    binned["p_corrupted"] = binned["corrupted"] / binned["total"]
    print("\nBinned P(corrected/corrupted) for normal patches vs distance:")
    print(binned[["corrected","corrupted","total","p_corrected","p_corrupted"]].to_string())

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3.8))

    # Panel B: P(corrected/corrupted) vs distance bin
    valid = binned[binned["total"] > 10]
    x = range(len(valid))
    ax.plot(x, valid["p_corrected"].values * 100, "o-",
            color="#4caf7d", lw=2, ms=6, label="P(FP→TN corrected)")
    ax.plot(x, valid["p_corrupted"].values * 100, "s--",
            color="#d94f3d", lw=1.5, ms=5, label="P(TN→FP corrupted)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(valid.index), fontsize=8)
    ax.set_xlabel("Distance to tumor boundary (µm)", fontsize=8.5)
    ax.set_ylabel("% of normal patches in bin", fontsize=8.5)
    ax.set_title("TTA correction rate vs distance to tumour boundary\n(normal patches, Camelyon17)", fontsize=8.5, pad=4)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for xi, (lbl, row) in enumerate(valid.iterrows()):
        ax.text(xi, -0.45, f"n={int(row['total'])}", ha="center",
                fontsize=5.5, color="#666666", transform=ax.get_xaxis_transform())

    fig.tight_layout(pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"boundary_proximity.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved → figures/boundary_proximity.pdf/.png")


# ── B. PANDA Per-Slide Stats ─────────────────────────────────────────────────

def analyze_panda_slides():
    df = pd.read_csv(RES_DIR / "patch_dice_panda.csv")
    pos = df[df.n_pos > 0].copy()
    pos["net_corr_rate"] = (pos.corrected - pos.corrupted) / pos.n_pos
    pos["improved_spec"]  = pos.delta_spec > 0
    pos["improved_dice"]  = pos.delta_dice > 0

    print(f"\nPANDA slides with tumor: {len(pos)}")
    print(f"  Spec improved: {pos.improved_spec.sum()} ({pos.improved_spec.mean()*100:.1f}%)")
    print(f"  Spec hurt:     {(~pos.improved_spec).sum()} ({(~pos.improved_spec).mean()*100:.1f}%)")
    print(f"  Δspec mean: {pos.delta_spec.mean()*100:+.3f} pp  median: {pos.delta_spec.median()*100:+.3f} pp")
    print(f"  Net corr rate mean: {pos.net_corr_rate.mean()*100:+.2f}%")

    summary = pos[["slide_id","n_patches","n_pos","pos_frac",
                    "corrected","corrupted","delta_dice","delta_spec",
                    "net_corr_rate","improved_spec"]].copy()
    summary.to_csv(RES_DIR / "panda_slide_summary_stats.csv", index=False)

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    # Panel A: Δspec vs pos_frac scatter
    ax = axes[0]
    c_map = pos["improved_spec"].map({True: "#4caf7d", False: "#d94f3d"})
    ax.scatter(pos.pos_frac * 100, pos.delta_spec * 100,
               c=c_map, s=4, alpha=0.3, linewidths=0)
    # Binned mean trend
    pos["pf_bin"] = pd.cut(pos.pos_frac, bins=10)
    trend = pos.groupby("pf_bin", observed=True)["delta_spec"].mean() * 100
    bin_centers = [(iv.left + iv.right) / 2 * 100 for iv in trend.index]
    ax.plot(bin_centers, trend.values, "k-", lw=2, zorder=5, label="binned mean")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Tumor patch fraction (%)", fontsize=8.5)
    ax.set_ylabel("Δ Specificity (pp)", fontsize=8.5)
    ax.set_title("A   Δ Specificity vs tumor density\n(PANDA, cancer slides only)", fontsize=8.5, pad=4)
    handles = [plt.scatter([], [], c="#4caf7d", s=20, label="Improved"),
               plt.scatter([], [], c="#d94f3d", s=20, label="Hurt")]
    ax.legend(handles=handles, fontsize=7)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # Panel B: histogram of Δspec
    ax = axes[1]
    ax.hist(pos.delta_spec * 100, bins=60, color="#2a5fa5", alpha=0.8, edgecolor="none")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.axvline(pos.delta_spec.mean() * 100, color="#e07b39", lw=1.8,
               label=f"mean {pos.delta_spec.mean()*100:+.3f} pp")
    ax.set_xlabel("Δ Specificity (pp)", fontsize=8.5)
    ax.set_ylabel("# slides", fontsize=8.5)
    ax.set_title("B   Distribution of Δ Specificity per slide\n(PANDA, cancer slides)", fontsize=8.5, pad=4)
    ax.legend(fontsize=8)
    pct_imp = pos.improved_spec.mean() * 100
    ax.text(0.97, 0.97, f"{pct_imp:.0f}% of slides improved",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            color="#4caf7d", fontweight="bold")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    fig.tight_layout(pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"panda_slide_stats.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → figures/panda_slide_stats.pdf/.png")


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    analyze_boundary_proximity()
    analyze_panda_slides()
