"""
Combined 2-panel inset for graphical abstract.
Left:  dot plot — full-D4 TTA gain per model family (linear head, tcga-ut)
Right: selective pareto curves — Δ balanced accuracy vs fraction receiving TTA
Both panels are driven by results/selective_tta_results.csv (linear head), so the
left-panel endpoints and right-panel curves are consistent.
Shared y-axis. Output: figures/combined_inset.png/.pdf
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import blended_transform_factory

from utils.plot_analysis import SAVEKW

C_HISTO   = "#2a5fa5"
C_GENERAL = "#e07b39"

SEL_CSV = Path(__file__).parent.parent / "results" / "selective_tta_results.csv"
DATASET = "tcga-ut"
HEAD    = "linear"

MODEL_TYPE = {
    'phikon':'histology','phikon2':'histology','uni':'histology','uni2':'histology',
    'virchow':'histology','virchow2':'histology','gigapath':'histology','hoptimus':'histology',
    'ctranspath':'histology',
    'dinov2_s':'general','dinov2_b':'general',
    'convnextv2_tiny':'general','convnextv2_base':'general',
    'resnet18':'general','resnet50':'general',
}


# ── data loaders ────────────────────────────────────────────────────────────

def _load_curves(sel_csv: Path = SEL_CSV):
    """Build per-family selective-TTA curves + per-model full-TTA endpoints.

    Reads the selective sweep CSV (linear head). Each model curve is averaged
    over seeds and aligned on the shared threshold grid. The full-TTA delta is
    the value at threshold 0 (100% coverage). Δ is converted to pp.
    """
    df = pd.read_csv(sel_csv)
    df = df[(df.dataset == DATASET) & (df.head_type == HEAD)]
    thresholds = np.sort(df.threshold.unique())

    curves: dict[str, list] = {"histology": [], "general": []}
    model_deltas: dict[str, list[float]] = {"histology": [], "general": []}
    for model, gm in df.groupby("model"):
        mtype = MODEL_TYPE.get(model)
        if mtype not in curves:
            continue
        # average over seeds, align by threshold grid
        agg = (gm.groupby("threshold")
                 .agg(coverage=("coverage_pct", "mean"),
                      delta=("delta_vs_baseline", "mean"))
                 .reindex(thresholds))
        f = agg["coverage"].values
        d = agg["delta"].values * 100.0          # fraction → pp
        curves[mtype].append((f, d))
        model_deltas[mtype].append(float(d[0]))   # threshold 0 → full coverage
    return curves, model_deltas, thresholds


def _family_avg(runs):
    # Each run is already a per-model average — weight models equally
    all_f = np.array([r[0] for r in runs])
    all_d = np.array([r[1] for r in runs])
    return all_f.mean(0), all_d.mean(0), all_d.std(0)


# ── plot ─────────────────────────────────────────────────────────────────────

def plot(out_dir: Path = Path("figures"), sel_csv: Path = SEL_CSV):
    curves, model_deltas, thresholds = _load_curves(sel_csv)

    # Per-model full-TTA deltas drive the dots; curves drive the right panel —
    # both come from the same selective CSV so the panels are consistent.
    histo_d   = np.array(model_deltas["histology"])
    general_d = np.array(model_deltas["general"])

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(11, 5.2), sharey=True,
        gridspec_kw={"width_ratios": [1.0, 2.5], "wspace": 0.08},
    )
    fig.patch.set_facecolor("white")

    # y-range: start at 0, clip top just above the highest dot
    Y_MIN = 0.0
    Y_MAX = max(general_d.max(), histo_d.max()) + 0.25
    TICK_STEP = 0.5 if (Y_MAX - Y_MIN) > 3 else 0.4

    # ── left: dot plot — values from CSV (all models) ────────────────────────
    rng = np.random.default_rng(0)
    jitter = 0.12

    ax_l.scatter(rng.uniform(-jitter, jitter, len(histo_d)),
                 histo_d, color=C_HISTO, s=48, alpha=0.85, lw=0, zorder=3, clip_on=False)
    ax_l.scatter(1 + rng.uniform(-jitter, jitter, len(general_d)),
                 general_d, color=C_GENERAL, s=48, alpha=0.85, lw=0, zorder=3, clip_on=False)

    hm, gm = histo_d.mean(), general_d.mean()
    ax_l.plot([-0.22, 0.22], [hm, hm], color=C_HISTO,   lw=3.2, zorder=4)
    ax_l.plot([ 0.78, 1.22], [gm, gm], color=C_GENERAL, lw=3.2, zorder=4)
    ax_l.text(0, hm + 0.75, f'+{hm:.2f}', va='bottom', ha='center',
              fontsize=19, color=C_HISTO, fontweight='bold')
    ax_l.text(1, gm + 0.75, f'+{gm:.2f}', va='bottom', ha='center',
              fontsize=19, color=C_GENERAL, fontweight='bold')

    ax_l.set_xticks([])
    ax_l.set_xlim(-0.55, 1.55)
    ax_l.set_ylabel('Δ Balanced Accuracy (pp)', fontsize=16)
    # group labels below x-axis
    tr = blended_transform_factory(ax_l.transData, ax_l.transAxes)
    ax_l.text(0,   -0.06, 'Histology', color=C_HISTO,   fontsize=14, fontweight='bold',
              ha='center', va='top', transform=tr)
    ax_l.text(1.0, -0.06, 'General',   color=C_GENERAL, fontsize=14, fontweight='bold',
              ha='center', va='top', transform=tr)

    # ── right: pareto curves ─────────────────────────────────────────────────
    for mtype, color in [("histology", C_HISTO), ("general", C_GENERAL)]:
        runs = curves[mtype]
        if not runs:
            continue
        f, d, s = _family_avg(runs)
        order = np.argsort(f)
        f, d, s = f[order], d[order], s[order]
        t_sorted = thresholds[order]
        ax_r.fill_between(f, d - s, d + s, color=color, alpha=0.15, lw=0, zorder=2)
        ax_r.plot(f, d, color=color, lw=3.2, zorder=3)
        cmap = LinearSegmentedColormap.from_list("", [color, "#e8eef7" if mtype == "histology" else "#fce8d8"])
        # markers every 0.1 in threshold (grid is 0.05-spaced → every 2nd point)
        ax_r.scatter(f[::2], d[::2], c=t_sorted[::2], cmap=cmap,
                     vmin=0, vmax=1, s=40, zorder=5, lw=0.8, edgecolors="#333333")

    ax_r.axhline(0, color='#888888', lw=0.8, ls='--', zorder=1)
    ax_r.set_xlim(0, 102)
    ax_r.set_xlabel("Samples receiving D₄ TTA (%)", fontsize=16)
    ax_r.set_xticks([0, 25, 50, 75, 100])
    ax_r.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:.0f}%"
    ))

    # ── shared y-axis ────────────────────────────────────────────────────────
    for ax in (ax_l, ax_r):
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(TICK_STEP))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.tick_params(axis="both", labelsize=15, length=5, width=1.0, color="#666666")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")
        ax.set_facecolor("white")
        ax.grid(False)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"combined_inset.{ext}", **SAVEKW)
    plt.close(fig)
    print(f"Saved → {out_dir}/combined_inset.png/.pdf")


if __name__ == "__main__":
    plot()
