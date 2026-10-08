"""Reliability diagram: base vs D4 TTA, by model family (TCGA-UT).

Curves and ECE are computed PER MODEL AND SEED (canonical linear probes,
results/raw/probe_persample_*) then averaged, so opposite-direction
deviations between models cannot cancel. Bins holding under MIN_FRAC of a
model's samples are dropped from the plotted curve (they remain in the ECE,
which is sample-weighted and therefore unaffected by them).
"""
import glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

FM = {"phikon", "phikon2", "uni", "uni2", "virchow", "virchow2",
      "gigapath", "hoptimus", "ctranspath"}
DS, NBINS, MIN_FRAC = "tcga-ut", 10, 0.005
EDGES = np.linspace(0, 1, NBINS + 1)
CTR = (EDGES[:-1] + EDGES[1:]) / 2

def curve(conf, correct):
    idx = np.clip(np.digitize(conf, EDGES) - 1, 0, NBINS - 1)
    acc, cnf, n = np.full(NBINS, np.nan), np.full(NBINS, np.nan), np.zeros(NBINS)
    for b in range(NBINS):
        m = idx == b
        n[b] = m.sum()
        if n[b] > 0:
            acc[b], cnf[b] = correct[m].mean(), conf[m].mean()
    ece = np.nansum(n / n.sum() * np.abs(acc - cnf))      # all bins count
    sparse = (n / n.sum()) < MIN_FRAC                      # but hide sparse ones
    acc, cnf = acc.copy(), cnf.copy()
    acc[sparse], cnf[sparse] = np.nan, np.nan
    return acc, cnf, n, ece

fam_data = {"Histology FM": [], "General-purpose": []}
GENERAL = {"resnet18", "resnet50", "convnextv2_tiny", "convnextv2_base",
           "dinov2_s", "dinov2_b"}
# Canonical frozen linear probes, all five seeds (one curve per model x seed).
for f in sorted(glob.glob(f"results/raw/probe_persample_{DS}_*_linear_seed*.parquet")):
    model = os.path.basename(f)[len(f"probe_persample_{DS}_"):].split("_linear_seed")[0]
    if model not in FM and model not in GENERAL:
        continue
    d = pd.read_parquet(f)
    fam = "Histology FM" if model in FM else "General-purpose"
    fam_data[fam].append((
        curve(d.conf_0.values, d.correct_0.values.astype(float)),
        curve(d.conf_d4.values, d.correct_d4.values.astype(float)),
    ))

def mean_curve(rs, i):
    st = np.vstack([r[i] for r in rs])
    with np.errstate(invalid="ignore"):
        out = np.where(np.isnan(st).all(0), np.nan, np.nanmean(st, axis=0))
    return out

BORDER = "#bbbbbb"

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), sharey=True)
for ax, (fam, models) in zip(axes, fam_data.items()):
    base, tta = [m[0] for m in models], [m[1] for m in models]

    ax.fill_between([0, 1], [0, 1], [0, 0], color="0.92", zorder=0)
    ax.plot([0, 1], [0, 1], ls="--", c="0.45", lw=1, zorder=1,
            label="perfect calibration")
    ax.plot(mean_curve(base, 1), mean_curve(base, 0), "o-", c="#8C8C8C",
            lw=2, ms=5, zorder=3, label="single view")
    ax.plot(mean_curve(tta, 1), mean_curve(tta, 0), "s-", c="#0072B2",
            lw=2, ms=5, zorder=3, label="$D_4$ TTA")

    nrm = mean_curve(base, 2); nrm = nrm / np.nansum(nrm)
    axh = ax.twinx()
    axh.bar(CTR, nrm, width=0.085, color="0.6", alpha=0.25, zorder=0)
    axh.set_ylim(0, 1); axh.set_yticks([])
    for s in axh.spines.values():
        s.set_color(BORDER)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("confidence"); ax.set_title(fam)
    ax.set_axisbelow(True); ax.grid(True)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(color=BORDER)
    for gl in ax.get_xgridlines() + ax.get_ygridlines():   # match plot_figure3._dim_grid
        gl.set_alpha(0.15)

axes[0].set_ylabel("accuracy")
leg = axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
leg.get_frame().set_edgecolor(BORDER)
fig.tight_layout()
os.makedirs("paper/latex/figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"paper/latex/figures/reliability_tcga.{ext}", dpi=170,
                bbox_inches="tight", facecolor="white")

print(f"per-model ECE, TCGA-UT (bins <{MIN_FRAC:.1%} hidden from curve only)")
for fam, models in fam_data.items():
    b = [m[0][3] for m in models]; t = [m[1][3] for m in models]
    print(f"  {fam:16s} base {np.mean(b):.4f} -> D4 {np.mean(t):.4f}"
          f"  ({100*(1-np.mean(t)/np.mean(b)):.1f}% reduction)")
top = {f: np.mean([m[0][2][-1] / m[0][2].sum() for m in ms])
       for f, ms in fam_data.items()}
print("share of predictions in top confidence bin:",
      {k: f"{v:.1%}" for k, v in top.items()})
