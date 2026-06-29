"""
Plot selective-TTA recovery curves: margin gate vs entropy gate.
One figure per dataset; a grid of per-model panels.
y = absolute balanced-accuracy gain (pp) vs % patches augmented.
Dashed line = full-D4 ceiling.  Reads results/selective_gating_summary.csv.
"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

FM = {"phikon","phikon2","uni","uni2","virchow","virchow2","gigapath","hoptimus","ctranspath"}
# FM first (by descending pretrain quality-ish), then general
ORDER = ["virchow2","hoptimus","uni2","gigapath","phikon2","uni","virchow","phikon","ctranspath",
         "dinov2_b","dinov2_s","convnextv2_base","convnextv2_tiny","resnet50","resnet18"]
PRETTY = {"phikon":"Phikon","phikon2":"Phikon-v2","uni":"UNI","uni2":"UNI2","virchow":"Virchow",
          "virchow2":"Virchow2","gigapath":"GigaPath","hoptimus":"H-Optimus","ctranspath":"CTransPath",
          "dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","convnextv2_tiny":"ConvNeXtV2-T",
          "convnextv2_base":"ConvNeXtV2-B","resnet18":"ResNet18","resnet50":"ResNet50"}

df = pd.read_csv("results/selective_gating_summary.csv")
df["gain_ent"] = df.gain_entropy_pp   # absolute pp gain, defined even when TTA hurts
df["gain_mar"] = df.gain_margin_pp
df["ceil"]     = (df.full_bacc-df.base_bacc)*100
Path("figures").mkdir(exist_ok=True)

for dataset in ["tcga-ut","nct-crc-100k","nct-crc-nonorm","mhist"]:
    sub = df[df.dataset==dataset]
    models = [m for m in ORDER if m in sub.model.unique()]
    n=len(models); ncol=5; nrow=int(np.ceil(n/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0*ncol, 2.5*nrow), sharex=True)
    axes=np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, models):
        s = sub[sub.model==m].sort_values("coverage")
        ceil = s.ceil.iloc[0]
        ax.axhline(ceil, ls=":", color="gray", lw=1, label=f"full-D4 ({ceil:+.2f})")
        # entropy thicker solid; margin thinner dashed on top, so when the two
        # coincide (e.g. binary mhist, where margin==entropy exactly) both show.
        ax.plot(s.coverage*100, s.gain_ent, "-o", ms=3.5, lw=2.4, color="#1f77b4",
                alpha=0.9, label="entropy")
        ax.plot(s.coverage*100, s.gain_mar, "--s", ms=2.2, lw=1.2, color="#d62728",
                label="margin")
        fam = "FM" if m in FM else "gen"
        col = "#0072B2" if m in FM else "#D4772A"
        ax.set_title(f"{PRETTY.get(m,m)} ({fam})", fontsize=9, color=col)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
        ax.legend(fontsize=5.5, loc="lower right", framealpha=0.7)
    for ax in axes[n:]: ax.axis("off")
    for ax in axes[(nrow-1)*ncol:]: ax.set_xlabel("% augmented", fontsize=8)
    for r in range(nrow): axes[r*ncol].set_ylabel("Δ bal-acc (pp)", fontsize=8)
    fig.suptitle(f"Selective TTA recovery: margin vs entropy gating — {dataset}", fontsize=12, y=1.0)
    fig.tight_layout()
    out=f"figures/selective_gating_{dataset}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("saved", out)
