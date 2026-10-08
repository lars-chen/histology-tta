"""
Candidate figure(s) for the §4.7 orbit mechanism. Generates a 3-panel version
and the standalone "money panel" (c) for review before deciding inclusion.

(a) AUROC predicting a flip, by predictor: orbit SIZE metrics ~0.57 vs
    geometric margin / straddle ~0.95.            [orbit_geometry_summary.csv]
(b) Flip-rate heatmap over geometric-margin x orbit-spread quartiles (TCGA-UT
    pooled): flips concentrate at low margin + high spread. [persample parquets]
(c) MONEY PANEL: embedding invariance (MPCS, all-8 embedding coincidence) vs
    prediction stability (all-8 prediction agreement). D4WRN sits alone at
    MPCS~1.0; FMs cluster with general on MPCS yet stay prediction-stable.
    [orbit_tightness_v2.csv + orbit_tightness_d4wrn_*.csv + embeddings/W + d4wrn JSON]
"""
from __future__ import annotations
import os
import glob, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch

EMB = Path(os.environ.get("HISTO_EMB_DIR", "embeddings"))
CKPT = Path("checkpoints/frozen")
FM = {"phikon","phikon2","uni","uni2","virchow","virchow2","gigapath","hoptimus","ctranspath"}
DS3 = ["tcga-ut","nct-crc-100k","nct-crc-nonorm"]
MODELS = ["phikon","phikon2","uni","uni2","virchow","virchow2","gigapath","hoptimus","ctranspath",
          "dinov2_s","dinov2_b","convnextv2_tiny","convnextv2_base","resnet18","resnet50"]
C_FM, C_GEN, C_EQ = "#0072B2", "#D4772A", "#009E73"
plt.rcParams.update({"font.size":10})


# ---- strict all-8-views prediction agreement, computed from embeddings + probe ----
def strict_agreement(model, ds, seed=0, chunk=8192):
    p = CKPT / f"{ds}_{model}_linear_seed{seed}.pt"
    if not p.exists(): return np.nan
    sd = torch.load(p, map_location="cpu")
    W, b = sd["net.0.weight"].numpy(), sd["net.0.bias"].numpy()
    e = np.load(EMB/model/ds/"test/embeddings.npy", mmap_mode="r")
    N,V,D = e.shape; agree = np.empty(N, bool)
    for s in range(0,N,chunk):
        blk = np.asarray(e[s:s+chunk],np.float32); B=blk.shape[0]
        pv = (blk.reshape(B*V,D)@W.T+b).reshape(B,V,-1).argmax(-1)
        agree[s:s+B] = (pv==pv[:,:1]).all(1)
    return float(agree.mean())


def d4wrn_strict_agreement():
    out = {}
    for f in glob.glob("results/raw/tta_results_*d4wrn*aug_seed*.json"):
        for r in json.load(open(f)):
            if r.get("strategy")=="d4" and r.get("agreement_rate") is not None:
                out.setdefault(r["dataset"], []).append(r["agreement_rate"])
    return {k: float(np.mean(v)) for k,v in out.items()}


def panel_a(ax):
    s = pd.read_csv("results/orbit_geometry_summary.csv")
    s = s[s.dataset != "mhist"]
    # one orbit-SIZE bar (MPCS; orbit-var is identical, (7/8)(1-MPCS)) vs the
    # position/predictive metrics.
    metrics = [("auroc_flip__straddle_neg","straddle ratio","pos"),
               ("auroc_flip__geo_margin_neg","geom. margin","pos"),
               ("auroc_flip__entropy_0","entropy","pos"),
               ("auroc_flip__mpcs_neg","orbit size (MPCS)","size")]
    vals = [s[m].mean() for m,_,_ in metrics]
    errs = [s[m].std() for m,_,_ in metrics]
    cols = [C_FM if k=="pos" else "0.6" for _,_,k in metrics]
    y = np.arange(len(metrics))[::-1]
    ax.barh(y, vals, xerr=errs, color=cols, alpha=0.9, capsize=3)
    ax.set_yticks(y); ax.set_yticklabels([n for _,n,_ in metrics], fontsize=10)
    ax.axvline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xlim(0.45,1.0); ax.set_xlabel("AUROC predicting a flip")
    ax.set_title("(a) Orbit position predicts flips; size does not", fontsize=10)
    ax.text(0.5,0.97,"chance",rotation=90,fontsize=7,va="top",ha="right",color="k",transform=ax.get_xaxis_transform())


def panel_b(ax):
    dfs = [pd.read_parquet(f) for f in glob.glob("results/orbit_geometry_persample/tcga-ut_*.parquet")]
    d = pd.concat(dfs, ignore_index=True)
    d["am"] = np.abs(d.geo_margin)
    d["mq"] = pd.qcut(d.am.rank(method="first"), 5, labels=False)
    d["pq"] = pd.qcut(d.mpcs.rank(method="first"), 5, labels=False)   # per-sample orbit size
    H = d.pivot_table("flipped","pq","mq",aggfunc="mean").values*100
    im = ax.imshow(H, origin="lower", cmap="magma", aspect="auto")
    ax.set_xlabel("geometric margin  (low → high)")
    ax.set_ylabel("orbit size: MPCS  (loose → tight)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(b) Flips occur at low margin, across orbit sizes", fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("flip rate (%)", fontsize=8)


def panel_c(ax):
    v2 = pd.read_csv("results/orbit_tightness_v2.csv")
    v2 = v2[v2.dataset.isin(DS3)][["model","dataset","mean_mpcs"]]
    rows = []
    for _, r in v2.iterrows():
        ag = strict_agreement(r.model, r.dataset)
        rows.append(dict(model=r.model, dataset=r.dataset, mpcs=r.mean_mpcs, agree=ag,
                         fam="FM" if r.model in FM else "gen"))
    R = pd.DataFrame(rows)
    # D4WRN points
    d4mpcs = {}
    for f in glob.glob("results/orbit_tightness_d4wrn_*.csv"):
        c = pd.read_csv(f); d4mpcs[c.dataset.iloc[0]] = c.mean_mpcs.iloc[0]
    d4ag = d4wrn_strict_agreement()
    for ds in d4mpcs:
        if ds in d4ag:
            R = pd.concat([R, pd.DataFrame([dict(model="d4wrn",dataset=ds,mpcs=d4mpcs[ds],agree=d4ag[ds],fam="eq")])], ignore_index=True)
    for fam,col,lab in [("gen",C_GEN,"general"),("FM",C_FM,"histology FM"),("eq",C_EQ,"D4WRN (equivariant)")]:
        s = R[R.fam==fam]
        ax.scatter(s.mpcs, s.agree*100, c=col, s=70 if fam=="eq" else 45,
                   marker="*" if fam=="eq" else "o", edgecolors="white", linewidths=0.6,
                   alpha=0.9, label=lab, zorder=3 if fam=="eq" else 2)
    ax.set_xlabel("embedding invariance  (orbit MPCS)")
    ax.set_ylabel("prediction stability  (all-8 agreement, %)")
    ax.set_title("(c) Prediction stability ≠ embedding invariance", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
    return R


# ---- assemble ----
Path("figures").mkdir(exist_ok=True)

# 3-panel
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
panel_a(axes[0]); panel_b(axes[1]); R = panel_c(axes[2])
fig.tight_layout()
fig.savefig("figures/orbit_mechanism_3panel.png", dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved figures/orbit_mechanism_3panel.png")

# standalone money panel
fig, ax = plt.subplots(figsize=(6, 5))
panel_c(ax)
fig.tight_layout()
fig.savefig("figures/orbit_mechanism_panelC.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved figures/orbit_mechanism_panelC.png")
print("\npanel (c) data:")
print(R.assign(mpcs=R.mpcs.round(3), agree=(R.agree*100).round(1)).sort_values(["fam","mpcs"]).to_string(index=False))
