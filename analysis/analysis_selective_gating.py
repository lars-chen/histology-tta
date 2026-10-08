"""
Selective-TTA gating: does base-view geometric MARGIN route better than ENTROPY?

Both signals are view-0 only (computable before augmenting). We route the
top-`coverage` fraction of most-uncertain patches to full D4 and leave the rest
at base view, then measure how much of the full-D4 benefit is recovered at each
coverage. Routing by coverage (not absolute threshold) makes the two signals an
apples-to-apples ranking comparison.

Headline metric: coverage needed to recover 90% of the full-D4 benefit
(lower = a better gate). Everything re-derived from embeddings + probe (W, b).

Output: results/selective_gating_summary.csv
        figures/selective_gating_pareto.png
"""
from __future__ import annotations
import os
import argparse, gc
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
from sklearn.metrics import balanced_accuracy_score

EMB_BASE = Path(os.environ.get("HISTO_EMB_DIR", "embeddings"))
CKPT_BASE = Path("checkpoints/frozen")
FM = {"phikon","phikon2","uni","uni2","virchow","virchow2","gigapath","hoptimus","ctranspath"}
MODELS = ["phikon","phikon2","uni","uni2","virchow","virchow2","gigapath","hoptimus","ctranspath",
          "dinov2_s","dinov2_b","convnextv2_tiny","convnextv2_base","resnet18","resnet50"]
DATASETS = ["tcga-ut","nct-crc-100k","nct-crc-nonorm","mhist"]
COVERAGES = np.round(np.arange(0.0, 1.0001, 0.05), 3)


def mtype(m): return "histology_fm" if m in FM else "general"


def load_probe(model, dataset, seed):
    p = CKPT_BASE / f"{dataset}_{model}_linear_seed{seed}.pt"
    if not p.exists(): return None
    sd = torch.load(p, map_location="cpu")
    return sd["net.0.weight"].numpy().astype(np.float32), sd["net.0.bias"].numpy().astype(np.float32)


def _softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True); e = np.exp(z); return e / e.sum(axis, keepdims=True)


def per_view(emb_mm, W, b, chunk=4096):
    """Return base probs p0 (N,C), full-d4 probs pd4 (N,C), base entropy, base |geo_margin|."""
    N, V, D = emb_mm.shape; C = W.shape[0]; Wt = W.T
    p0 = np.empty((N, C), np.float32); pd4 = np.empty((N, C), np.float32)
    ent = np.empty(N, np.float32); margin = np.empty(N, np.float32)
    for s in range(0, N, chunk):
        e = min(s+chunk, N); blk = np.asarray(emb_mm[s:e], np.float32); B = blk.shape[0]
        logits = (blk.reshape(B*V, D) @ Wt + b).reshape(B, V, C)
        probs = _softmax(logits, -1)
        b0 = probs[:, 0, :]; p0[s:e] = b0; pd4[s:e] = probs.mean(1)
        pc = np.clip(b0, 1e-12, 1); ent[s:e] = -(pc*np.log(pc)).sum(1)/np.log(C)
        l0 = logits[:, 0, :]; order = np.argsort(l0, 1); i, j = order[:, -1], order[:, -2]
        nrm = np.linalg.norm(W[i]-W[j], axis=1)+1e-8
        margin[s:e] = (l0[np.arange(B), i]-l0[np.arange(B), j])/nrm
    return p0, pd4, ent, margin


def selective_curve(p0, pd4, y, signal):
    """For each coverage, route top-fraction by `signal` (high=uncertain) to d4.
    Return array of balanced acc per coverage."""
    base_pred = p0.argmax(1); d4_pred = pd4.argmax(1); N = len(y)
    rank = np.argsort(-signal)           # most-uncertain first
    out = []
    for c in COVERAGES:
        k = int(round(c*N))
        pred = base_pred.copy()
        if k > 0:
            sel = rank[:k]; pred[sel] = d4_pred[sel]
        out.append(balanced_accuracy_score(y, pred))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for model in args.models:
        for dataset in args.datasets:
            tdir = EMB_BASE/model/dataset/"test"
            if not (tdir/"embeddings.npy").exists(): continue
            probe = load_probe(model, dataset, args.seed)
            if probe is None: continue
            emb = np.load(tdir/"embeddings.npy", mmap_mode="r")
            y = np.load(tdir/"labels.npy").astype(np.int64)
            p0, pd4, ent, margin = per_view(emb, *probe)
            base_b = balanced_accuracy_score(y, p0.argmax(1))
            full_b = balanced_accuracy_score(y, pd4.argmax(1))
            denom = full_b - base_b
            cur_ent = selective_curve(p0, pd4, y, ent)
            cur_mar = selective_curve(p0, pd4, y, -margin)   # low margin = uncertain
            for c, be, bm in zip(COVERAGES, cur_ent, cur_mar):
                rows.append(dict(model=model, dataset=dataset, model_type=mtype(model),
                    coverage=c, base_bacc=base_b, full_bacc=full_b,
                    sel_bacc_entropy=be, sel_bacc_margin=bm,             # absolute selective acc
                    gain_entropy_pp=(be-base_b)*100, gain_margin_pp=(bm-base_b)*100,  # always defined
                    rec_entropy=(be-base_b)/denom if denom>1e-9 else np.nan,
                    rec_margin=(bm-base_b)/denom if denom>1e-9 else np.nan))
            del emb; gc.collect()
            print(f"  {model:<16}{dataset:<16} Δ={denom*100:+.2f}pp  done")

    df = pd.DataFrame(rows)
    df.to_csv("results/selective_gating_summary.csv", index=False)

    # coverage to reach 90% recovery, per signal, per combo
    def cov_to_90(sub, col):
        s = sub.sort_values("coverage")
        hit = s[s[col] >= 0.90]
        return hit.coverage.min() if len(hit) else 1.0
    recs = []
    for (m, d), sub in df.groupby(["model","dataset"]):
        recs.append(dict(model=m, dataset=d, model_type=mtype(m),
            cov90_entropy=cov_to_90(sub,"rec_entropy"),
            cov90_margin=cov_to_90(sub,"rec_margin")))
    cov = pd.DataFrame(recs)
    print("\n=== Coverage to recover 90% of full-D4 benefit (lower=better) ===")
    for mt in ["histology_fm","general"]:
        s = cov[cov.model_type==mt]
        print(f"  {mt:<14} entropy={s.cov90_entropy.mean():.3f}  margin={s.cov90_margin.mean():.3f}  "
              f"(margin better in {int((s.cov90_margin<s.cov90_entropy).sum())}/{len(s)} combos)")
    print(f"\n  Overall mean Δcoverage (entropy-margin) = "
          f"{(cov.cov90_entropy-cov.cov90_margin).mean():+.4f}  "
          f"(>0 means margin needs less budget)")

    # mean recovery curves
    g = df[df.dataset!='mhist'].groupby(["model_type","coverage"])[["rec_entropy","rec_margin"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, mt in zip(axes, ["histology_fm","general"]):
        s = g[g.model_type==mt]
        ax.plot(s.coverage*100, s.rec_entropy*100, "-o", ms=3, label="entropy gate", color="#1f77b4")
        ax.plot(s.coverage*100, s.rec_margin*100, "-s", ms=3, label="margin gate", color="#d62728")
        ax.axhline(90, ls=":", color="gray", lw=0.8)
        ax.set_title(mt.replace("_"," ")); ax.set_xlabel("% patches augmented")
        ax.set_ylabel("% of full-D4 benefit recovered"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle("Selective TTA: margin vs entropy gating", fontsize=12)
    plt.tight_layout(); Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/selective_gating_pareto.png", dpi=170, bbox_inches="tight", facecolor="white")
    print("\nSaved results/selective_gating_summary.csv and figures/selective_gating_pareto.png")


if __name__ == "__main__":
    main()
