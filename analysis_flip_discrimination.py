"""
Among FLIPPED samples (pred_0 != pred_d4), can any signal separate
CORRECTED flips (TTA fixes it) from CORRUPTED flips (TTA breaks it)?

Routing tells you a flip is *likely*; this asks whether we can tell a *good*
flip from a *bad* one. Prediction entropy reportedly cannot (corrected/corrupted
entropy distributions overlap; corrected just wins on volume). We test whether
geometric / consensus signals do better.

Everything re-derived from raw 8-view embeddings + saved linear probe (W, b).

Signal families (per sample):
  BASELINE (view-0 only — usable for routing BEFORE augmenting):
    entropy_0     normalised Shannon entropy of base softmax
    conf_0        base top-1 probability
    geo_margin    geometric distance of base emb to its top-2 boundary
  ORBIT (all 8 views, direction-agnostic size — expected ~0.5):
    orbit_var, orbit_spread
  DESTINATION / CONSENSUS (all 8 views — needs augmentation):
    conf_d4         post-TTA top-1 probability (mean softmax)
    entropy_d4      post-TTA entropy
    view_agreement  fraction of 8 views whose argmax == pred_d4
    delta_conf      conf_d4 - conf_0
    d4_geo_margin   geometric distance of mean-emb to ITS top-2 boundary

Target (among flips): corrected = 1, corrupted = 0 (lateral wrong→wrong dropped).

Outputs:
  results/flip_discrimination_summary.csv   AUROC(corrected vs corrupted) per signal, per combo
  figures/flip_discrimination.png           distributions: entropy vs best signal
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

EMB_BASE  = Path("/gpfs/scratch/lpc8816/histology_embeddings")
CKPT_BASE = Path("checkpoints/frozen")
OUT_DIR   = Path("results")

FM_MODELS = {"phikon", "phikon2", "uni", "uni2", "virchow", "virchow2",
             "gigapath", "hoptimus", "ctranspath"}
ALL_MODELS = ["phikon", "phikon2", "uni", "uni2", "virchow", "virchow2",
              "gigapath", "hoptimus", "ctranspath", "dinov2_s", "dinov2_b",
              "convnextv2_tiny", "convnextv2_base", "resnet18", "resnet50"]
ALL_DATASETS = ["tcga-ut", "nct-crc-100k", "nct-crc-nonorm", "mhist"]

# Signals and the sign that should make "higher score → corrected".
# Unknown a priori for several — we report AUROC and let it speak (an AUROC far
# from 0.5 in EITHER direction means the signal separates; we orient by sign of
# the raw correlation so the reported AUROC is the separable (>=0.5) direction).
SIGNALS = ["entropy_0", "conf_0", "geo_margin",
           "orbit_var", "orbit_spread",
           "conf_d4", "entropy_d4", "view_agreement", "delta_conf", "d4_geo_margin"]


def model_type(m): return "histology_fm" if m in FM_MODELS else "general"


def load_probe(model, dataset, seed):
    p = CKPT_BASE / f"{dataset}_{model}_linear_seed{seed}.pt"
    if not p.exists():
        return None
    sd = torch.load(p, map_location="cpu")
    return sd["net.0.weight"].numpy().astype(np.float32), sd["net.0.bias"].numpy().astype(np.float32)


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _entropy_norm(p):
    C = p.shape[-1]
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(-1) / np.log(C)


def _geo_margin(logits, W):
    """Geometric distance of each row to its own top-2 class boundary.
    logits: (M, C), W: (C, D). Returns (M,)."""
    order = np.argsort(logits, axis=1)
    i, j = order[:, -1], order[:, -2]
    nrm = np.linalg.norm(W[i] - W[j], axis=1) + 1e-8
    M = logits.shape[0]
    return (logits[np.arange(M), i] - logits[np.arange(M), j]) / nrm


def compute(emb_mm, y, W, b, chunk=4096):
    N, V, D = emb_mm.shape
    C = W.shape[0]
    Wt = W.T
    sig = {s: np.empty(N, dtype=np.float32) for s in SIGNALS}
    pred_0  = np.empty(N, dtype=np.int64)
    pred_d4 = np.empty(N, dtype=np.int64)

    for s0 in range(0, N, chunk):
        e0 = min(s0 + chunk, N)
        blk = np.asarray(emb_mm[s0:e0], dtype=np.float32)       # (B,8,D)
        B = blk.shape[0]
        logits = (blk.reshape(B * V, D) @ Wt + b).reshape(B, V, C)
        probs = _softmax(logits, -1)                            # (B,8,C)

        p0  = probs[:, 0, :]
        pd4 = probs.mean(1)                                     # mean softmax
        a0  = p0.argmax(1)
        ad4 = pd4.argmax(1)
        pred_0[s0:e0], pred_d4[s0:e0] = a0, ad4

        # baseline
        sig["entropy_0"][s0:e0] = _entropy_norm(p0)
        sig["conf_0"][s0:e0]    = p0.max(1)
        sig["geo_margin"][s0:e0]= _geo_margin(logits[:, 0, :], W)

        # orbit size (direction-agnostic)
        vn = blk / (np.linalg.norm(blk, axis=-1, keepdims=True) + 1e-8)
        cen = vn.mean(1, keepdims=True)
        sig["orbit_var"][s0:e0]    = ((vn - cen) ** 2).sum(-1).mean(1)
        # spread along base top-2 normal
        order = np.argsort(logits[:, 0, :], axis=1)
        wdiff = W[order[:, -1]] - W[order[:, -2]]
        nhat = wdiff / (np.linalg.norm(wdiff, axis=1, keepdims=True) + 1e-8)
        sig["orbit_spread"][s0:e0] = np.einsum("bvd,bd->bv", blk, nhat).std(1)

        # destination / consensus
        sig["conf_d4"][s0:e0]    = pd4.max(1)
        sig["entropy_d4"][s0:e0] = _entropy_norm(pd4)
        sig["delta_conf"][s0:e0] = pd4.max(1) - p0.max(1)
        view_arg = probs.argmax(-1)                             # (B,8)
        sig["view_agreement"][s0:e0] = (view_arg == ad4[:, None]).mean(1)
        # geometric margin of the MEAN embedding to its top-2 boundary
        mean_emb = blk.mean(1)                                  # (B,D)
        mean_logits = mean_emb @ Wt + b
        sig["d4_geo_margin"][s0:e0] = _geo_margin(mean_logits, W)

    correct_0  = pred_0 == y
    correct_d4 = pred_d4 == y
    flipped    = pred_0 != pred_d4
    corrected  = (~correct_0) & correct_d4
    corrupted  = correct_0 & (~correct_d4)
    df = pd.DataFrame(sig)
    df["flipped"], df["corrected"], df["corrupted"] = flipped, corrected, corrupted
    return df


def discrimination_auroc(df):
    """Among flips that are corrected XOR corrupted: AUROC of each signal for
    corrected(1) vs corrupted(0), oriented to the separable direction."""
    sub = df[(df.corrected | df.corrupted)]
    y = sub.corrected.astype(int).values
    if y.sum() < 10 or (1 - y).sum() < 10:
        return None, len(sub), int(y.sum()), int((1 - y).sum())
    out = {}
    for s in SIGNALS:
        x = sub[s].values
        auc = roc_auc_score(y, x)
        # orient: report the >=0.5 side, note direction
        out[s] = max(auc, 1 - auc)
        out[s + "__dir"] = "+" if auc >= 0.5 else "-"
    return out, len(sub), int(y.sum()), int((1 - y).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR / "flip_discrimination_summary.csv"))
    ap.add_argument("--fig", default="figures/flip_discrimination.png")
    args = ap.parse_args()

    combos = [(m, d) for m in args.models for d in args.datasets
              if (EMB_BASE / m / d / "test" / "embeddings.npy").exists()
              and (CKPT_BASE / f"{d}_{m}_linear_seed{args.seed}.pt").exists()]
    print(f"Processing {len(combos)} combos (seed {args.seed})...")

    rows = []
    pooled_frames = []   # keep flips for the figure (tcga-ut representative)
    for model, dataset in combos:
        tdir = EMB_BASE / model / dataset / "test"
        emb = np.load(tdir / "embeddings.npy", mmap_mode="r")
        y = np.load(tdir / "labels.npy").astype(np.int64)
        probe = load_probe(model, dataset, args.seed)
        if probe is None:
            continue
        df = compute(emb, y, *probe)
        res, n_flip, n_corr, n_corrupt = discrimination_auroc(df)
        if res is not None:
            row = {"model": model, "dataset": dataset, "model_type": model_type(model),
                   "n_flip": n_flip, "n_corrected": n_corr, "n_corrupted": n_corrupt,
                   "corr_corrupt_ratio": n_corr / max(n_corrupt, 1)}
            row.update({k: v for k, v in res.items()})
            rows.append(row)
            print(f"  {model:<16}{dataset:<16} flips={n_flip:5d} "
                  f"(C:{n_corr} X:{n_corrupt})  "
                  f"AUROC sep  entropy={res['entropy_0']:.3f} "
                  f"view_agree={res['view_agreement']:.3f} "
                  f"conf_d4={res['conf_d4']:.3f} geo_margin={res['geo_margin']:.3f}")
        if dataset == "tcga-ut":
            d2 = df[df.corrected | df.corrupted].copy()
            d2["model"] = model
            pooled_frames.append(d2)
        del emb, df
        gc.collect()

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out, index=False)
    print(f"\nSaved → {args.out}")

    # ---- summary print ----
    sub = summary[summary.dataset != "mhist"]
    print("\n" + "=" * 74)
    print("AUROC separating CORRECTED vs CORRUPTED flips (pooled mean±std, n combos)")
    print("oriented to separable direction; 0.5 = no separation")
    print("=" * 74)
    means = {s: sub[s].mean() for s in SIGNALS}
    for s in sorted(SIGNALS, key=lambda s: -means[s]):
        fam = "view-0 " if s in ("entropy_0", "conf_0", "geo_margin") else \
              "orbit  " if s in ("orbit_var", "orbit_spread") else "dest.  "
        print(f"  [{fam}] {s:<16} {means[s]:.3f} ± {sub[s].std():.3f}")

    # ---- figure: entropy vs best destination signal ----
    if pooled_frames:
        allf = pd.concat(pooled_frames, ignore_index=True)
        best = max(SIGNALS, key=lambda s: means[s])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, signame, title in [(axes[0], "entropy_0", "Prediction entropy (base view)"),
                                    (axes[1], best, f"Best separator: {best}")]:
            c = allf[allf.corrected][signame]
            x = allf[allf.corrupted][signame]
            bins = np.linspace(min(c.min(), x.min()), max(c.max(), x.max()), 40)
            ax.hist(c, bins=bins, alpha=0.6, density=True, label=f"corrected (n={len(c)})", color="#2E7D32")
            ax.hist(x, bins=bins, alpha=0.6, density=True, label=f"corrupted (n={len(x)})", color="#C62828")
            ax.set_title(title, fontsize=11)
            ax.set_xlabel(signame); ax.set_ylabel("density"); ax.legend(fontsize=8)
        plt.suptitle("Corrected vs corrupted flips — TCGA-UT, all models pooled", fontsize=12)
        plt.tight_layout()
        Path(args.fig).parent.mkdir(exist_ok=True)
        plt.savefig(args.fig, dpi=170, bbox_inches="tight", facecolor="white")
        print(f"Figure → {args.fig}")


if __name__ == "__main__":
    main()
