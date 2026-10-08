"""
Per-sample orbit geometry: does orbit POSITION relative to the decision boundary
(not orbit SIZE) predict which samples TTA flips/corrects?

Everything is re-derived from raw inputs — the 8-view cached embeddings and the
saved linear-probe weights (W, b) — without trusting intermediate result CSVs.
Model-level TTA deltas computed here are cross-checked against canonical_results.csv.

For each test sample we decompose its D4 orbit relative to the linear probe's
decision boundary between the sample's baseline top-2 classes (i, j):

  geometric margin   m = (logit_i - logit_j) / ||w_i - w_j||
                       = signed distance of the (view-0) embedding to the i|j plane
  orbit spread       s = std over the 8 views of  emb_v · (w_i - w_j)/||w_i - w_j||
                       = how far the orbit moves ALONG the boundary normal
  straddle ratio     r = m / s
                       small |r|  → orbit crosses the boundary → TTA can flip it

Size controls (direction-agnostic; expected to be weak predictors):
  orbit_var = mean squared distance of L2-normalised views to their centroid
  mpcs      = mean pairwise cosine similarity within the orbit

Outputs:
  results/orbit_geometry_persample/{dataset}_{model}.parquet   (optional, --save_persample)
  results/orbit_geometry_summary.csv      per (model, dataset): AUROCs + delta cross-check
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

EMB_BASE  = Path("/gpfs/scratch/lpc8816/histology_embeddings")
CKPT_BASE = Path("checkpoints/frozen")
CANON     = Path("results/canonical_results.csv")
OUT_DIR   = Path("results")
PS_DIR    = OUT_DIR / "orbit_geometry_persample"

FM_MODELS = {
    "phikon", "phikon2", "uni", "uni2",
    "virchow", "virchow2", "gigapath", "hoptimus", "ctranspath",
}
ALL_MODELS = [
    "phikon", "phikon2", "uni", "uni2",
    "virchow", "virchow2", "gigapath", "hoptimus", "ctranspath",
    "dinov2_s", "dinov2_b", "convnextv2_tiny", "convnextv2_base",
    "resnet18", "resnet50",
]
ALL_DATASETS = ["tcga-ut", "nct-crc-100k", "nct-crc-nonorm", "mhist"]


def model_type(m: str) -> str:
    return "histology_fm" if m in FM_MODELS else "general"


def load_probe(model: str, dataset: str, seed: int):
    """Return (W: (C,D), b: (C,)) numpy float32 from saved linear checkpoint."""
    path = CKPT_BASE / f"{dataset}_{model}_linear_seed{seed}.pt"
    if not path.exists():
        return None
    sd = torch.load(path, map_location="cpu")
    W = sd["net.0.weight"].numpy().astype(np.float32)   # (C, D)
    b = sd["net.0.bias"].numpy().astype(np.float32)     # (C,)
    return W, b


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _entropy_norm(probs):
    C = probs.shape[-1]
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(-1) / np.log(C)


def compute_persample(emb_mm, y, W, b, chunk=4096):
    """
    Stream over samples; return a dict of per-sample arrays.

    emb_mm : (N, 8, D) mmap or array of raw embeddings
    W,b    : linear probe weights
    """
    N, V, D = emb_mm.shape
    C = W.shape[0]
    Wt = W.T                                  # (D, C)

    out = {k: np.empty(N, dtype=np.float32) for k in
           ["entropy_0", "orbit_var", "mpcs", "geo_margin", "orbit_spread",
            "straddle", "conf_0"]}
    pred_0  = np.empty(N, dtype=np.int64)
    pred_d4 = np.empty(N, dtype=np.int64)

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        blk = np.asarray(emb_mm[s:e], dtype=np.float32)        # (B,8,D)
        B = blk.shape[0]

        # --- logits / probs for all views ---
        logits = blk.reshape(B * V, D) @ Wt + b                # (B*8, C)
        logits = logits.reshape(B, V, C)
        probs  = _softmax(logits, axis=-1)                     # (B,8,C)

        p0  = probs[:, 0, :]                                   # (B,C)
        pd4 = probs.mean(axis=1)                               # (B,C)
        pred_0[s:e]  = p0.argmax(1)
        pred_d4[s:e] = pd4.argmax(1)
        out["entropy_0"][s:e] = _entropy_norm(p0)
        out["conf_0"][s:e]    = p0.max(1)

        # --- orbit SIZE (direction-agnostic) ---
        vn = blk / (np.linalg.norm(blk, axis=-1, keepdims=True) + 1e-8)  # (B,8,D) unit
        centroid = vn.mean(axis=1, keepdims=True)              # (B,1,D)
        out["orbit_var"][s:e] = ((vn - centroid) ** 2).sum(-1).mean(1)
        ocn = np.linalg.norm(centroid[:, 0, :], axis=-1)       # (B,)
        out["mpcs"][s:e] = (V * ocn ** 2 - 1.0) / (V - 1)

        # --- orbit POSITION relative to top-2 decision boundary ---
        l0 = logits[:, 0, :]                                   # (B,C) baseline logits
        order = np.argsort(l0, axis=1)                         # ascending
        top1 = order[:, -1]
        top2 = order[:, -2]
        w_diff = W[top1] - W[top2]                             # (B,D) boundary normals
        nrm = np.linalg.norm(w_diff, axis=1) + 1e-8            # (B,)
        # geometric margin = signed distance of view-0 emb to the i|j hyperplane
        out["geo_margin"][s:e] = (l0[np.arange(B), top1] - l0[np.arange(B), top2]) / nrm
        # orbit spread ALONG the boundary normal
        n_hat = w_diff / nrm[:, None]                          # (B,D)
        proj = np.einsum("bvd,bd->bv", blk, n_hat)             # (B,8)
        spread = proj.std(axis=1)                              # (B,)
        out["orbit_spread"][s:e] = spread
        out["straddle"][s:e] = out["geo_margin"][s:e] / (spread + 1e-8)

    res = {**out}
    res["pred_0"]  = pred_0
    res["pred_d4"] = pred_d4
    res["true"]    = y
    res["correct_0"]  = pred_0 == y
    res["correct_d4"] = pred_d4 == y
    res["flipped"]    = pred_0 != pred_d4
    res["corrected"]  = (~res["correct_0"]) & res["correct_d4"]
    res["corrupted"]  = res["correct_0"] & (~res["correct_d4"])
    return res


def _safe_auroc(score, target):
    target = np.asarray(target).astype(int)
    if target.sum() < 10 or target.sum() > len(target) - 10:
        return np.nan
    return roc_auc_score(target, score)


def summarize(res, model, dataset, seed):
    """Per-(model,dataset) AUROCs predicting flip / correction, plus delta cross-check."""
    flip = res["flipped"]
    corr = res["corrected"]
    N = len(flip)

    # Predictors. Note signs: small margin / small straddle / low conf → more flips,
    # so we feed the negative for those to keep AUROC interpreted as
    # "higher score → event".
    preds = {
        "orbit_var":     res["orbit_var"],          # size: expect ~0.5
        "mpcs_neg":     -res["mpcs"],                # size: tighter→? expect weak
        "orbit_spread":  res["orbit_spread"],        # spread along normal
        "geo_margin_neg":-np.abs(res["geo_margin"]), # small margin → flip
        "entropy_0":     res["entropy_0"],           # softmax proxy (semi-circular)
        "straddle_neg": -np.abs(res["straddle"]),    # the joint geometry metric
    }

    row = {
        "model": model, "dataset": dataset, "seed": seed,
        "model_type": model_type(model), "n": N,
        "flip_rate": float(flip.mean()),
        "n_corrected": int(res["corrected"].sum()),
        "n_corrupted": int(res["corrupted"].sum()),
        "frac_straddle": float((np.abs(res["straddle"]) < 1).mean()),
    }
    for name, score in preds.items():
        row[f"auroc_flip__{name}"] = _safe_auroc(score, flip)
        row[f"auroc_corrected__{name}"] = _safe_auroc(score, corr)

    # --- model-level TTA delta, re-derived here, for cross-check ---
    bacc_0  = balanced_accuracy_score(res["true"], res["pred_0"])
    bacc_d4 = balanced_accuracy_score(res["true"], res["pred_d4"])
    row["base_bacc_recomputed"] = bacc_0
    row["d4_bacc_recomputed"]   = bacc_d4
    row["delta_recomputed"]     = bacc_d4 - bacc_0
    return row


def crosscheck(summary: pd.DataFrame, seed: int):
    """Compare recomputed TTA delta to canonical_results.csv (seed-matched)."""
    if not CANON.exists():
        print("  (canonical_results.csv not found — skipping cross-check)")
        return
    canon = pd.read_csv(CANON)
    lin = canon[(canon.head_type == "linear") & (canon.seed == seed)
                & (canon.backbone_mode == "frozen")]
    none = lin[lin.strategy == "none"][["model", "dataset", "balanced_acc"]] \
        .rename(columns={"balanced_acc": "canon_base"})
    d4 = lin[(lin.strategy == "d4") & (lin.aggregation == "mean")][
        ["model", "dataset", "balanced_acc"]].rename(columns={"balanced_acc": "canon_d4"})
    cc = none.merge(d4, on=["model", "dataset"])
    cc["canon_delta"] = cc.canon_d4 - cc.canon_base
    m = summary.merge(cc, on=["model", "dataset"], how="left")
    m["delta_abs_err"] = (m.delta_recomputed - m.canon_delta).abs()
    max_err = m["delta_abs_err"].max()
    mean_err = m["delta_abs_err"].mean()
    print(f"\n  Cross-check vs canonical (seed {seed}): "
          f"mean |Δ err| = {mean_err:.5f}, max = {max_err:.5f}")
    bad = m[m.delta_abs_err > 0.005]
    if len(bad):
        print(f"  WARNING: {len(bad)} combos differ >0.5pp from canonical:")
        print(bad[["model", "dataset", "delta_recomputed", "canon_delta",
                   "delta_abs_err"]].to_string(index=False))
    else:
        print("  All recomputed deltas match canonical within 0.5pp. ✓")


def print_summary(summary: pd.DataFrame):
    print("\n" + "=" * 78)
    print("PER-SAMPLE: AUROC predicting FLIP (pooled across model×dataset, mean±std)")
    print("=" * 78)
    sub = summary[summary.dataset != "mhist"]
    cols_flip = [c for c in summary.columns if c.startswith("auroc_flip__")]
    print(f"\n  {'predictor':<20} {'mean AUROC':>12} {'std':>8}  (n combos)")
    print("  " + "-" * 50)
    order = sorted(cols_flip, key=lambda c: -sub[c].mean(skipna=True))
    for c in order:
        name = c.replace("auroc_flip__", "")
        print(f"  {name:<20} {sub[c].mean():>12.3f} {sub[c].std():>8.3f}   {sub[c].notna().sum()}")

    print("\n  AUROC predicting CORRECTION (TTA fixes the sample):")
    print(f"  {'predictor':<20} {'mean AUROC':>12} {'std':>8}")
    print("  " + "-" * 50)
    cols_corr = [c for c in summary.columns if c.startswith("auroc_corrected__")]
    order = sorted(cols_corr, key=lambda c: -sub[c].mean(skipna=True))
    for c in order:
        name = c.replace("auroc_corrected__", "")
        print(f"  {name:<20} {sub[c].mean():>12.3f} {sub[c].std():>8.3f}")

    # By family
    print("\n  Best flip predictor by family (excl. mhist):")
    for c in order[:1]:
        pass
    top_flip = order[0] if (order := sorted(cols_flip, key=lambda c: -sub[c].mean(skipna=True))) else None
    for mt in ["histology_fm", "general"]:
        s2 = sub[sub.model_type == mt]
        line = f"    {mt:<14}"
        for c in sorted(cols_flip, key=lambda c: -sub[c].mean(skipna=True))[:3]:
            line += f"  {c.replace('auroc_flip__','')}={s2[c].mean():.3f}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_persample", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR / "orbit_geometry_summary.csv"))
    args = ap.parse_args()

    rows = []
    combos = [(m, d) for m in args.models for d in args.datasets
              if (EMB_BASE / m / d / "test" / "embeddings.npy").exists()
              and (CKPT_BASE / f"{d}_{m}_linear_seed{args.seed}.pt").exists()]
    print(f"Processing {len(combos)} model×dataset combos (seed {args.seed})...")

    if args.save_persample:
        PS_DIR.mkdir(parents=True, exist_ok=True)

    for model, dataset in combos:
        tdir = EMB_BASE / model / dataset / "test"
        emb = np.load(tdir / "embeddings.npy", mmap_mode="r")     # (N,8,D)
        y   = np.load(tdir / "labels.npy").astype(np.int64)
        probe = load_probe(model, dataset, args.seed)
        if probe is None:
            continue
        W, b = probe
        res = compute_persample(emb, y, W, b)
        row = summarize(res, model, dataset, args.seed)
        rows.append(row)
        print(f"  {model:<16} {dataset:<16} "
              f"flip={row['flip_rate']*100:5.1f}%  "
              f"AUROC(straddle→flip)={row['auroc_flip__straddle_neg']:.3f}  "
              f"AUROC(orbit_var→flip)={row['auroc_flip__orbit_var']:.3f}  "
              f"Δ={row['delta_recomputed']*100:+.2f}pp")

        if args.save_persample:
            keep = ["entropy_0", "conf_0", "orbit_var", "mpcs", "geo_margin",
                    "orbit_spread", "straddle", "pred_0", "pred_d4", "true",
                    "correct_0", "correct_d4", "flipped", "corrected", "corrupted"]
            df = pd.DataFrame({k: res[k] for k in keep})
            df.insert(0, "model", model)
            df.insert(1, "dataset", dataset)
            df.to_parquet(PS_DIR / f"{dataset}_{model}.parquet", index=False)

        del emb, res
        gc.collect()

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out, index=False)
    print(f"\nSaved summary → {args.out}  ({len(summary)} rows)")

    crosscheck(summary, args.seed)
    print_summary(summary)


if __name__ == "__main__":
    main()
