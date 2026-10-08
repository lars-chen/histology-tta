"""
Compute orbit tightness metrics directly from cached embeddings.

Embeddings are pre-extracted to:
  $HISTO_EMB_DIR/{model}/{dataset}/test/embeddings.npy
  shape: (N, 8, D) — all 8 D4 views

Metrics computed per (model, dataset):
  OCN  = || mean_v(norm(phi_v(x))) ||  ∈ [0, 1]   (orbit centroid norm)
  MPCS = (8·OCN² - 1) / 7             ∈ [-1/7, 1]  (mean pairwise cosine sim, within orbit)
  within_class_mpcs                                  (mean pairwise cos sim, same-class pairs)
  between_class_mpcs                                 (mean pairwise cos sim, cross-class pairs; uses identity view)
  separability_ratio = within_class / between_class  (Fisher-like)
  per_class_ocn_{classname}                         (per-class mean OCN)

Usage:
  python analysis_orbit_from_embeddings.py
  python analysis_orbit_from_embeddings.py --models phikon uni --datasets tcga-ut
  python analysis_orbit_from_embeddings.py --per_class     # include per-class OCN in output
"""

import os
import argparse
import gc
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

SCRATCH = Path(os.environ.get("HISTO_EMB_DIR", "embeddings"))
OUT_CSV = Path("results/orbit_tightness_v2.csv")

FM_MODELS = {
    "phikon", "phikon2", "uni", "uni2",
    "virchow", "virchow2", "gigapath", "hoptimus", "ctranspath",
}

ALL_MODELS = [
    "phikon", "phikon2", "uni", "uni2",
    "virchow", "virchow2", "gigapath", "hoptimus", "ctranspath",
    "dinov2_s", "dinov2_b",
    "convnextv2_tiny", "convnextv2_base",
    "resnet18", "resnet50",
]

ALL_DATASETS = ["tcga-ut", "nct-crc-100k", "nct-crc-nonorm", "mhist"]


def model_type(name: str) -> str:
    return "histology_fm" if name in FM_MODELS else "general"


def compute_metrics(embs_raw: np.ndarray, labels: np.ndarray, class_names: list[str] | None,
                    chunk: int = 4096):
    """
    Args:
        embs_raw: (N, 8, D) float32 — raw (unnormalized) embeddings (may be mmap)
        labels:   (N,) int
        class_names: list of class name strings, length = max(labels)+1
        chunk:    rows processed at a time to bound peak RAM

    Returns dict of scalar metrics + optional per_class_ocn dict.
    """
    N, V, D = embs_raw.shape
    assert V == 8, f"Expected 8 views, got {V}"

    # --- Streaming OCN pass (avoids materialising all N×8×D at once) ---
    all_ocn = np.empty(N, dtype=np.float32)
    id_embs = np.empty((N, D), dtype=np.float32)   # identity-view unit vectors

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        blk = embs_raw[start:end].astype(np.float32)        # copy chunk to RAM
        norms = np.linalg.norm(blk, axis=-1, keepdims=True).clip(min=1e-8)
        blk = blk / norms                                    # (B, 8, D) unit vectors
        centroid = blk.mean(axis=1)                          # (B, D)
        all_ocn[start:end] = np.linalg.norm(centroid, axis=-1)
        id_embs[start:end] = blk[:, 0, :]

    ocn = all_ocn
    mpcs = (8.0 * ocn**2 - 1.0) / 7.0        # (N,)

    # --- Within-class MPCS (identity view, between samples of same class) ---
    classes = np.unique(labels)
    within_vals = []
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        c_embs = id_embs[idx]      # (k, D)
        c_mean = c_embs.mean(axis=0)  # (D,)
        k = len(idx)
        # mean pairwise cos sim = (k * ||centroid||^2 - 1) / (k-1)
        wmpcs = (k * float(np.dot(c_mean, c_mean)) - 1.0) / (k - 1)
        within_vals.append(wmpcs)
    mean_within = float(np.mean(within_vals))

    # --- Between-class MPCS (cross-class centroid similarity) ---
    # Compute per-class unit centroid; mean pairwise cos sim across class pairs
    class_centroids = []
    for c in classes:
        idx = np.where(labels == c)[0]
        m = id_embs[idx].mean(axis=0)
        norm_m = np.linalg.norm(m)
        if norm_m > 1e-8:
            class_centroids.append(m / norm_m)
    if len(class_centroids) > 1:
        CC = np.stack(class_centroids)  # (C, D)
        gram = CC @ CC.T                # (C, C)
        C = len(class_centroids)
        # Mean of off-diagonal elements
        mean_between = (gram.sum() - np.trace(gram)) / (C * (C - 1))
    else:
        mean_between = float("nan")

    sep_ratio = (mean_within / mean_between) if abs(mean_between) > 1e-8 else float("nan")

    out = {
        "mean_ocn":               float(ocn.mean()),
        "std_ocn":                float(ocn.std()),
        "p10_ocn":                float(np.percentile(ocn, 10)),
        "p90_ocn":                float(np.percentile(ocn, 90)),
        "mean_mpcs":              float(mpcs.mean()),
        "std_mpcs":               float(mpcs.std()),
        "mean_within_class_mpcs": mean_within,
        "mean_between_class_mpcs": mean_between,
        "separability_ratio":     sep_ratio,
        "n_samples":              N,
        "embed_dim":              D,
    }

    # --- Per-class OCN ---
    per_class = {}
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        cname = class_names[c] if class_names and c < len(class_names) else str(c)
        per_class[cname] = float(ocn[idx].mean())
    out["per_class_ocn"] = per_class

    return out


def run(models, datasets, include_per_class: bool, out_csv: Path = OUT_CSV):
    rows = []
    combos = [(m, d) for m in models for d in datasets
              if (SCRATCH / m / d / "test" / "embeddings.npy").exists()]

    print(f"Found {len(combos)} model×dataset combos in scratch.")

    for model, dataset in tqdm(combos, desc="computing orbit metrics"):
        test_dir = SCRATCH / model / dataset / "test"

        embs   = np.load(test_dir / "embeddings.npy", mmap_mode="r")  # (N, 8, D)
        labels = np.load(test_dir / "labels.npy")                     # (N,)

        class_file = SCRATCH / model / dataset / "classes.txt"
        class_names = class_file.read_text().strip().split("\n") if class_file.exists() else None

        res = compute_metrics(embs, labels, class_names)
        del embs  # free ~1 GB before next iteration
        gc.collect()

        row = {
            "model":        model,
            "dataset":      dataset,
            "model_type":   model_type(model),
            **{k: v for k, v in res.items() if k != "per_class_ocn"},
        }
        if include_per_class:
            for cname, val in res["per_class_ocn"].items():
                row[f"ocn__{cname}"] = val

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} rows → {out_csv}")
    return df


def print_summary(df: pd.DataFrame):
    print("\n" + "="*70)
    print("ORBIT TIGHTNESS SUMMARY (from cached embeddings)")
    print("="*70)

    for mt in ["histology_fm", "general"]:
        sub = df[df["model_type"] == mt]
        if sub.empty:
            continue
        print(f"\n  {mt}  (n={len(sub)} model×dataset combos)")
        print(f"  Mean OCN:            {sub['mean_ocn'].mean():.4f} ± {sub['mean_ocn'].std():.4f}")
        print(f"  Mean MPCS (orbit):   {sub['mean_mpcs'].mean():.4f} ± {sub['mean_mpcs'].std():.4f}")
        print(f"  Mean within-cls:     {sub['mean_within_class_mpcs'].mean():.4f}")
        print(f"  Mean between-cls:    {sub['mean_between_class_mpcs'].mean():.4f}")
        print(f"  Separability ratio:  {sub['separability_ratio'].mean():.3f}")

    print()
    # Per-model mean OCN across datasets
    pivot = df.pivot_table(index="model", columns="dataset", values="mean_ocn")
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean", ascending=False)
    print("  Mean OCN by model (across datasets):")
    print(pivot.round(4).to_string())

    # Rank correlation: OCN vs TTA delta
    try:
        from scipy import stats
        canon = pd.read_csv("results/canonical_results.csv")
        none_acc = canon[(canon["head_type"] == "linear") & (canon["strategy"] == "none")][
            ["model", "dataset", "seed", "balanced_acc"]
        ].rename(columns={"balanced_acc": "base_acc"})
        d4_acc = canon[(canon["head_type"] == "linear") & (canon["strategy"] == "d4") & (canon["aggregation"] == "mean")][
            ["model", "dataset", "seed", "balanced_acc"]
        ]
        delta = d4_acc.merge(none_acc, on=["model", "dataset", "seed"])
        delta["delta"] = delta["balanced_acc"] - delta["base_acc"]
        delta_mean = delta.groupby(["model", "dataset"])["delta"].mean().reset_index()

        merged = df[["model", "dataset", "mean_ocn", "model_type"]].merge(delta_mean, on=["model", "dataset"])
        merged_no_mhist = merged[merged["dataset"] != "mhist"]

        if len(merged_no_mhist) > 3:
            r, p = stats.spearmanr(merged_no_mhist["mean_ocn"], merged_no_mhist["delta"])
            print(f"\n  Spearman ρ(OCN, TTA delta) pooled 3 datasets (excl. MHIST), n={len(merged_no_mhist)}: ρ={r:.3f} p={p:.4f}")
            for ds in merged_no_mhist["dataset"].unique():
                sub2 = merged_no_mhist[merged_no_mhist["dataset"] == ds]
                r2, p2 = stats.spearmanr(sub2["mean_ocn"], sub2["delta"])
                print(f"    {ds}: ρ={r2:.3f} (n={len(sub2)}, p={p2:.3f})")
    except Exception as e:
        print(f"  (delta correlation skipped: {e})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=ALL_MODELS)
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    p.add_argument("--per_class", action="store_true",
                   help="Include per-class OCN columns in output CSV")
    p.add_argument("--out", default="results/orbit_tightness_v2.csv")
    args = p.parse_args()

    out_csv = Path(args.out)

    df = run(args.models, args.datasets, args.per_class, out_csv)
    print_summary(df)


if __name__ == "__main__":
    main()
