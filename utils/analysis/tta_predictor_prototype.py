"""
Prototype: can per-sample TTA behavior be predicted from orbit variance + margin?

Single case: dinov2_s on TCGA-UT (weak general model, many flips).
- orbit variance: spread of the 8 D4-view embeddings (L2-normalized) about their centroid
- margin proxy: baseline confidence / entropy (from the per-sample parquet)
- outcome: corrected / corrupted / unchanged (from the parquet)

Hypothesis: flips concentrate where orbit variance is HIGH and margin is LOW;
beneficial flips (corrected) dominate there.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

MODEL, DATASET, SEED = "dinov2_s", "tcga-ut", 0
EMB = Path(f"/gpfs/scratch/lpc8816/histology_embeddings/{MODEL}/{DATASET}/test")
PARQ = Path(f"results/raw/probe_persample_{DATASET}_{MODEL}_linear_seed{SEED}.parquet")


def orbit_variance(emb):
    # emb: (N, 8, D). L2-normalize each view, then mean squared dist to per-sample centroid.
    v = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)
    centroid = v.mean(axis=1, keepdims=True)
    return ((v - centroid) ** 2).sum(-1).mean(1)   # (N,)


def main():
    emb = np.load(EMB / "embeddings.npy")             # (N,8,D)
    sid = np.load(EMB / "sample_ids.npy", allow_pickle=True)
    ov = orbit_variance(emb)
    e = pd.DataFrame({"sample_id": sid, "orbit_var": ov})

    p = pd.read_parquet(PARQ)
    df = p.merge(e, on="sample_id", how="inner")
    print(f"merged {len(df)} samples")

    df["changed"]   = df.pred_0 != df.pred_d4
    df["corrected"] = (~df.correct_0) & (df.correct_d4)
    df["corrupted"] = (df.correct_0) & (~df.correct_d4)
    df["margin"]    = df.conf_0                         # high conf = high margin
    n = len(df)
    print(f"changed={df.changed.mean()*100:.1f}%  corrected={df.corrected.sum()}  corrupted={df.corrupted.sum()}")

    # --- correlations ---
    print("\nPoint-biserial corr with 'changed':")
    for c in ["orbit_var", "margin", "entropy_0"]:
        r = np.corrcoef(df[c], df.changed.astype(float))[0, 1]
        print(f"  {c:10s} r={r:+.3f}")

    # --- 2D picture: flip & net-correction rate by orbit_var × entropy quartiles ---
    df["ov_q"]  = pd.qcut(df.orbit_var, 4, labels=["Q1lo", "Q2", "Q3", "Q4hi"])
    df["ent_q"] = pd.qcut(df.entropy_0, 4, labels=["Q1lo", "Q2", "Q3", "Q4hi"])
    print("\nFlip rate (%) by orbit_var (rows) × entropy (cols):")
    print((df.pivot_table("changed", "ov_q", "ent_q", aggfunc="mean") * 100).round(1).to_string())
    print("\nNet correction rate (corrected-corrupted, %) by orbit_var × entropy:")
    net = df.assign(net=df.corrected.astype(int) - df.corrupted.astype(int))
    print((net.pivot_table("net", "ov_q", "ent_q", aggfunc="mean") * 100).round(2).to_string())

    # --- logistic: predict 'changed' from orbit_var, entropy, interaction ---
    X = df[["orbit_var", "entropy_0"]].copy()
    X["interax"] = X.orbit_var * X.entropy_0
    X = (X - X.mean()) / X.std()
    for target in ["changed", "corrected"]:
        y = df[target].astype(int)
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
        coefs = dict(zip(X.columns, clf.coef_[0].round(3)))
        print(f"\nLogReg predict '{target}': AUC={auc:.3f}  coefs={coefs}")


if __name__ == "__main__":
    main()
