#!/usr/bin/env python3
"""probe_d4.py — Probe D4 symmetry structure of frozen backbone embeddings.

Usage:
    python probe_d4.py [path_to_embeddings]

If no path is given, looks for embeddings.pt in cwd, then falls back to the
first valid .npy in the embeddings/ subfolder.
"""

import sys, json, pathlib, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_embeddings(path: pathlib.Path):
    """Return (emb: ndarray N×V×D, transform_names: list[str])."""
    if path.suffix == ".pt":
        import torch
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            emb   = data["embeddings"].numpy()
            names = data.get("transform_names")
        else:
            emb   = data.numpy()
            names = None
    else:
        emb   = np.load(path)
        meta  = path.with_name(path.stem + "_meta.json")
        names = json.loads(meta.read_text()).get("transform_names") if meta.exists() else None

    if names is None:
        names = [f"transform_{i}" for i in range(emb.shape[1])]
    return emb, names


def resolve_path(default="embeddings.pt") -> pathlib.Path:
    p = pathlib.Path(default)
    if p.exists():
        return p
    skip = {"meta", "labels", "indices", "logits"}
    candidates = [
        c for c in sorted(pathlib.Path("embeddings").glob("*.npy"))
        if not any(s in c.stem for s in skip)
    ]
    if candidates:
        chosen = candidates[0]
        print(f"[info] '{default}' not found — using {chosen}")
        return chosen
    sys.exit("No embeddings found. Pass a path explicitly.")


# ── Analysis ──────────────────────────────────────────────────────────────────

PCA_DIMS = 64  # overridden by --pca_dims


def fit_linear_maps(emb, names):
    """PCA-reduce to 256 dims (orig view), then fit R_k: orig → transform_k."""
    N, V, D = emb.shape
    idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.2, random_state=0)

    pca = PCA(n_components=PCA_DIMS, random_state=0)
    pca.fit(emb[:, 0, :])
    views_pca = np.stack([pca.transform(emb[:, k, :]) for k in range(V)], axis=1)

    X = views_pca[:, 0, :]
    I = np.eye(PCA_DIMS)
    models, r2s_lin, r2s_identity, coef_dist_I = {}, {}, {}, {}
    for k in range(1, V):
        Y = views_pca[:, k, :]
        lin = LinearRegression(fit_intercept=False)
        lin.fit(X[idx_tr], Y[idx_tr])
        r2s_lin[k]      = r2_score(Y[idx_te], lin.predict(X[idx_te]), multioutput="uniform_average")
        r2s_identity[k] = r2_score(Y[idx_te], X[idx_te],              multioutput="uniform_average")
        coef_dist_I[k]  = np.linalg.norm(lin.coef_ - I, "fro") / np.sqrt(PCA_DIMS)
        models[k] = (lin, views_pca[:, k, :], lin.coef_.copy())
        print(f"  {names[k]:20s}  R²={r2s_lin[k]:.4f}  R²(I)={r2s_identity[k]:.4f}  ||W-I||/√D={coef_dist_I[k]:.4f}")

    return models, r2s_lin, r2s_identity, coef_dist_I, idx_te, pca, views_pca


def check_group_relation(views_pca, names, models, idx_te):
    """||R_90 ∘ R_90 − R_180||_F (per-sample avg) in PCA space."""
    X_te = views_pca[idx_te, 0, :]
    k90, k180 = names.index("rot90"), names.index("rot180")
    reg90,  _, _ = models[k90]
    reg180, _, _ = models[k180]
    R90sq  = reg90.predict(reg90.predict(X_te))
    R180   = reg180.predict(X_te)
    diff   = R90sq - R180
    frob   = np.sqrt((diff ** 2).sum(axis=1)).mean()
    rel    = frob / np.sqrt((R180 ** 2).sum(axis=1)).mean()
    return frob, rel


def plot_pca(emb, names, pca256, out="pca_d4.png", n_patches=20):
    N, V, D = emb.shape
    coords256 = pca256.transform(emb.reshape(N * V, D))
    pca2 = PCA(n_components=2, random_state=0)
    Z    = pca2.fit_transform(coords256).reshape(N, V, 2)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap    = plt.get_cmap("tab10")
    mstyles = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "<", ">", "p"]
    for pi in range(min(n_patches, N)):
        for ki in range(V):
            ax.scatter(Z[pi, ki, 0], Z[pi, ki, 1],
                       color=cmap(ki), marker=mstyles[pi % len(mstyles)],
                       s=55, alpha=0.75, label=names[ki] if pi == 0 else "")
    ax.legend(title="transform", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"PCA of D4 embeddings  (colour=transform, marker=patch, first {n_patches} patches)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_W(models, names, out="W_maps.png"):
    V = len(models) + 1
    I = np.eye(list(models.values())[0][2].shape[0])
    fig, axes = plt.subplots(3, V - 1, figsize=(3 * (V - 1), 9))

    for col, k in enumerate(range(1, V)):
        W    = models[k][2]
        vmax = np.abs(W).max()

        axes[0, col].imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        axes[0, col].set_title(names[k], fontsize=8)
        axes[0, col].axis("off")
        if col == 0: axes[0, col].set_ylabel("W", fontsize=8)

        diff = W - I
        vd   = np.abs(diff).max()
        axes[1, col].imshow(diff, cmap="RdBu_r", vmin=-vd, vmax=vd, aspect="auto")
        axes[1, col].axis("off")
        if col == 0: axes[1, col].set_ylabel("W − I", fontsize=8)

        diag    = np.diag(W)
        offdiag = W[~np.eye(len(W), dtype=bool)]
        axes[2, col].hist(offdiag, bins=60, alpha=0.6, color="steelblue", label="off-diag", density=True)
        axes[2, col].hist(diag,    bins=20, alpha=0.8, color="tomato",    label="diag",     density=True)
        axes[2, col].axvline(1, color="k", lw=0.8, ls="--")
        axes[2, col].axvline(0, color="k", lw=0.8, ls=":")
        axes[2, col].set_xlabel("value", fontsize=7)
        if col == 0: axes[2, col].legend(fontsize=6)

    plt.suptitle("Learned linear maps W  (top: W, mid: W−I, bottom: value histogram)", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


# ── Summary ───────────────────────────────────────────────────────────────────

def interpret(r2, gap):
    base = "linear" if r2 > 0.9 else "partially linear" if r2 > 0.7 else "nonlinear"
    return base if gap > 0.02 else f"{base} (≈ invariant)"


def print_summary(names, r2s_lin, r2s_identity, coef_dist_I, frob_abs, frob_rel):
    cw = max(len(n) for n in names[1:]) + 2
    print(f"\n{'transform':<{cw}}  {'R²(W)':>7}  {'R²(I)':>7}  {'||W-I||/√D':>11}  interpretation")
    print("-" * (cw + 46))
    for k in r2s_lin:
        gap = r2s_lin[k] - r2s_identity[k]
        print(f"{names[k]:<{cw}}  {r2s_lin[k]:>7.4f}  {r2s_identity[k]:>7.4f}  {coef_dist_I[k]:>11.4f}  {interpret(r2s_lin[k], gap)}")

    verdict = "holds" if frob_rel < 0.05 else "approximate" if frob_rel < 0.20 else "violated"
    print(f"\nGroup relation  R_90 ∘ R_90 ≈ R_180:")
    print(f"  ||R_90²(x) − R_180(x)||  per-sample avg : {frob_abs:.4f}")
    print(f"  relative to ||R_180(x)|| per-sample avg : {frob_rel:.4f}  → {verdict}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", default="embeddings.pt")
    ap.add_argument("--pca_dims", type=int, default=64,
                    help="PCA dims before fitting linear maps (default: 64)")
    args = ap.parse_args()

    path = resolve_path(args.path)
    emb, names = load_embeddings(path)
    N, V, D = emb.shape
    print(f"Loaded {path}:  N={N}  V={V}  D={D}")
    print(f"Transforms: {names}\n")

    global PCA_DIMS
    PCA_DIMS = args.pca_dims
    print(f"Fitting linear maps in PCA-{PCA_DIMS} space...")
    models, r2s_lin, r2s_identity, coef_dist_I, idx_te, pca256, views_pca = fit_linear_maps(emb, names)

    frob_abs, frob_rel = check_group_relation(views_pca, names, models, idx_te)

    print(f"\nSaved {plot_pca(emb, names, pca256)}")
    print_summary(names, r2s_lin, r2s_identity, coef_dist_I, frob_abs, frob_rel)
    print(f"Saved {plot_W(models, names)}")


if __name__ == "__main__":
    main()
