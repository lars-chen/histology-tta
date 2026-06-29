"""
Train a lightweight classifier head on pre-extracted frozen embeddings and
evaluate with D4 TTA. Writes two output files per run:

  results/raw/probe_{dataset}_{model}_{head_type}_seed{N}.json
      Aggregate metrics: one record per (strategy × aggregation).

  results/raw/probe_persample_{dataset}_{model}_{head_type}_seed{N}.parquet
      Per-sample: entropy, predictions, correctness for baseline and full D4.

Used downstream by:
  compile_results.py  → canonical_results.csv, selective_tta_results.csv
  compute_calibration.py → calibration_aggregate.csv, calibration/per_sample/
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

EMB_BASE = Path("/gpfs/scratch/lpc8816/histology_embeddings")
CKPT_BASE = Path("checkpoints/frozen")
OUT_DIR   = Path("results/raw")

# Compute budget. Both HP selection (k-fold CV over the lr×wd grid) and the
# final fit use the FULL training set — k-fold CV on all training data to pick
# lr/wd, then train on all data with the chosen values. kNN keeps a capped
# reference set, the one unavoidable exception: full brute-force kNN on 190k
# refs × 40k queries is infeasible without an ANN index.
TRAIN_CAP    = 100_000_000  # effectively no cap on the final fit (override via --train_cap)
CV_CAP       = 100_000_000  # effectively no cap on k-fold HP selection either
KNN_REF_CAP  = 20_000       # reference samples for kNN (brute-force tractability)
CV_FOLDS     = 3
CV_SEED      = 0            # fixed seed for CV folds → HP selection independent of run seed
# Trainable (torch) heads — all share one training recipe, differing only in
# depth. linear is a bare nn.Linear (0 hidden layers), i.e. an SGD-trained
# linear probe (the DINOv2/SimCLR-style protocol).
HIDDEN_LAYERS = {"linear": 0, "mlp_1h": 1, "mlp_2h": 2}

# lr × weight-decay grid (weight decay is the regularizer for the linear head).
MLP_LRS     = [1e-3, 3e-4]
MLP_WDS     = [1e-4, 1e-3]

# Torch training: fixed batch size across all models/datasets; early stopping on
# a held-out validation split (max MLP_MAX_EPOCHS epochs, patience MLP_PATIENCE).
MLP_BATCH      = 512
MLP_MAX_EPOCHS = 30
MLP_PATIENCE   = 3
MLP_VAL_FRAC   = 0.1

# ── helpers ─────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(model: str, dataset: str, split: str):
    p = EMB_BASE / model / dataset / split
    emb = np.load(p / "embeddings.npy", mmap_mode="r").astype(np.float32)
    lbl = np.load(p / "labels.npy").astype(np.int64)
    ids = np.load(p / "sample_ids.npy", allow_pickle=True)
    classes = (EMB_BASE / model / dataset / "classes.txt").read_text().splitlines()
    return emb, lbl, ids, classes


def _entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy of probability rows, normalised to [0, 1]."""
    C = probs.shape[-1]
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(-1) / np.log(C)


def _balanced_acc(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)


def _subsample(X: np.ndarray, y: np.ndarray, cap: int, seed: int):
    """Stratified subsample to at most `cap` rows. Returns (X_sub, y_sub)."""
    n = len(y)
    if n <= cap:
        return X, y
    rng = np.random.default_rng(seed)
    # stratified: sample proportionally within each class
    idx_keep = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        n_c = max(1, int(round(len(c_idx) * cap / n)))
        n_c = min(n_c, len(c_idx))
        idx_keep.append(rng.choice(c_idx, size=n_c, replace=False))
    idx = np.concatenate(idx_keep)
    rng.shuffle(idx)
    return np.ascontiguousarray(X[idx]), y[idx]


# ── head definitions ─────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Probe head with `hidden_layers` ∈ {0, 1, 2}. 0 = bare linear classifier
    (the linear probe), so linear / mlp_1h / mlp_2h share one training recipe."""
    def __init__(self, in_dim: int, n_classes: int, hidden_layers: int):
        super().__init__()
        h = in_dim // 2
        if hidden_layers == 0:
            layers = [nn.Linear(in_dim, n_classes)]
        elif hidden_layers == 1:
            layers = [nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, n_classes)]
        else:  # 2
            layers = [nn.Linear(in_dim, in_dim), nn.ReLU(),
                      nn.Linear(in_dim, h),      nn.ReLU(),
                      nn.Linear(h, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              n_classes: int, hidden_layers: int,
              lr: float = 1e-3, wd: float = 1e-4,
              max_epochs: int = MLP_MAX_EPOCHS, patience: int = MLP_PATIENCE,
              batch: int = MLP_BATCH, val_frac: float = MLP_VAL_FRAC,
              seed: int = 0):
    """Train an MLP head on the full data with early stopping.

    A stratified `val_frac` slice is held out for validation; training runs up
    to `max_epochs` and stops when val loss has not improved for `patience`
    epochs, restoring the best weights. Fixed batch size across all runs.
    """
    D = X_train.shape[1]
    # stratified validation split for early stopping
    rng = np.random.default_rng(seed)
    tr_idx, va_idx = [], []
    for c in np.unique(y_train):
        ci = np.where(y_train == c)[0]
        rng.shuffle(ci)
        nval = max(1, int(round(len(ci) * val_frac)))
        va_idx.append(ci[:nval]); tr_idx.append(ci[nval:])
    tr_idx = np.concatenate(tr_idx); va_idx = np.concatenate(va_idx)
    Xtr = torch.from_numpy(np.ascontiguousarray(X_train[tr_idx]))
    ytr = torch.from_numpy(y_train[tr_idx])
    Xva = torch.from_numpy(np.ascontiguousarray(X_train[va_idx]))
    yva = torch.from_numpy(y_train[va_idx])

    model = MLP(D, n_classes, hidden_layers)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    N = len(Xtr)
    best_loss, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        idx = torch.randperm(N)
        for i in range(0, N, batch):
            b = idx[i:i+batch]
            opt.zero_grad()
            loss_fn(model(Xtr[b]), ytr[b]).backward()
            opt.step()
        # batched validation loss
        model.eval()
        with torch.no_grad():
            tot, n = 0.0, 0
            for i in range(0, len(Xva), 8192):
                xb, yb = Xva[i:i+8192], yva[i:i+8192]
                tot += loss_fn(model(xb), yb).item() * len(xb); n += len(xb)
            vloss = tot / n
        if vloss < best_loss - 1e-4:
            best_loss = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def get_mlp_logits(model: nn.Module, X: np.ndarray, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.from_numpy(np.ascontiguousarray(X[i:i+batch]))).numpy())
    return np.concatenate(out)


def select_mlp_hp(X: np.ndarray, y: np.ndarray, n_classes: int,
                  hidden_layers: int) -> tuple[float, float, float]:
    """Grid-search lr × weight_decay by stratified K-fold CV (balanced acc).

    Uses a FIXED fold seed so the chosen hyperparameters are independent of the
    run seed (selection vs. init-variance are kept separate). Returns the best
    (lr, wd, cv_score).
    """
    Xs, ys = _subsample(X, y, CV_CAP, CV_SEED)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    best = (MLP_LRS[0], MLP_WDS[0], -1.0)
    for lr in MLP_LRS:
        for wd in MLP_WDS:
            scores = []
            for tr_idx, va_idx in skf.split(Xs, ys):
                torch.manual_seed(CV_SEED)
                m = train_mlp(np.ascontiguousarray(Xs[tr_idx]), ys[tr_idx],
                              n_classes, hidden_layers, lr=lr, wd=wd, seed=CV_SEED)
                preds = get_mlp_logits(m, np.ascontiguousarray(Xs[va_idx])).argmax(1)
                scores.append(balanced_accuracy_score(ys[va_idx], preds))
            mean_score = float(np.mean(scores))
            if mean_score > best[2]:
                best = (lr, wd, mean_score)
    return best


# ── train head ───────────────────────────────────────────────────────────────

def train_head(head_type: str, X_train: np.ndarray, y_train: np.ndarray,
               n_classes: int, seed: int, mlp_hp: tuple | None = None):
    """Train a head with cross-validated hyperparameters.

    Returns (head, hp_dict) where hp_dict records the selected hyperparameters
    for provenance. HP selection (CV grid) is done on a stratified subsample;
    the final head is refit on the full training set.

    `mlp_hp`: optional (lr, wd) to skip lr/wd selection (used so the chosen
    architecture is shared across seeds — selection is done once and cached).
    """
    # Final training uses the full training set (no cap by default).
    Xf, yf = _subsample(X_train, y_train, TRAIN_CAP, CV_SEED)

    # Torch heads: linear (0 hidden), mlp_1h (1), mlp_2h (2) — one shared recipe.
    if head_type in HIDDEN_LAYERS:
        hidden = HIDDEN_LAYERS[head_type]
        if mlp_hp is None:
            lr, wd, _ = select_mlp_hp(X_train, y_train, n_classes, hidden)
        else:
            lr, wd = mlp_hp
        torch.manual_seed(seed)
        head = train_mlp(Xf, yf, n_classes, hidden, lr=lr, wd=wd, seed=seed)
        return head, {"lr": lr, "wd": wd, "n_train": int(len(yf))}

    if head_type in ("knn_5", "knn_10"):
        k = int(head_type.split("_")[1])
        # Cap reference set on large datasets for tractable brute-force search.
        Xr, yr = _subsample(X_train, y_train, KNN_REF_CAP, CV_SEED)
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        clf.fit(Xr, yr)
        return clf, {"k": k, "n_ref": int(len(yr))}

    raise ValueError(f"Unknown head_type: {head_type}")


# ── inference: get logits / scores for (N, D) ───────────────────────────────

def head_logits(head_type: str, head, X: np.ndarray) -> np.ndarray:
    """Return (N, C) score/logit matrix."""
    if head_type in HIDDEN_LAYERS:               # torch heads (linear / mlp)
        return get_mlp_logits(head, X)
    if head_type in ("knn_5", "knn_10"):
        return head.predict_proba(X)             # probabilities
    raise ValueError(head_type)


def head_probs(head_type: str, head, X: np.ndarray) -> np.ndarray:
    """Return (N, C) probability matrix (softmax or predict_proba)."""
    logits = head_logits(head_type, head, X)
    if head_type in ("knn_5", "knn_10"):
        return logits                            # already probabilities
    # torch heads: softmax over logits
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def head_scores(head_type: str, head, X: np.ndarray) -> np.ndarray:
    """Return (N, C) *pre-softmax* scores, suitable for averaging in logit space.

    For a linear head these are the raw logits Wx+b, so averaging them across
    views equals evaluating on the mean embedding (zero Jensen gap). Only used
    for the `logit_mean` aggregation; scale/offset are irrelevant to argmax.
    Not defined for kNN.
    """
    if head_type in HIDDEN_LAYERS:           # torch heads (linear / mlp)
        return get_mlp_logits(head, X)
    raise ValueError(f"logit_mean not defined for head_type={head_type}")


# ── TTA evaluation ───────────────────────────────────────────────────────────

def evaluate(head_type: str, head, test_emb: np.ndarray,
             test_lbl: np.ndarray, test_ids: np.ndarray,
             classes: list[str]) -> tuple[list[dict], pd.DataFrame]:
    """
    Evaluate head on 8-view test embeddings.

    Returns
    -------
    records   : list of aggregate dicts (one per strategy × aggregation)
    per_sample: DataFrame with per-sample stats for baseline and full D4
    """
    N, V, D = test_emb.shape
    assert V == 8

    # For kNN, TTA = average embeddings first, then predict
    is_knn = head_type.startswith("knn")

    # --- per-view scores/probs ---
    if is_knn:
        # kNN: can't meaningfully apply per-view independently → use mean emb for d4
        probs_0   = head_probs(head_type, head, test_emb[:, 0, :])      # (N, C) baseline
        probs_d4  = head_probs(head_type, head, test_emb.mean(1))       # (N, C) d4 mean emb
        all_probs = np.stack([probs_0, probs_d4], axis=0)               # (2, N, C) pseudo-views
        views_for_mean = all_probs                                       # just use both
    else:
        all_probs = np.stack(
            [head_probs(head_type, head, test_emb[:, v, :]) for v in range(V)],
            axis=0
        )  # (8, N, C)
        probs_0   = all_probs[0]
        probs_d4  = all_probs.mean(0)

    pred_0  = probs_0.argmax(1)
    pred_d4 = probs_d4.argmax(1)

    # entropy
    ent_0   = _entropy(probs_0)
    ent_d4  = _entropy(probs_d4)

    # epistemic uncertainty: std of softmax across views
    if is_knn:
        epist = np.zeros(N)
    else:
        epist = all_probs.std(0).mean(1)   # mean std across classes

    # corrected / corrupted
    n_corrected = int(((pred_0 != test_lbl) & (pred_d4 == test_lbl)).sum())
    n_corrupted = int(((pred_0 == test_lbl) & (pred_d4 != test_lbl)).sum())

    # --- aggregate records ---
    records = []
    base_args = dict(strategy="none", aggregation="mean",
                     balanced_acc=_balanced_acc(test_lbl, pred_0),
                     acc=float(accuracy_score(test_lbl, pred_0)),
                     macro_f1=float(f1_score(test_lbl, pred_0, average="macro", zero_division=0)),
                     n_corrected=None, n_corrupted=None)
    records.append(base_args)

    def _add(strategy, aggregation, preds):
        nc = int(((pred_0 != test_lbl) & (preds == test_lbl)).sum())
        ncorr = int(((pred_0 == test_lbl) & (preds != test_lbl)).sum())
        records.append(dict(
            strategy=strategy, aggregation=aggregation,
            balanced_acc=_balanced_acc(test_lbl, preds),
            acc=float(accuracy_score(test_lbl, preds)),
            macro_f1=float(f1_score(test_lbl, preds, average="macro", zero_division=0)),
            n_corrected=nc, n_corrupted=ncorr,
        ))

    # V4 (Klein four-group) subset of the cached 8 D4 views:
    # {orig, rot180, hflip, hflip+rot180} = indices {0, 2, 4, 6}
    # (see d4_ops ordering in data/transforms.py); pure re-aggregation, no
    # extra inference. kNN omitted (per-view probs not computed for kNN).
    V4_VIEWS = [0, 2, 4, 6]

    if is_knn:
        _add("d4", "mean", pred_d4)
    else:
        # mean aggregation (arithmetic mean of softmax probabilities)
        _add("d4", "mean", pred_d4)
        # V4 (4-view flips) mean aggregation
        probs_v4 = all_probs[V4_VIEWS].mean(0)
        _add("flips", "mean", probs_v4.argmax(1))
        # logit_mean: average pre-softmax scores then argmax. For a linear head
        # this equals classifying the mean embedding (zero Jensen gap); it is the
        # geometric-mean / logit-space ensemble alternative to arithmetic `mean`.
        all_scores = np.stack(
            [head_scores(head_type, head, test_emb[:, v, :]) for v in range(V)],
            axis=0,
        )  # (8, N, C)
        logit_mean_pred = all_scores.mean(0).argmax(1)
        _add("d4", "logit_mean", logit_mean_pred)
        # vote aggregation
        votes = np.stack([all_probs[v].argmax(1) for v in range(V)], axis=1)  # (N, 8)
        vote_pred = np.array([np.bincount(row, minlength=len(classes)).argmax()
                              for row in votes])
        _add("d4", "vote", vote_pred)
        # confidence: pick view with highest max-prob
        max_conf = all_probs.max(2)  # (8, N)
        best_view = max_conf.argmax(0)  # (N,)
        conf_pred = all_probs[best_view, np.arange(N)].argmax(1)
        _add("d4", "confidence", conf_pred)

    # --- per-sample DataFrame ---
    class_names = np.array(classes)
    df = pd.DataFrame({
        "sample_id":       test_ids,
        "true_label":      test_lbl,
        "true_label_name": class_names[test_lbl],
        "pred_0":          pred_0,
        "pred_d4":         pred_d4,
        "entropy_0":       ent_0,
        "entropy_d4":      ent_d4,
        "epistemic_unc":   epist,
        "conf_0":          probs_0.max(1),
        "conf_d4":         probs_d4.max(1),
        "correct_0":       (pred_0  == test_lbl),
        "correct_d4":      (pred_d4 == test_lbl),
    })

    return records, df


# ── checkpoint save / load ───────────────────────────────────────────────────

def save_checkpoint(head_type: str, head, dataset: str, model: str, seed: int):
    CKPT_BASE.mkdir(parents=True, exist_ok=True)
    path = CKPT_BASE / f"{dataset}_{model}_{head_type}_seed{seed}.pt"
    if head_type in HIDDEN_LAYERS:           # torch heads → state_dict
        torch.save(head.state_dict(), path)
    else:                                    # kNN → pickle
        import pickle
        with open(path, "wb") as f:
            pickle.dump(head, f)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    global EMB_BASE, TRAIN_CAP
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",     required=True)
    ap.add_argument("--dataset",   required=True)
    ap.add_argument("--head_type", required=True,
                    choices=["linear", "mlp_1h", "mlp_2h", "knn_5", "knn_10"])
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--emb_dir",   default=str(EMB_BASE))
    ap.add_argument("--out_dir",   default=str(OUT_DIR))
    ap.add_argument("--train_cap", type=int, default=TRAIN_CAP,
                    help="Max stratified training samples for the final head fit "
                         "(default no cap; lower it to subsample).")
    ap.add_argument("--force", action="store_true",
                    help="Recompute and overwrite even if output JSON/parquet exist.")
    args = ap.parse_args()

    TRAIN_CAP = args.train_cap
    EMB_BASE = Path(args.emb_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{args.dataset}_{args.model}_{args.head_type}_seed{args.seed}"
    json_path    = out_dir / f"probe_{tag}.json"
    parquet_path = out_dir / f"probe_persample_{tag}.parquet"

    if json_path.exists() and parquet_path.exists() and not args.force:
        print(f"Already done: {tag}")
        return

    set_seed(args.seed)
    print(f"Model={args.model}  Dataset={args.dataset}  "
          f"Head={args.head_type}  Seed={args.seed}")

    # Load embeddings
    X_train, y_train, _,        classes = load_split(args.model, args.dataset, "train")
    X_test,  y_test,  test_ids, _       = load_split(args.model, args.dataset, "test")
    n_classes = len(classes)
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}  Classes: {n_classes}")

    # lr/wd are selected once (deterministic CV) and cached, then reused across
    # run seeds so error bars reflect init variance, not HP wobble. Applies to
    # all torch heads (linear / mlp_1h / mlp_2h).
    mlp_hp = None
    if args.head_type in HIDDEN_LAYERS:
        CKPT_BASE.mkdir(parents=True, exist_ok=True)
        hp_path = CKPT_BASE / f"hp_{args.dataset}_{args.model}_{args.head_type}.json"
        if hp_path.exists():
            cached = json.loads(hp_path.read_text())
            mlp_hp = (cached["lr"], cached["wd"])
            print(f"  Loaded cached hp: {mlp_hp}")

    # Train
    print(f"  Training {args.head_type}...")
    head, hp = train_head(args.head_type, X_train, y_train, n_classes,
                          args.seed, mlp_hp=mlp_hp)
    print(f"  Selected hyperparameters: {hp}")
    save_checkpoint(args.head_type, head, args.dataset, args.model, args.seed)
    print(f"  Checkpoint saved.")

    # Cache selected lr/wd for reuse by other seeds (write once, deterministic).
    if args.head_type in HIDDEN_LAYERS and mlp_hp is None:
        hp_path.write_text(json.dumps(hp))

    # Evaluate
    print(f"  Evaluating...")
    records, per_sample_df = evaluate(args.head_type, head,
                                      X_test, y_test, test_ids, classes)

    # Annotate records with metadata + selected hyperparameters
    meta = dict(model=args.model, dataset=args.dataset, backbone_mode="frozen",
                head_type=args.head_type, seed=args.seed)
    for r in records:
        r.update(meta)
        r["hp"] = hp

    # Add metadata columns to per-sample frame (robust downstream parsing —
    # model / head_type names contain underscores, so don't parse filenames).
    per_sample_df.insert(0, "model",      args.model)
    per_sample_df.insert(1, "dataset",    args.dataset)
    per_sample_df.insert(2, "head_type",  args.head_type)
    per_sample_df.insert(3, "seed",       args.seed)

    # Write outputs
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    per_sample_df.to_parquet(parquet_path, index=False)

    best = max(records, key=lambda r: r["balanced_acc"])
    print(f"  Best: strategy={best['strategy']} agg={best['aggregation']} "
          f"bal_acc={best['balanced_acc']:.4f}")
    print(f"  Saved: {json_path.name}  {parquet_path.name}")


if __name__ == "__main__":
    main()
