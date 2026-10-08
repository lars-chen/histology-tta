"""
Pure-inference V4 (flips) re-aggregation.

The cached test embeddings store all 8 D4 views per patch (N, 8, D). The V4
(Klein four-group) subset is views {0:orig, 2:rot180, 4:hflip, 6:hflip+rot180}
(see d4_ops ordering in data/transforms.py). For each saved linear-probe
checkpoint we load the head, push the relevant cached views through it, and
average softmax probabilities — no retraining, no GPU, no overwrite of any
existing results file.

Output: results/v4_flips_results.csv with one row per (model, dataset, seed)
for strategy `flips` (mean aggregation), plus recomputed `none` / `d4` columns
for a reproduction sanity-check against canonical_results.csv.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

EMB_BASE  = Path(os.environ.get("HISTO_EMB_DIR", "embeddings"))
CKPT_BASE = Path("checkpoints/frozen")
OUT_CSV   = Path("results/v4_flips_results.csv")

MODELS = ["convnextv2_base", "convnextv2_tiny", "ctranspath", "dinov2_b",
          "dinov2_s", "gigapath", "hoptimus", "phikon", "phikon2",
          "resnet18", "resnet50", "uni", "uni2", "virchow", "virchow2"]
DATASETS = ["tcga-ut", "nct-crc-100k", "nct-crc-nonorm", "mhist"]
SEEDS = [0, 1, 2, 3, 42]
V4_VIEWS = [0, 2, 4, 6]   # orig, rot180, hflip, hflip+rot180


class MLPHead(nn.Module):
    """hidden_layers=0 → bare nn.Linear, matching train_probe.MLPHead."""
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, n_classes))

    def forward(self, x):
        return self.net(x)


def load_head(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    w = sd["net.0.weight"]                 # (n_classes, in_dim)
    n_classes, in_dim = w.shape
    head = MLPHead(in_dim, n_classes)
    head.load_state_dict(sd)
    head.eval()
    return head


@torch.no_grad()
def probs_for_view(head, emb_v):
    logits = head(torch.from_numpy(np.ascontiguousarray(emb_v))).numpy()
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def metrics(y, pred):
    return (balanced_accuracy_score(y, pred),
            accuracy_score(y, pred),
            f1_score(y, pred, average="macro", zero_division=0))


rows = []
for model in MODELS:
    for dataset in DATASETS:
        test_dir = EMB_BASE / model / dataset / "test"
        if not test_dir.exists():
            print(f"  skip (no emb): {model}/{dataset}")
            continue
        emb = np.load(test_dir / "embeddings.npy", mmap_mode="r").astype(np.float32)
        lbl = np.load(test_dir / "labels.npy").astype(np.int64)
        for seed in SEEDS:
            ckpt = CKPT_BASE / f"{dataset}_{model}_linear_seed{seed}.pt"
            if not ckpt.exists():
                print(f"  skip (no ckpt): {ckpt.name}")
                continue
            head = load_head(ckpt)
            # per-view probs for the union we need (V4 ∪ all 8 for sanity)
            all_probs = np.stack([probs_for_view(head, emb[:, v, :])
                                  for v in range(emb.shape[1])], axis=0)  # (8,N,C)
            pred_none = all_probs[0].argmax(1)
            pred_d4   = all_probs.mean(0).argmax(1)
            pred_v4   = all_probs[V4_VIEWS].mean(0).argmax(1)
            for strat, pred in [("none", pred_none), ("flips", pred_v4), ("d4", pred_d4)]:
                ba, acc, mf1 = metrics(lbl, pred)
                nc = int(((pred_none != lbl) & (pred == lbl)).sum())
                ncorr = int(((pred_none == lbl) & (pred != lbl)).sum())
                rows.append(dict(model=model, dataset=dataset, backbone_mode="frozen",
                                 seed=seed, head_type="linear", strategy=strat,
                                 aggregation="mean", balanced_acc=ba, acc=acc,
                                 macro_f1=mf1, n_corrected=nc, n_corrupted=ncorr))
            print(f"  {model:16s} {dataset:14s} seed{seed:<2d}  "
                  f"none={metrics(lbl,pred_none)[0]:.4f} "
                  f"v4={metrics(lbl,pred_v4)[0]:.4f} d4={metrics(lbl,pred_d4)[0]:.4f}")

df = pd.DataFrame(rows)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"\nWrote {len(df)} rows → {OUT_CSV}")
