"""
Compile all result JSONs into clean CSVs.

Outputs
-------
results/tta_results.csv          Finetuned backbone results (existing pipeline)
results/tta_per_class.csv        Per-class breakdown for finetuned runs
results/tta_results_ablation.csv Noaug / subset ablation rows
results/canonical_results.csv   All frozen probe + finetuned linear results
results/selective_tta_results.csv Selective TTA pareto data (linear head only)
"""
import json, glob, re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score

RAW = Path("results/raw")

HISTOLOGY  = {'gigapath', 'hoptimus', 'phikon', 'phikon2', 'uni', 'uni2',
              'virchow', 'virchow2', 'ctranspath'}
EQUIVARIANT = {'d4wrn'}

# ---------------------------------------------------------------------------
# 1. Finetuned tta_results_*.json  →  tta_results.csv / tta_per_class.csv
# ---------------------------------------------------------------------------

rows, pc_rows = [], []

for f in sorted(glob.glob(str(RAW / "tta_results_*.json"))):
    with open(f) as fh:
        data = json.load(fh)
    for entry in data:
        model      = entry["model"]
        model_type = ("histology" if model in HISTOLOGY
                      else "equivariant" if model in EQUIVARIANT else "general")
        ckpt       = entry.get("checkpoint", "")
        m          = re.search(r"_sub(\d+)_", ckpt)
        train_subset = int(m.group(1)) if m else None

        row = {
            "model": model, "dataset": entry["dataset"],
            "backbone_mode": entry["backbone_mode"],
            "train_augment": entry["train_augment"],
            "seed": entry["seed"], "model_type": model_type,
            "mlp_hidden": (f"mlp{int(entry['mlp_hidden'])}"
                           if entry.get("mlp_hidden") else "linear"),
            "strategy": entry["strategy"], "aggregation": entry["aggregation"],
            "train_subset": train_subset,
            "n_test": entry.get("n_total"),
            "acc": entry["acc"], "balanced_acc": entry["balanced_acc"],
            "n_correct": entry.get("n_correct"), "n_wrong": entry.get("n_wrong"),
            "n_corrected": entry.get("n_corrected"),
            "n_corrupted": entry.get("n_corrupted"),
            "total_unc": entry.get("total_unc"),
            "aleatoric_unc": entry.get("aleatoric_unc"),
            "epistemic_unc": entry.get("epistemic_unc"),
            "agreement_rate": entry.get("agreement_rate"),
            "ece": entry.get("ece"),
        }
        rows.append(row)

        if entry.get("per_class"):
            for cls_data in entry["per_class"]:
                pc_rows.append({
                    "model": model, "dataset": entry["dataset"],
                    "backbone_mode": entry["backbone_mode"],
                    "train_augment": entry["train_augment"],
                    "seed": entry["seed"], "model_type": model_type,
                    "mlp_hidden": row["mlp_hidden"],
                    "strategy": entry["strategy"],
                    "aggregation": entry["aggregation"],
                    "train_subset": train_subset,
                    "class_name": cls_data.get("class"),
                    "precision": cls_data.get("precision"),
                    "recall": cls_data.get("recall"),
                    "f1": cls_data.get("f1"),
                    "support": cls_data.get("support"),
                })

df    = pd.DataFrame(rows)
df_pc = pd.DataFrame(pc_rows)

if not df.empty:
    canon     = (df.train_augment == True)  & (df.train_subset.isna())
    canon_pc  = (df_pc.train_augment == True) & (df_pc.train_subset.isna())
    df[canon].to_csv("results/tta_results.csv", index=False)
    df_pc[canon_pc].to_csv("results/tta_per_class.csv", index=False)
    df[~canon].to_csv("results/tta_results_ablation.csv", index=False)
    df_pc[~canon_pc].to_csv("results/tta_per_class_ablation.csv", index=False)
    print(f"tta_results      — canonical: {canon.sum():>5}  ablation: {(~canon).sum():>5}")
else:
    print("tta_results      — no finetuned JSON files found")

# ---------------------------------------------------------------------------
# 2. probe_*.json  →  canonical_results.csv  (frozen probe results)
# ---------------------------------------------------------------------------

probe_rows = []

for f in sorted(glob.glob(str(RAW / "probe_*.json"))):
    if "persample" in f:
        continue
    with open(f) as fh:
        data = json.load(fh)
    for entry in data:
        model      = entry["model"]
        model_type = ("histology" if model in HISTOLOGY
                      else "equivariant" if model in EQUIVARIANT else "general")
        probe_rows.append({
            "model": model, "dataset": entry["dataset"],
            "backbone_mode": entry["backbone_mode"],
            "seed": entry["seed"], "model_type": model_type,
            "head_type": entry["head_type"],
            "strategy": entry["strategy"], "aggregation": entry["aggregation"],
            "balanced_acc": entry["balanced_acc"],
            "acc": entry["acc"], "macro_f1": entry["macro_f1"],
            "n_corrected": entry.get("n_corrected"),
            "n_corrupted": entry.get("n_corrupted"),
        })

# Add finetuned linear rows from tta_results if available
if not df.empty:
    ft = df[(df.backbone_mode == "finetuned") & (df.train_augment == True)
            & (df.train_subset.isna())].copy()
    ft["head_type"] = ft["mlp_hidden"]
    for _, row in ft.iterrows():
        probe_rows.append({
            "model": row.model, "dataset": row.dataset,
            "backbone_mode": row.backbone_mode,
            "seed": row.seed, "model_type": row.model_type,
            "head_type": row.head_type,
            "strategy": row.strategy, "aggregation": row.aggregation,
            "balanced_acc": row.balanced_acc, "acc": row.acc,
            "macro_f1": None,
            "n_corrected": row.n_corrected, "n_corrupted": row.n_corrupted,
        })

canonical = pd.DataFrame(probe_rows)
canonical.to_csv("results/canonical_results.csv", index=False)
print(f"canonical_results — {len(canonical):>6} rows")

# ---------------------------------------------------------------------------
# 3. probe_persample_*_{linear}_*.parquet  →  selective_tta_results.csv
# ---------------------------------------------------------------------------

THRESHOLDS = np.linspace(0, 1, 21)  # 0.00, 0.05, ..., 1.00
sel_rows = []

for f in sorted(glob.glob(str(RAW / "probe_persample_*.parquet"))):
    df_ps = pd.read_parquet(f)
    # Read identity from columns (model / head names contain underscores, so
    # filenames are not safely parseable). Linear head only.
    if "head_type" not in df_ps.columns or df_ps["head_type"].iloc[0] != "linear":
        continue
    dataset_name = df_ps["dataset"].iloc[0]
    model_name   = df_ps["model"].iloc[0]
    seed_str     = int(df_ps["seed"].iloc[0])

    y_true  = df_ps["true_label"].values
    ent     = df_ps["entropy_0"].values
    pred_0  = df_ps["pred_0"].values
    pred_d4 = df_ps["pred_d4"].values
    base_acc = balanced_accuracy_score(y_true, pred_0)
    d4_acc   = balanced_accuracy_score(y_true, pred_d4)
    d4_delta = d4_acc - base_acc

    for t in THRESHOLDS:
        mask     = ent > t
        coverage = float(mask.mean() * 100)
        sel_pred = np.where(mask, pred_d4, pred_0)
        sel_acc  = balanced_accuracy_score(y_true, sel_pred)
        sel_rows.append({
            "model": model_name, "dataset": dataset_name,
            "head_type": "linear", "seed": int(seed_str),
            "threshold": round(float(t), 4),
            "coverage_pct": round(coverage, 2),
            "baseline_balanced_acc": round(base_acc, 6),
            "d4_balanced_acc": round(d4_acc, 6),
            "selective_balanced_acc": round(sel_acc, 6),
            "delta_vs_baseline": round(sel_acc - base_acc, 6),
            "pct_of_full_benefit": (round((sel_acc - base_acc) / d4_delta * 100, 2)
                                    if abs(d4_delta) > 1e-9 else 0.0),
        })

sel_df = pd.DataFrame(sel_rows)
sel_df.to_csv("results/selective_tta_results.csv", index=False)
print(f"selective_tta    — {len(sel_df):>6} rows  "
      f"({sel_df['model'].nunique() if not sel_df.empty else 0} models)")
