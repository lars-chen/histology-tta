"""Compile selective TTA pareto curves for mlp_1h head.
Reads probe_persample_*.parquet files (head_type==mlp_1h) and writes
results/selective_tta_results_mlp1h.csv — does NOT touch any existing CSV.
"""
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score

RAW        = Path("results/raw")
THRESHOLDS = np.linspace(0, 1, 21)
HEAD       = "mlp_1h"

sel_rows = []

for f in sorted(glob.glob(str(RAW / "probe_persample_*.parquet"))):
    df_ps = pd.read_parquet(f)
    if "head_type" not in df_ps.columns or df_ps["head_type"].iloc[0] != HEAD:
        continue

    dataset_name = df_ps["dataset"].iloc[0]
    model_name   = df_ps["model"].iloc[0]
    seed_val     = int(df_ps["seed"].iloc[0])

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
            "model":    model_name,
            "dataset":  dataset_name,
            "head_type": HEAD,
            "seed":     seed_val,
            "threshold":               round(float(t), 4),
            "coverage_pct":            round(coverage, 2),
            "baseline_balanced_acc":   round(base_acc, 6),
            "d4_balanced_acc":         round(d4_acc, 6),
            "selective_balanced_acc":  round(sel_acc, 6),
            "delta_vs_baseline":       round(sel_acc - base_acc, 6),
            "pct_of_full_benefit":     (round((sel_acc - base_acc) / d4_delta * 100, 2)
                                        if abs(d4_delta) > 1e-9 else 0.0),
        })

out = Path("results/selective_tta_results_mlp1h.csv")
sel_df = pd.DataFrame(sel_rows)
sel_df.to_csv(out, index=False)
print(f"Wrote {len(sel_df)} rows → {out}")
print(f"  models: {sorted(sel_df['model'].unique()) if not sel_df.empty else '(none)'}")
print(f"  datasets: {sorted(sel_df['dataset'].unique()) if not sel_df.empty else '(none)'}")
