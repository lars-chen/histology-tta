"""
Build calibration outputs from probe per-sample parquets.

Outputs
-------
results/calibration/per_sample/{model}_{dataset}.parquet
    All seeds concatenated. Schema: model, dataset, seed, sample_id,
    true_label, true_label_name, pred_0, pred_d4, entropy_0, entropy_d4,
    epistemic_unc, conf_0, conf_d4, correct_0, correct_d4.

results/calibration_aggregate.csv
    Binned reliability stats per (model, dataset, seed, entropy_bin).
"""
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import brier_score_loss

RAW    = Path("results/raw")
CAL_PS = Path("results/calibration/per_sample")
RES    = Path("results")
N_BINS = 10


def ece(probs_max: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected Calibration Error from confidence and correctness arrays."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    N = len(probs_max)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs_max >= lo) & ((probs_max < hi) | (hi == 1.0))  # last bin includes 1.0
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = probs_max[mask].mean()
        ece_val += (mask.sum() / N) * abs(acc_bin - conf_bin)
    return float(ece_val)


def main():
    CAL_PS.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # 1. Collect all per-sample parquets, group by (model, dataset)
    # -------------------------------------------------------------------
    files = sorted(glob.glob(str(RAW / "probe_persample_*.parquet")))
    if not files:
        print("No per-sample parquets found — run train_probe.py first.")
        return

    # Group parquets by (model, dataset, head_type), read from columns
    # (model / head names contain underscores → don't parse filenames).
    groups: dict[tuple, list[pd.DataFrame]] = {}
    for f in files:
        df = pd.read_parquet(f)
        if "head_type" not in df.columns:
            continue
        key = (df["model"].iloc[0], df["dataset"].iloc[0], df["head_type"].iloc[0])
        groups.setdefault(key, []).append(df)

    # -------------------------------------------------------------------
    # 2. Per-sample parquets (linear head only, all seeds concatenated)
    # -------------------------------------------------------------------
    agg_rows = []

    for (model, dataset, head_type), seed_dfs in sorted(groups.items()):
        if head_type != "linear":
            continue

        combined = pd.concat(seed_dfs, ignore_index=True)
        out_path = CAL_PS / f"{model}_{dataset}.parquet"
        combined.to_parquet(out_path, index=False)
        print(f"  Wrote {out_path.name}  ({len(combined):,} rows)")

        # -------------------------------------------------------------------
        # 3. Aggregate — per (model, dataset, seed, entropy_bin)
        # -------------------------------------------------------------------
        bins = np.linspace(0, 1, N_BINS + 1)
        bin_labels = [f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(bins[:-1], bins[1:])]

        for seed, sdf in combined.groupby("seed"):
            ent   = sdf["entropy_0"].values
            corr0 = sdf["correct_0"].values.astype(float)
            corrd = sdf["correct_d4"].values.astype(float)
            conf0 = sdf["conf_0"].values

            ece_base = ece(conf0, corr0.astype(bool))
            ece_d4   = ece(sdf["conf_d4"].values, corrd.astype(bool))
            y_true   = sdf["true_label"].values

            for b_idx, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
                mask = (ent >= lo) & ((ent < hi) | (hi == bins[-1]))  # last bin includes right edge
                if mask.sum() == 0:
                    continue
                agg_rows.append({
                    "model": model, "dataset": dataset, "seed": seed,
                    "entropy_bin": bin_labels[b_idx],
                    "n_samples": int(mask.sum()),
                    "baseline_acc": float(corr0[mask].mean()),
                    "tta_acc":      float(corrd[mask].mean()),
                    "ece":          ece_base,   # overall ECE, same for all bins
                    "ece_d4":       ece_d4,
                    "brier_score":  float(brier_score_loss(
                        corr0[mask].astype(int), conf0[mask])),
                })

    agg_df = pd.DataFrame(agg_rows)
    agg_path = RES / "calibration_aggregate.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"\nWrote {agg_path}  ({len(agg_df):,} rows)")


if __name__ == "__main__":
    main()
