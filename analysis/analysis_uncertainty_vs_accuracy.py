"""
Analyze whether per-sample entropy and epistemic uncertainty predict incorrect predictions.
Also computes per-model agreement_rate from existing persample parquets.
"""

import glob
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

RAW_DIR = Path("results/raw")
OUT_DIR = Path("results")

FM_MODELS = {
    "ctranspath", "phikon", "phikon2", "uni", "uni2",
    "virchow", "virchow2", "gigapath", "hoptimus"
}


def load_persample_linear(datasets=None):
    """Load all linear-probe persample parquets into one DataFrame."""
    files = sorted(RAW_DIR.glob("probe_persample_*_linear_seed*.parquet"))
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    if datasets:
        df = df[df["dataset"].isin(datasets)]
    df["model_type"] = df["model"].apply(
        lambda m: "histology_fm" if m in FM_MODELS else "general"
    )
    # normalize entropy by log(C) so it's in [0,1] regardless of n_classes
    n_classes = df.groupby("dataset")["true_label"].transform(lambda x: x.nunique())
    df["entropy_0_norm"] = df["entropy_0"] / np.log(n_classes)
    df["flipped"] = df["pred_0"] != df["pred_d4"]   # TTA changed the prediction
    return df


def auroc_binary(scores, labels):
    """AUROC of scores for predicting labels=1 (incorrect)."""
    from sklearn.metrics import roc_auc_score
    if labels.nunique() < 2:
        return float("nan")
    return roc_auc_score(labels.astype(int), scores)


def point_biserial(scores, binary_labels):
    """Point-biserial correlation (= Pearson for continuous vs binary)."""
    r, p = stats.pointbiserialr(binary_labels.astype(int), scores)
    return r, p


def analyze_entropy_vs_error(df):
    """
    For each model x dataset: does baseline entropy predict incorrect predictions?
    Returns a summary DataFrame with AUROC and point-biserial r.
    """
    rows = []
    for (model, dataset, model_type), g in df.groupby(["model", "dataset", "model_type"]):
        incorrect = ~g["correct_0"]
        if incorrect.sum() < 10:
            continue
        auc_entropy   = auroc_binary(g["entropy_0_norm"], incorrect)
        auc_epistemic = auroc_binary(g["epistemic_unc"],  incorrect)
        r_entropy,  p_entropy  = point_biserial(g["entropy_0_norm"], incorrect)
        r_epistemic, p_epistemic = point_biserial(g["epistemic_unc"], incorrect)

        # agreement between base and TTA view (proxy for view stability)
        agree_rate = (~g["flipped"]).mean()

        # mean entropy per outcome bin
        mean_entropy_correct   = g.loc[g["correct_0"],  "entropy_0_norm"].mean()
        mean_entropy_incorrect = g.loc[~g["correct_0"], "entropy_0_norm"].mean()

        rows.append({
            "model": model,
            "dataset": dataset,
            "model_type": model_type,
            "n": len(g),
            "error_rate": incorrect.mean(),
            "agree_rate_base_vs_d4": agree_rate,
            "auroc_entropy": auc_entropy,
            "auroc_epistemic": auc_epistemic,
            "r_entropy": r_entropy,
            "p_entropy": p_entropy,
            "r_epistemic": r_epistemic,
            "p_epistemic": p_epistemic,
            "mean_entropy_correct": mean_entropy_correct,
            "mean_entropy_incorrect": mean_entropy_incorrect,
            "entropy_ratio_incorrect_correct": mean_entropy_incorrect / (mean_entropy_correct + 1e-12),
        })
    return pd.DataFrame(rows)


def analyze_uncertainty_vs_flip(df):
    """
    Does entropy predict whether TTA flips the prediction (disagreement)?
    """
    rows = []
    for (model, dataset, model_type), g in df.groupby(["model", "dataset", "model_type"]):
        if g["flipped"].sum() < 5:
            continue
        auc = auroc_binary(g["entropy_0_norm"], g["flipped"])
        r, p = point_biserial(g["entropy_0_norm"], g["flipped"])
        rows.append({
            "model": model, "dataset": dataset, "model_type": model_type,
            "flip_rate": g["flipped"].mean(),
            "auroc_entropy_predicts_flip": auc,
            "r_entropy_flip": r, "p_entropy_flip": p,
        })
    return pd.DataFrame(rows)


def compute_agreement_rate(df):
    """
    Derive agreement_rate per (model, dataset) from the per-sample parquet.
    Here "agreement" = base-view prediction matches D4-aggregated prediction
    (a proxy for full 8-view agreement; full agreement would require all 8 logits).
    """
    agg = (
        df.groupby(["model", "dataset", "model_type"])
        .apply(lambda g: pd.Series({
            "n": len(g),
            "agree_rate_base_d4": (~g["flipped"]).mean(),
            "mean_entropy_0_norm": g["entropy_0_norm"].mean(),
            "mean_epistemic_unc": g["epistemic_unc"].mean(),
            "error_rate": (~g["correct_0"]).mean(),
        }))
        .reset_index()
    )
    return agg


def print_summary(df_err, df_flip, df_agree):
    print("\n" + "="*70)
    print("ENTROPY → ERROR RATE PREDICTION (AUROC, point-biserial r)")
    print("="*70)
    for mt in ["histology_fm", "general"]:
        sub = df_err[df_err["model_type"] == mt]
        print(f"\n  {mt}  (n={len(sub)} model×dataset combos)")
        print(f"  Mean AUROC(entropy→incorrect):   {sub['auroc_entropy'].mean():.3f} ± {sub['auroc_entropy'].std():.3f}")
        print(f"  Mean AUROC(epistemic→incorrect): {sub['auroc_epistemic'].mean():.3f} ± {sub['auroc_epistemic'].std():.3f}")
        print(f"  Mean r(entropy, incorrect):       {sub['r_entropy'].mean():.3f}  (range {sub['r_entropy'].min():.3f}–{sub['r_entropy'].max():.3f})")
        print(f"  Mean entropy ratio (wrong/right): {sub['entropy_ratio_incorrect_correct'].mean():.2f}x")

    print("\n" + "="*70)
    print("ENTROPY → FLIP PREDICTION (does entropy predict TTA disagreement?)")
    print("="*70)
    for mt in ["histology_fm", "general"]:
        sub = df_flip[df_flip["model_type"] == mt]
        print(f"\n  {mt}")
        print(f"  Mean AUROC(entropy→flip): {sub['auroc_entropy_predicts_flip'].mean():.3f}")
        print(f"  Mean r(entropy, flip):    {sub['r_entropy_flip'].mean():.3f}")
        print(f"  Mean flip rate:           {sub['flip_rate'].mean():.3f}")

    print("\n" + "="*70)
    print("AGREEMENT RATE (base vs D4 prediction) BY MODEL FAMILY")
    print("="*70)
    for mt in ["histology_fm", "general"]:
        sub = df_agree[df_agree["model_type"] == mt]
        print(f"\n  {mt}")
        print(f"  Mean agree rate (base==D4): {sub['agree_rate_base_d4'].mean():.4f}")
        print(f"  Range: {sub['agree_rate_base_d4'].min():.4f}–{sub['agree_rate_base_d4'].max():.4f}")
        print(f"  Mean entropy (normalized):  {sub['mean_entropy_0_norm'].mean():.4f}")

    # Cross-model correlation: does mean entropy predict TTA delta?
    # Load canonical results to join
    canon = pd.read_csv("results/canonical_results.csv")
    canon_d4 = canon[(canon["head_type"] == "linear") & (canon["strategy"] == "d4") & (canon["aggregation"] == "mean")].copy()
    # compute delta = d4 balanced_acc - none balanced_acc per model×dataset
    none_acc = canon[(canon["head_type"] == "linear") & (canon["strategy"] == "none")][["model","dataset","seed","balanced_acc"]].rename(columns={"balanced_acc":"base_acc"})
    d4_acc   = canon_d4[["model","dataset","seed","balanced_acc"]]
    delta_df = d4_acc.merge(none_acc, on=["model","dataset","seed"]).assign(delta=lambda x: x["balanced_acc"] - x["base_acc"])
    delta_mean = delta_df.groupby(["model","dataset"])["delta"].mean().reset_index()
    agree_tcga = df_agree[df_agree["dataset"] == "tcga-ut"].copy()
    merged = agree_tcga.merge(delta_mean[delta_mean["dataset"] == "tcga-ut"][["model", "delta"]],
                              on="model", how="inner")
    if len(merged) > 3:
        r, p = stats.pearsonr(merged["mean_entropy_0_norm"], merged["delta"])
        rs, ps = stats.spearmanr(merged["mean_entropy_0_norm"], merged["delta"])
        print(f"\n  Pearson r(mean entropy, TTA delta) on TCGA-UT:  r={r:.3f} p={p:.3f}  (n={len(merged)})")
        print(f"  Spearman ρ(mean entropy, TTA delta) on TCGA-UT: ρ={rs:.3f} p={ps:.3f}")


def main():
    print("Loading persample parquets (linear probe, seed 0)...")
    df = load_persample_linear()
    df = df[df["seed"] == 0]  # one seed for speed; results stable across seeds
    print(f"  Total rows: {len(df):,}  |  models: {df['model'].nunique()}  |  datasets: {df['dataset'].nunique()}")

    print("Computing entropy → error analysis...")
    df_err   = analyze_entropy_vs_error(df)
    print("Computing entropy → flip analysis...")
    df_flip  = analyze_uncertainty_vs_flip(df)
    print("Computing agreement rate summary...")
    df_agree = compute_agreement_rate(df)

    # Save
    df_err.to_csv(OUT_DIR / "uncertainty_vs_error.csv", index=False)
    df_flip.to_csv(OUT_DIR / "uncertainty_vs_flip.csv", index=False)
    df_agree.to_csv(OUT_DIR / "agreement_rate_summary.csv", index=False)
    print(f"\nSaved:\n  results/uncertainty_vs_error.csv\n  results/uncertainty_vs_flip.csv\n  results/agreement_rate_summary.csv")

    print_summary(df_err, df_flip, df_agree)


if __name__ == "__main__":
    main()
