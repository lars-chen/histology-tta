"""Generate LaTeX tables directly from canonical_results.csv — no manual transcription."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

RESULTS_CSV = Path("results/canonical_results.csv")
V4_CSV = Path("results/v4_flips_results.csv")  # flips re-aggregation for the 15 cached models
OUT_DIR = Path("paper")

FM_HIST = {
    "phikon", "phikon2", "uni", "uni2",
    "virchow", "virchow2", "gigapath", "hoptimus", "ctranspath",
}

MODEL_LABELS = {
    "resnet18":        "ResNet-18",
    "resnet50":        "ResNet-50",
    "convnextv2_tiny": "ConvNeXtV2-Tiny",
    "convnextv2_base": "ConvNeXtV2-Base",
    "dinov2_s":        "DINOv2-S",
    "dinov2_b":        "DINOv2-B",
    "ctranspath":      "CTransPath",
    "phikon":          "Phikon",
    "phikon2":         "Phikon-v2",
    "uni":             "UNI",
    "uni2":            "UNI-v2",
    "virchow":         "Virchow",
    "virchow2":        "Virchow-v2",
    "gigapath":        "GigaPath",
    "hoptimus":        "H-Optimus-1",
    "d4wrn":           r"D4WRN$^\dagger$",
}

DATASET_LABELS = {
    "tcga-ut":        r"TCGA-UT (31 cls)",
    "nct-crc-100k":   r"NCT-CRC-100K (9 cls)",
    "nct-crc-nonorm": r"NCT-CRC-NoNorm (9 cls)",
    "mhist":          r"MHIST (2 cls)",
}

# Compact dataset headers for the wide per-model table.
DATASET_LABELS_SHORT = {
    "tcga-ut":        r"TCGA-UT",
    "nct-crc-100k":   r"NCT-100K",
    "nct-crc-nonorm": r"NCT-NoNorm",
    "mhist":          r"MHIST",
}

MODEL_ORDER_GENERAL  = ["resnet18", "resnet50", "convnextv2_tiny", "convnextv2_base", "dinov2_s", "dinov2_b"]
MODEL_ORDER_HISTO    = ["ctranspath", "phikon", "phikon2", "uni", "uni2", "virchow", "virchow2", "gigapath", "hoptimus"]
DATASET_ORDER        = ["tcga-ut", "nct-crc-100k", "nct-crc-nonorm", "mhist"]
HEAD_ORDER           = ["knn_10", "linear", "mlp_1h", "mlp_2h"]
HEAD_LABELS          = {"knn_10": r"kNN ($k$=10)", "linear": "Linear", "mlp_1h": "MLP-1h", "mlp_2h": "MLP-2h"}


def _sig_star(deltas: np.ndarray) -> str:
    """Two-sided one-sample t-test of deltas vs 0; return significance star(s)."""
    if len(deltas) < 2:
        return ""
    _, p = scipy_stats.ttest_1samp(deltas, 0)
    if p < 0.001: return r"^{***}"
    if p < 0.01:  return r"^{**}"
    if p < 0.05:  return r"^{*}"
    return ""


def _fmt_delta(v: float, star: str = "", sem: float | None = None) -> str:
    """Format a delta value uniformly in math mode: $+X.XX^{***}$."""
    sign = "+" if v >= -0.005 else "-"
    body = f"{abs(v):.2f}"
    inner = f"{sign}{body}{star}"
    if sem is not None and not np.isnan(sem):
        inner += rf"{{\,\scriptsize\pm {sem:.2f}}}"
    return rf"${inner}$"


def _load_deltas(backbone_mode: str = "frozen", aggregation: str = "mean",
                 extra_models: dict[str, str] | None = None) -> pd.DataFrame:
    """Return DataFrame with columns: model, dataset, head_type, seed, base_pct, delta_pp.

    extra_models: {model_name: backbone_mode} to pull in additional models with a
    different backbone_mode (e.g. d4wrn which is always finetuned).
    """
    df = pd.read_csv(RESULTS_CSV)
    main = df[(df.backbone_mode == backbone_mode) & (df.aggregation == aggregation)]

    extra_frames = []
    if extra_models:
        for mdl, bm in extra_models.items():
            extra_frames.append(
                df[(df.model == mdl) & (df.backbone_mode == bm) & (df.aggregation == aggregation)]
            )
    combined = pd.concat([main] + extra_frames, ignore_index=True) if extra_frames else main

    base = (combined[combined.strategy == "none"]
            [["model", "dataset", "head_type", "seed", "balanced_acc"]]
            .rename(columns={"balanced_acc": "acc_base"}))
    d4   = (combined[combined.strategy == "d4"]
            [["model", "dataset", "head_type", "seed", "balanced_acc"]]
            .rename(columns={"balanced_acc": "acc_d4"}))
    m = base.merge(d4, on=["model", "dataset", "head_type", "seed"])
    m["base_pct"] = m["acc_base"] * 100
    m["delta_pp"] = (m["acc_d4"] - m["acc_base"]) * 100

    # V4 (flips) delta: sourced from v4_flips_results.csv for the 15 cached
    # models and from canonical (`flips` strategy) for d4wrn, which is finetuned
    # and went through the original eval pipeline.
    flips_sources = []
    if V4_CSV.exists():
        flips_sources.append(pd.read_csv(V4_CSV))
    flips_sources.append(combined[combined.strategy == "flips"])
    flips = pd.concat(flips_sources, ignore_index=True)
    flips = flips[(flips.strategy == "flips") & (flips.aggregation == aggregation)]
    v4 = (flips[["model", "dataset", "head_type", "seed", "balanced_acc"]]
          .drop_duplicates(subset=["model", "dataset", "head_type", "seed"])
          .rename(columns={"balanced_acc": "acc_v4"}))
    m = m.merge(v4, on=["model", "dataset", "head_type", "seed"], how="left")
    m["delta_v4_pp"] = (m["acc_v4"] - m["acc_base"]) * 100
    return m


# ── Table 1: full per-model × per-dataset results (linear head) ──────────────

def generate_full_results_table(out_path: Path) -> None:
    m = _load_deltas(extra_models={"d4wrn": "finetuned"})
    m = m[m.head_type == "linear"]

    stats = (m.groupby(["model", "dataset"])
               .agg(base=("base_pct", "mean"),
                    delta_v4=("delta_v4_pp", "mean"),
                    delta=("delta_pp", "mean"),
                    delta_sem=("delta_pp", lambda x: x.sem() if len(x) > 1 else float("nan")))
               .round(3))

    # Per-seed deltas for significance testing (list of arrays, indexed by (model, dataset))
    per_seed_deltas = {
        key: grp["delta_pp"].values
        for key, grp in m.groupby(["model", "dataset"])
    }

    # best delta per dataset (excluding D4WRN as reference-only)
    main_models = set(MODEL_ORDER_GENERAL) | set(MODEL_ORDER_HISTO)
    best = {}
    for ds in DATASET_ORDER:
        sub = stats.xs(ds, level="dataset")
        sub_main = sub[sub.index.isin(main_models)]
        best[ds] = sub_main["delta"].idxmax()

    lines: list[str] = []
    a = lines.append

    a(r"% Auto-generated by utils/generate_tables.py — do not edit by hand.")
    a(r"\begin{table*}[t]")
    a(r"\centering")
    a(r"\caption{Per-model TTA results across all four datasets (frozen backbone / SGD linear probe,")
    a(r"probability-mean aggregation, 5 seeds). \textit{Base}: no-TTA balanced accuracy (\%).")
    a(r"$V_4$ and $D_4$: change in percentage points from 4-view (flips) and 8-view (full dihedral)")
    a(r"TTA respectively. Significance of $\Delta_{\text{TTA}}$ ($D_4$) vs.\ zero (two-sided paired $t$-test, $n=5$ seeds,")
    a(r"uncorrected): $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.")
    a(r"Best $\Delta_{\text{TTA}}$ ($D_4$) per dataset in \textbf{bold}. Class counts: TCGA-UT 31, NCT-100K / NCT-NoNorm 9, MHIST 2.")
    a(r"D4WRN is finetuned (seeds vary by dataset; see $^\dagger$) and shown as an")
    a(r"uncontrolled equivariant reference (see Section~\ref{sec:d4wrn_caveats}).}")
    a(r"\label{tab:full_results}")
    a(r"\vspace{4pt}")
    a(r"\small")
    a(r"\setlength{\tabcolsep}{4pt}")
    a(r"\begin{tabular}{@{}ll" + "ccc" * len(DATASET_ORDER) + r"@{}}")
    a(r"\toprule")
    ds_header = " & ".join(
        rf"\multicolumn{{3}}{{c}}{{{DATASET_LABELS_SHORT[ds]}}}" for ds in DATASET_ORDER
    )
    a(rf"& & {ds_header} \\")
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{3+i*3}-{5+i*3}}}" for i in range(len(DATASET_ORDER))
    )
    a(cmidrules)
    a(r"Type & Model & " + " & ".join([r"Base & $V_4$ & $D_4$"] * len(DATASET_ORDER)) + r" \\")
    a(r"\midrule")

    ncols = 2 + 3 * len(DATASET_ORDER)

    def _model_row(model: str) -> str:
        label = MODEL_LABELS[model]
        cells: list[str] = []
        skip_stars = (model == "d4wrn")  # variable/fewer seeds — stars not meaningful
        for ds in DATASET_ORDER:
            try:
                row = stats.loc[(model, ds)]
                deltas = per_seed_deltas.get((model, ds), np.array([]))
                star = "" if skip_stars else _sig_star(deltas)
                d = _fmt_delta(row["delta"], star=star)
                if best.get(ds) == model:
                    d = rf"\textbf{{{d}}}"
                v4 = _fmt_delta(row["delta_v4"]) if not np.isnan(row["delta_v4"]) else "---"
                cells.append(f"{row['base']:.1f} & {v4} & {d}")
            except KeyError:
                cells.append(r"\multicolumn{3}{c}{---}")
        return rf"& {label:<18} & " + " & ".join(cells) + r" \\"

    a(rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\textit{{General-purpose}}}} \\")
    for mdl in MODEL_ORDER_GENERAL:
        a(_model_row(mdl))

    a(r"\addlinespace[3pt]")
    a(rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\textit{{Histology foundation models}}}} \\")
    for mdl in MODEL_ORDER_HISTO:
        a(_model_row(mdl))

    a(r"\addlinespace[3pt]")
    a(rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\textit{{Equivariant reference}}}} \\")
    a(_model_row("d4wrn"))

    a(r"\bottomrule")
    a(r"\end{tabular}")
    a(r"\vspace{4pt}")
    a(r"")
    a(r"\noindent\small$^\dagger$ D4WRN is evaluated in finetuned mode; number of seeds varies by dataset")
    a(r"(TCGA-UT: 5, NCT-CRC-100K/MHIST: 2, NCT-CRC-NoNorm: 1). All other models use frozen backbone")
    a(r"with SGD linear probe (5 seeds).")
    a(r"\end{table*}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


# ── Table 2: TTA benefit by head type ────────────────────────────────────────

def generate_head_comparison_table(out_path: Path) -> None:
    m = _load_deltas()
    m = m[m.head_type.isin(HEAD_ORDER)]
    m["family"] = m.model.apply(lambda x: "histology" if x in FM_HIST else "general")
    # exclude d4wrn from averages
    m = m[m.model != "d4wrn"]

    stats = (m.groupby(["head_type", "dataset", "family"])
               .agg(base=("base_pct", "mean"), delta=("delta_pp", "mean"))
               .round(2))

    n_ds = len(DATASET_ORDER)
    n_cols = 2 + n_ds * 2  # family + head + (base+delta) per dataset

    lines: list[str] = []
    a = lines.append

    a(r"% Auto-generated by utils/generate_tables.py — do not edit by hand.")
    a(r"\begin{table*}[t]")
    a(r"\centering")
    a(r"\caption{Mean $D_4$ TTA benefit ($\Delta_{\text{TTA}}$ in balanced accuracy, pp) by classifier head type and")
    a(r"model family (frozen backbone, probability-mean aggregation). Values averaged over all models")
    a(r"within each family and 5 seeds (kNN: 1 seed, deterministic).")
    a(r"\textit{Base}: mean no-TTA balanced accuracy (\%). $\Delta_{\text{TTA}}$: pp gain from $D_4$ TTA.}")
    a(r"\label{tab:head_comparison}")
    a(r"\vspace{4pt}")
    a(r"\small")
    a(r"\setlength{\tabcolsep}{4pt}")
    col_spec = "@{}ll" + "cc" * n_ds + "@{}"
    a(rf"\begin{{tabular}}{{{col_spec}}}")
    a(r"\toprule")
    ds_header = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{DATASET_LABELS[ds]}}}" for ds in DATASET_ORDER
    )
    a(rf"& & {ds_header} \\")
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{3+i*2}-{4+i*2}}}" for i in range(n_ds)
    )
    a(cmidrules)
    a(r"Family & Head & " + " & ".join(["Base & $\\Delta_{\\text{TTA}}$"] * n_ds) + r" \\")
    a(r"\midrule")

    for family_key, family_label in [("general", "General-purpose models"), ("histology", "Histology FMs")]:
        a(rf"\multicolumn{{{n_cols}}}{{@{{}}l}}{{\textit{{{family_label}}}}} \\")
        for ht in HEAD_ORDER:
            cells = []
            for ds in DATASET_ORDER:
                try:
                    row = stats.loc[(ht, ds, family_key)]
                    cells.append(f"{row['base']:.1f} & {_fmt_delta(row['delta'])}")
                except KeyError:
                    cells.append(r"\multicolumn{2}{c}{---}")
            a(rf"& {HEAD_LABELS[ht]:<18} & " + " & ".join(cells) + r" \\")
        if family_key == "general":
            a(r"\addlinespace[3pt]")

    a(r"\bottomrule")
    a(r"\end{tabular}")
    a(r"\end{table*}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_full_results_table(OUT_DIR / "table_full_results.tex")
    generate_head_comparison_table(OUT_DIR / "table_head_comparison.tex")
