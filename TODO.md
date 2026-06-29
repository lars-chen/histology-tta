# TTA Research — TODO

---

## Paper status: near-complete draft

Core results are in. Remaining items are polish / submission prep.

---

## Submission blockers

- [ ] **`paper/latex/figures/graphabs.pdf`** — assemble Figure 1 (graphical abstract) from components:
  - `figures/ga_d4variance_patient_096_node_0.pdf` — Camelyon17 WSI heatmap panel
  - `figures/panels/p096_node0_*` — outcome / delta_p zoom panels
  - `utils/plot_graphabs_inset.py` — bottom-right inset (TTA improvement dots + selective curves)
  - Left panel: method schematic (needs Illustrator/Figma)
- [ ] Fill MELBA metadata in `paper.tex`: `\melbaid`, `\doi`, `\melbaauthors`, `\melbaspecialissue`, `\datesubmitted`, `\datepublished`, `\melbayear`, `\firstpageno`, `\ShortHeadings`, `\volume`
- [ ] Add bibliography file (`sample.bib`) to `paper/` 
- [ ] Add `melba.cls` style file to `paper/`
- [ ] Verify paper compiles end-to-end on Overleaf

---

## Results to double-check

- [ ] Confirm numbers in paper abstract match `figures/canonical_results.csv`:
  - Histo FMs: +0.30 pp overall, +0.50 pp TCGA-UT
  - General: +1.28 pp overall, +1.98 pp TCGA-UT
  - D4WRN: +0.17 pp
- [ ] Check selective TTA claims (t=0.3 → 13% samples, ~70% benefit recovered) against `figures/selective_gating_summary.csv`
- [ ] Check agreement rate claims (97.0% FM, 91.2% general) against `figures/agreement_rate_summary.csv`
- [ ] Check MPCS claims (FM 0.946, general 0.950, D4WRN ≈0.999) against `figures/orbit_tightness.csv`
- [ ] Verify Camelyon17 AUC deltas (≤0.009) and PANDA kappa (0.901 vs 0.905) against `figures/camelyon17_tta.csv` / `figures/panda_tta.csv`

---

## Figure polish

- [ ] Figure 3 panel C: decide between `figure3_panel_c_v1.pdf` and `figure3_panel_c_v2.pdf`
- [ ] Graphical abstract inset: run `utils/plot_graphabs_inset.py` to regenerate if data changed

---

## Done (paper-complete)

- [x] Benchmark 16 models × 4 patch datasets (linear probe + frozen backbone)
- [x] V4 vs D4 strategy comparison, all 4 aggregation methods
- [x] Frozen vs finetuned comparison (6 general-purpose models, 3 datasets)
- [x] Entropy / correction / corruption analysis (Figure 3)
- [x] Selective TTA Pareto curves (Figure 4)
- [x] Orbit tightness (MPCS) analysis
- [x] Flip discrimination analysis
- [x] Camelyon17 LOPO patch-Dice + ABMIL MIL evaluation
- [x] PANDA patch probe + ABMIL evaluation
- [x] Head comparison (linear / MLP-1h / MLP-2h / kNN)
- [x] D4WRN equivariant reference model
- [x] Table generation (`utils/generate_tables.py`)
- [x] All figures generated and saved to `figures/`
- [x] `paper/latex/` structure set up (figures 2–4, both tables)
