# TTA Research — Experiment TODO List

---

## General To-Do's
> Things to expand on
- [ ] Think about more experiments or tasks to add
- [ ] Consider range of classification labels (binary, smaller classes, harder data)
- [ ] Consider other tasks than classification 
    - [ ] Segmentation
    - [ ] Tracking
- [ ] Make sure all models run:
    - [x] ResNet-18
    - [x] ResNet-50
    - [ ] EfficientNet-B7
    - [x] DINOV2-s
    - [x] DINOV2-b
    - [x] DINOV2-l
    - [x] DINOV2-g
    - [x] ConvNeXt
- [ ] Make sure all datasets accessible
    - [ ] NCT-CRC-100k
    - [ ] NCT-CRC-7k
    - [ ] NCT-CRC-NONORM
    - [x] TCGA-UT

---

## Experiment 0 — Baseline Instrumentation
> Do this first — it unblocks all downstream experiments.

- [x] Add per-class accuracy / F1 logging to `metrics.py`
- [ ] Log confusion matrix at evaluation time
- [x] Save per-class TTA delta (TTA acc minus baseline acc per class)

---

## Experiment 1 — TTA on Finetuned Models
> Hypothesis: TTA delta shrinks after finetuning as augmentation invariance is baked into weights.

- [x] Add finetuning mode to `trainer.py` (unfreeze backbone, lower LR)
- [x] Train finetuned versions of each model
- [x] Compare TTA delta: linear probe vs finetuned
- [x] Check per-class TTA delta — does finetuning close the gap uniformly?

---


---

## Experiment 3 — Per-Class TTA Analysis
> Requires Experiment 0.

- [ ] Identify which classes benefit most/least from TTA
- [ ] Cross-reference with cancer types that are rotationally ambiguous
- [ ] Check if low-sample classes benefit more or less

---

## Experiment 4 — Aggregation Strategy Comparison
> Hypothesis: Mean probability aggregation outperforms majority vote and max, especially for ambiguous classes.

- [ ] Benchmark all three aggregators (mean, max, vote) under identical TTA strategies
- [ ] Compare across model families (CNN vs ViT)
- [ ] Check if aggregation choice interacts with number of TTA views
- [ ] Check per-class results — does vote hurt on ambiguous classes vs mean?

---

## Experiment 5 — TTA on Published Histopathology Models
> Core publishable contribution: show that SOTA histopathology models left performance on the table by not using TTA.

- [ ] Identify target models — candidates: CONCH, UNI, PLIP
  - [ ] Confirm none used TTA in their original evaluation protocol
  - [ ] Confirm published benchmark numbers are reproducible on your datasets
- [ ] Integrate each model as a frozen backbone in the existing pipeline
- [ ] Run `evaluate_tta.py` with full TTA strategy sweep
- [ ] Compare TTA delta against your general-purpose models (CNN / DINOv2)
- [ ] Document delta relative to their published numbers — this is the headline result