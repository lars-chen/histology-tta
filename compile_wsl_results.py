"""Merge camelyon17_tta.csv and panda_tta.csv into results/wsl_results.csv."""
import pandas as pd
from pathlib import Path

RES = Path("results")

cam = pd.read_csv(RES / "camelyon17_tta.csv")
cam["dataset"] = "camelyon17"

pan = pd.read_csv(RES / "panda_tta.csv")
pan["dataset"] = "panda"
pan = pan.rename(columns={"fold": "seed"})

combined = pd.concat([cam, pan], ignore_index=True)

# normalise column order
cols = ["dataset", "model", "mil_type", "train_tta", "tta_eval", "seed",
        "auc", "kappa", "acc", "f1", "n"]
for c in cols:
    if c not in combined.columns:
        combined[c] = None
combined = combined[cols]

out = RES / "wsl_results.csv"
combined.to_csv(out, index=False)
print(f"Wrote {out}  ({len(combined)} rows)")
