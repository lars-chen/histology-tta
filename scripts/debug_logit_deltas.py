import torch, numpy as np, glob
from sklearn.metrics import balanced_accuracy_score
import torch.nn.functional as F

MODEL_TYPE = {
    'dinov2_s':'general','dinov2_b':'general',
    'convnextv2_tiny':'general','convnextv2_base':'general',
}

results = {}
for pt_file in sorted(glob.glob("logits/tcga-ut/*.pt")):
    ck = torch.load(pt_file, map_location="cpu", weights_only=False)
    if ck.get("backbone_mode") != "frozen": continue
    if ck.get("aug_tag", "aug") != "aug": continue
    model = ck.get("model", "")
    if model not in MODEL_TYPE: continue
    logits = ck["logits"].float()
    labels = ck["labels"].numpy()
    base_preds = F.softmax(logits[0], dim=-1).argmax(-1).numpy()
    d4_preds   = F.softmax(logits, dim=-1).mean(0).argmax(-1).numpy()
    base_bacc  = balanced_accuracy_score(labels, base_preds) * 100
    d4_bacc    = balanced_accuracy_score(labels, d4_preds)   * 100
    delta = d4_bacc - base_bacc
    results.setdefault(model, []).append((ck.get("seed"), round(delta, 3)))
    print(f"{model} seed={ck.get('seed')}: base={base_bacc:.3f} d4={d4_bacc:.3f} delta={delta:.3f}")

print("\n--- per-model means ---")
for m, runs in sorted(results.items()):
    print(f"{m}: {np.mean([r[1] for r in runs]):.3f}")
print(f"\ngeneral mean: {np.mean([np.mean([r[1] for r in v]) for v in results.values()]):.3f}")
