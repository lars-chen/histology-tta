import torch, os
from pathlib import Path

def load_meta(path):
    try:
        ck = torch.load(path, map_location=lambda s, l: torch.UntypedStorage(), weights_only=False)
        return {k: v for k, v in ck.items() if isinstance(v, (int, float, str, type(None)))}
    except Exception as e:
        return {'error': str(e)}

ckpt_dir = Path('checkpoints')
results = []
for f in sorted(ckpt_dir.glob('*.pt')):
    meta = load_meta(f)
    size_mb = f.stat().st_size / 1e6
    results.append((f.stem, meta.get('epoch','?'), meta.get('val_acc','?'), size_mb, meta.get('error','')))

print(f"{'checkpoint':<65} {'epoch':>5} {'val_acc':>8} {'MB':>7}")
print('-'*92)
for stem, epoch, val_acc, mb, err in results:
    va = f'{val_acc:.4f}' if isinstance(val_acc, float) else str(val_acc)
    flag = ' ERROR' if err else (' <<' if isinstance(epoch, int) and epoch <= 2 else '')
    print(f"{stem:<65} {str(epoch):>5} {va:>8} {mb:>7.1f}{flag}")
