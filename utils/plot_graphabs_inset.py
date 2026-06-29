"""
Small inset plot for graphical abstract:
Δ balanced accuracy (D4 TTA vs baseline) per model, grouped by family.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

MODEL_TYPE = {
    'phikon':'histology','phikon2':'histology','uni':'histology','uni2':'histology',
    'virchow':'histology','virchow2':'histology','gigapath':'histology','hoptimus':'histology',
    'ctranspath':'histology',
    'dinov2_s':'general','dinov2_b':'general',
    'convnextv2_tiny':'general','convnextv2_base':'general','d4wrn':'general',
}
MODEL_FULL = {
    'phikon':'Phikon','phikon2':'Phikon-v2','uni':'UNI','uni2':'UNI2-h',
    'virchow':'Virchow','virchow2':'Virchow2','gigapath':'GigaPath','hoptimus':'H-optimus',
    'ctranspath':'CTransPath',
    'dinov2_s':'DINOv2-S','dinov2_b':'DINOv2-B',
    'convnextv2_tiny':'ConvNeXtV2-T','convnextv2_base':'ConvNeXtV2-B','d4wrn':'D4-WRN',
}

# Match graphical abstract palette
C_HISTO   = "#2a5fa5"   # blue
C_GENERAL = "#e07b39"   # orange


def build_data(results_csv="results/tta_results.csv"):
    df = pd.read_csv(results_csv)
    df = df[
        df.strategy.isin(['none', 'd4']) &
        (df.backbone_mode == 'frozen') &
        (df.train_augment == True) &
        (df.dataset != 'mhist') &
        (df.aggregation == 'mean')
    ]
    base = df[df.strategy == 'none'].groupby(['model','dataset','seed'])['balanced_acc'].mean()
    tta  = df[df.strategy == 'd4'  ].groupby(['model','dataset','seed'])['balanced_acc'].mean()
    merged = base.rename('base').reset_index().merge(
        tta.rename('tta').reset_index(), on=['model','dataset','seed'])
    merged['delta'] = (merged.tta - merged.base) * 100
    per_model = merged.groupby('model')['delta'].mean().reset_index()
    per_model['type']  = per_model['model'].map(MODEL_TYPE)
    per_model['label'] = per_model['model'].map(MODEL_FULL)
    return per_model.dropna(subset=['type'])


def plot(out_dir: Path = Path("figures"), results_csv: str = "results/tta_results.csv"):
    df = build_data(results_csv)

    histo   = df[df.type == 'histology'].sort_values('delta')
    general = df[df.type == 'general'  ].sort_values('delta')

    histo_mean   = histo.delta.mean()
    general_mean = general.delta.mean()

    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    jitter = 0.12

    # Histology dots
    xs_h = rng.uniform(-jitter, jitter, len(histo))
    ax.scatter(xs_h, histo.delta, color=C_HISTO, s=28, zorder=3,
               alpha=0.85, linewidths=0, clip_on=False)

    # General dots
    xs_g = 1 + rng.uniform(-jitter, jitter, len(general))
    ax.scatter(xs_g, general.delta, color=C_GENERAL, s=28, zorder=3,
               alpha=0.85, linewidths=0, clip_on=False)

    # Mean lines
    lw = 2.2
    ax.plot([-0.28, 0.28], [histo_mean,   histo_mean  ], color=C_HISTO,   lw=lw, zorder=4)
    ax.plot([ 0.72, 1.28], [general_mean, general_mean], color=C_GENERAL, lw=lw, zorder=4)

    # Mean labels
    ax.text(0.32, histo_mean,   f'+{histo_mean:.2f} pp',
            va='center', ha='left', fontsize=7.5, color=C_HISTO,   fontweight='bold')
    ax.text(1.32, general_mean, f'+{general_mean:.2f} pp',
            va='center', ha='left', fontsize=7.5, color=C_GENERAL, fontweight='bold')

    ax.axhline(0, color='#999999', lw=0.8, ls='--', zorder=1)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Histology\nFMs', 'General-\npurpose'], fontsize=8.5)
    ax.set_xlim(-0.55, 1.85)
    ax.set_ylabel('Δ Balanced Accuracy (pp)', fontsize=8)


    ax.set_ylim(-0.2, 1.6)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(axis='y', labelsize=7.5)
    ax.tick_params(axis='x', length=0)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    fig.tight_layout(pad=0.4)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        p = out_dir / f"graphabs_inset.{ext}"
        fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved → {out_dir}/graphabs_inset.png/.pdf")


if __name__ == "__main__":
    plot()
