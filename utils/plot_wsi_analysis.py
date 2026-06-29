"""
WSI-level TTA analysis figure.
Layout: [p096 3-panel stack] [p052 3-panel stack] [stats bars]
Each patient column: H&E outline | ΔP(tumor) | per-patch outcome + zoom inset
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from PIL import Image

PANDA_CSV = "results/patch_dice_panda.csv"
CAM_CSV   = "results/patch_dice_camelyon17.csv"

P096_PATH = "figures/wsi_heatmaps/wsi_tta_heatmap_patient_096_node_0.png"
P052_PATH = "figures/wsi_heatmaps/wsi_tta_heatmap_patient_052_node_1.png"

# Manually tuned crop boxes (r0, r1, c0, c1) within each panel for content
# Format: {patient: {panel_idx: (y0, y1, x0, x1)}}
CROPS = {
    "p096": {
        0: (340, 610, 222, 597),   # H&E
        1: (319, 655, 60,  645),   # ΔP
        2: (340, 625, 268, 588),   # outcome
    },
    "p052": {
        0: (270, 735, 15,  599),   # H&E
        1: (270, 700, 34,  655),   # ΔP
        2: (270, 728, 33,  588),   # outcome
    },
}

# Zoom boxes in cropped-outcome-image coordinates (x0, y0, x1, y1)
# These are fractions [0,1] of the cropped image width/height
ZOOM_BOX = {
    "p096": (0.02, 0.02, 0.52, 0.75),   # left metastasis, top-left focus
    "p052": (0.0,  0.25, 0.55, 0.78),   # left tumor region
}


def _load_panel(img_path: str, crops: dict) -> dict[int, np.ndarray]:
    arr = np.array(Image.open(img_path))
    pw = arr.shape[1] // 4
    out = {}
    for idx, (r0, r1, c0, c1) in crops.items():
        panel = arr[r0:r1, pw * idx + c0: pw * idx + c1]
        out[idx] = panel
    return out


def _micro(df: pd.DataFrame, suffix: str):
    tp = df[f"tp_{suffix}"].sum()
    fp = df[f"fp_{suffix}"].sum()
    fn = df[f"fn_{suffix}"].sum()
    tn = (df.n_patches - df.n_pos).sum() - fp
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    return sens, spec


def _find_green_zoom(outcome_arr: np.ndarray):
    """Return (r0,r1,c0,c1) of a 30%-wide window with most green pixels."""
    r, g, b = outcome_arr[:,:,0], outcome_arr[:,:,1], outcome_arr[:,:,2]
    green = (g.astype(int) > r.astype(int) + 40) & (g > b.astype(int) + 30) & (g > 80)
    H, W = outcome_arr.shape[:2]
    win_h, win_w = H // 2, W // 2
    best, best_box = -1, (0, win_h, 0, win_w)
    for ry in range(0, H - win_h, 10):
        for cx in range(0, W - win_w, 10):
            n = green[ry:ry+win_h, cx:cx+win_w].sum()
            if n > best:
                best, best_box = n, (ry, ry+win_h, cx, cx+win_w)
    return best_box


PANEL_LABELS = {0: "H&E + outline", 1: "ΔP(tumor)", 2: "Per-patch outcome"}
PANEL_CMAPS  = {0: None, 1: None, 2: None}


def plot(out_dir: Path = Path("figures"),
         panda_csv: str = PANDA_CSV, cam_csv: str = CAM_CSV):

    panda = pd.read_csv(panda_csv)
    cam   = pd.read_csv(cam_csv)

    # Stats
    stats = {}
    for name, df in [("PANDA", panda), ("Camelyon17", cam)]:
        sb, spb = _micro(df, "base")
        st, spt = _micro(df, "tta")
        pos = df[df.n_pos > 0]
        stats[name] = dict(
            sens_base=sb, sens_tta=st, sens_delta=(st - sb) * 100,
            spec_base=spb, spec_tta=spt, spec_delta=(spt - spb) * 100,
            corr_pct=pos.corrected.sum() / pos.n_pos.sum() * 100,
            corru_pct=pos.corrupted.sum() / pos.n_pos.sum() * 100,
        )

    panels_96 = _load_panel(P096_PATH, CROPS["p096"])
    panels_52 = _load_panel(P052_PATH, CROPS["p052"])

    # ── Layout ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 8))
    # 3 columns: p096, p052, stats; 3 rows: H&E, ΔP, outcome
    gs_outer = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.85],
                                 wspace=0.12, left=0.03, right=0.97,
                                 top=0.93, bottom=0.06)

    gs_96 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_outer[0],
                                             hspace=0.08)
    gs_52 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_outer[1],
                                             hspace=0.08)
    gs_st = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[2],
                                             hspace=0.45)

    pat_info = [
        ("p096", panels_96, "Patient 096, node 0\ncorrected=178  corrupted=12  net=+166"),
        ("p052", panels_52, "Patient 052, node 1\ncorrected=219  corrupted=11  net=+208"),
    ]
    gs_list = [gs_96, gs_52]

    for (pid, panels, title), gs in zip(pat_info, gs_list):
        zoom_box = ZOOM_BOX[pid]
        for row, panel_idx in enumerate([0, 1, 2]):
            ax = fig.add_subplot(gs[row])
            arr = panels[panel_idx]
            ax.imshow(arr, aspect="auto")
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=7.5, pad=3, loc="left")
            ax.text(0.01, 0.97, PANEL_LABELS[panel_idx],
                    transform=ax.transAxes, fontsize=6.5, va="top",
                    color="white" if panel_index_has_dark_bg(panel_index=panel_idx) else "black",
                    bbox=dict(facecolor="black", alpha=0.35, pad=1.5, boxstyle="round"))

            # Zoom inset on outcome panel
            if panel_idx == 2:
                H, W = arr.shape[:2]
                zr0, zr1, zc0, zc1 = _find_green_zoom(arr)
                # Create inset
                axins = inset_axes(ax, width="45%", height="45%",
                                   loc="upper left",
                                   bbox_to_anchor=ax.bbox,
                                   bbox_transform=ax.transData,
                                   borderpad=0)
                axins.imshow(arr[zr0:zr1, zc0:zc1], aspect="auto")
                axins.axis("off")
                # Rectangle on main image
                rect = mpatches.Rectangle(
                    (zc0, zr0), zc1 - zc0, zr1 - zr0,
                    linewidth=1.2, edgecolor="yellow", facecolor="none")
                ax.add_patch(rect)
                # Connect inset corners to rect corners
                mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="yellow",
                           lw=0.8)

    # ── Stats panels ────────────────────────────────────────────────────────
    C_PANDA = "#e07b39"
    C_CAM   = "#2a5fa5"

    # Top: specificity improvement
    ax_spec = fig.add_subplot(gs_st[0])
    names = ["PANDA", "Camelyon17"]
    colors = [C_PANDA, C_CAM]
    x = [0, 1]
    bar_w = 0.32
    for i, (name, c) in enumerate(zip(names, colors)):
        s = stats[name]
        b_val = ax_spec.bar(i - bar_w/2, s["spec_base"] * 100, width=bar_w,
                            color=c, alpha=0.4, label=f"{name} baseline")
        t_val = ax_spec.bar(i + bar_w/2, s["spec_tta"]  * 100, width=bar_w,
                            color=c, alpha=0.95, label=f"{name} D4-TTA")
        # delta annotation
        top = max(s["spec_base"], s["spec_tta"]) * 100
        ax_spec.annotate(f'{s["spec_delta"]:+.2f} pp',
                         xy=(i + bar_w/2, top), xytext=(0, 3),
                         textcoords="offset points", ha="center",
                         fontsize=7, color=c, fontweight="bold")

    ax_spec.set_xticks(x)
    ax_spec.set_xticklabels(names, fontsize=8.5)
    ax_spec.set_ylabel("Specificity (%)", fontsize=8)
    ax_spec.set_title("Specificity: baseline vs D4-TTA", fontsize=8.5, pad=4)
    ymin = min(stats["PANDA"]["spec_base"], stats["Camelyon17"]["spec_base"]) * 100 - 1.5
    ymax = max(stats["PANDA"]["spec_tta"],  stats["Camelyon17"]["spec_tta"])  * 100 + 1.5
    ax_spec.set_ylim(ymin, ymax)
    ax_spec.tick_params(axis="x", length=0)
    ax_spec.tick_params(axis="y", labelsize=7.5)
    handles = [mpatches.Patch(color=c, alpha=0.4, label="Baseline"),
               mpatches.Patch(color="grey", alpha=0.95, label="D4-TTA")]
    ax_spec.legend(handles=handles, fontsize=7, loc="lower right")
    for sp in ["top", "right"]:
        ax_spec.spines[sp].set_visible(False)

    # Bottom: corrected vs corrupted
    ax_corr = fig.add_subplot(gs_st[1])
    y_pos = [0, 1]
    for i, (name, c) in enumerate(zip(names, colors)):
        s = stats[name]
        ax_corr.barh(i + 0.18, s["corr_pct"],  height=0.3, color="#4caf7d", alpha=0.9,
                     label="Corrected" if i == 0 else "")
        ax_corr.barh(i - 0.18, s["corru_pct"], height=0.3, color="#d94f3d", alpha=0.9,
                     label="Corrupted" if i == 0 else "")
        ax_corr.text(s["corr_pct"]  + 0.15, i + 0.18, f'{s["corr_pct"]:.1f}%',
                     va="center", fontsize=7, color="#4caf7d", fontweight="bold")
        ax_corr.text(s["corru_pct"] + 0.15, i - 0.18, f'{s["corru_pct"]:.1f}%',
                     va="center", fontsize=7, color="#d94f3d", fontweight="bold")

    ax_corr.set_yticks(y_pos)
    ax_corr.set_yticklabels(names, fontsize=8.5)
    ax_corr.set_xlabel("% of tumor patches", fontsize=8)
    ax_corr.set_title("Corrected vs corrupted tumor patches", fontsize=8.5, pad=4)
    ax_corr.set_xlim(0, 13)
    ax_corr.legend(fontsize=7, loc="lower right")
    ax_corr.tick_params(axis="y", length=0)
    ax_corr.tick_params(axis="x", labelsize=7.5)
    for sp in ["top", "right"]:
        ax_corr.spines[sp].set_visible(False)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"figure_wsi_analysis.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_dir}/figure_wsi_analysis.png/.pdf")


def panel_index_has_dark_bg(panel_index: int) -> bool:
    return panel_index in (1, 2)


if __name__ == "__main__":
    plot()
