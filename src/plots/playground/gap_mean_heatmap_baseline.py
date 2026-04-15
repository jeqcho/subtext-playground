"""Baseline heatmaps using uniform[0,1] random uplift data.

Generates both raw and normed maxmean heatmaps to show what the plots look
like with no real signal — just random noise.
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

from common import (MODEL_KEYS, MODEL_DISPLAY, MODEL_SHORT, FAMILY_COLORS,
                    load_per_codeword_deltas, save_plot)
from gap_per_secret_grid import (
    _render_subplot, _set_colored_labels, _add_legend, _FAMILY_BOUNDS,
)

# Raw colormap (same as gap_mean_heatmap_raw_maxmean)
_RAW_CMAP = LinearSegmentedColormap.from_list("red_ylbu_raw", [
    (0.0,  "#b2182b"),
    (0.30, "#d73027"),
    (0.45, "#fc8d59"),
    (0.50, "#ffffbf"),
    (0.55, "#abd9e9"),
    (0.70, "#74add1"),
    (1.0,  "#4575b4"),
])
_RAW_CMAP.set_bad(color="#e0e0e0")
_DIAG_COLOR = "#a0a0a0"
_FAMILY_BOUNDS_IDX = [3, 6]

N_CODEWORDS = 40
SEED = 42


def _generate_baseline_uplift(seed=SEED):
    """Generate uniform[0,1] uplift for each (sender, evaluator, codeword)."""
    rng = np.random.default_rng(seed)
    rows = []
    for sender in MODEL_KEYS:
        for evaluator in MODEL_KEYS:
            for cw_idx in range(N_CODEWORDS):
                rows.append({
                    "sender": sender,
                    "evaluator": evaluator,
                    "codeword": f"cw_{cw_idx}",
                    "uplift": rng.choice(np.arange(0, 1.05, 0.05)),
                })
    return pd.DataFrame(rows)


def _compute_maxmean(df):
    """Compute max(0, mean(uplift)) per (sender, evaluator)."""
    mean_uplift = df.groupby(["sender", "evaluator"])["uplift"].mean()
    return mean_uplift.clip(lower=0)


def plot_raw(save=True):
    df = _generate_baseline_uplift()
    mean_uplift_oi = _compute_maxmean(df)
    self_uplift = {m: mean_uplift_oi[(m, m)] for m in MODEL_KEYS}

    n = len(MODEL_KEYS)
    matrix = np.full((n, n), np.nan)
    for recv in MODEL_KEYS:
        for sent in MODEL_KEYS:
            if recv == sent:
                continue
            i = MODEL_KEYS.index(recv)
            j = MODEL_KEYS.index(sent)
            matrix[i, j] = self_uplift[recv] - mean_uplift_oi[(recv, sent)]

    mask = np.eye(n, dtype=bool)
    off_vals = matrix[~mask]
    vmax = max(abs(np.nanmin(off_vals)), abs(np.nanmax(off_vals)))

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, ax=ax, cmap=_RAW_CMAP, vmin=-vmax, vmax=vmax,
                mask=mask, cbar=False, xticklabels=False, yticklabels=False,
                linewidths=0.5, linecolor="white", annot=False)

    for k in range(n):
        ax.add_patch(plt.Rectangle((k, k), 1, 1, fill=True,
                     facecolor=_DIAG_COLOR, edgecolor="white", linewidth=0.5))
    for b in _FAMILY_BOUNDS_IDX:
        ax.axhline(y=b, color="gray", linewidth=1.0, zorder=3)
        ax.axvline(x=b, color="gray", linewidth=1.0, zorder=3)

    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(MODEL_DISPLAY, fontsize=11, rotation=0)
    for i, label in enumerate(ax.get_yticklabels()):
        label.set_color(FAMILY_COLORS[i])
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(MODEL_SHORT, fontsize=11, rotation=45, ha="right")
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(FAMILY_COLORS[i])

    ax.set_ylabel("Receiver", fontsize=13)
    ax.set_xlabel("Sentinel", fontsize=13)
    ax.set_title("Baseline: Mean Steganographic Gap (Raw)\nuniform[0,1] random uplift",
                 fontsize=15, fontweight="bold", pad=10)
    ax.set_aspect("equal")

    sm = ScalarMappable(cmap=_RAW_CMAP, norm=mcolors.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8, pad=0.02)
    cbar.set_label("Mean steganographic gap (raw)", fontsize=11)

    plt.tight_layout()
    if save:
        save_plot(fig, "gap_mean_heatmap_raw_maxmean_baseline.png")
    plt.close()


def plot_normed(save=True):
    """Normed baseline using _render_subplot from gap_per_secret_grid."""
    df = _generate_baseline_uplift()
    mean_uplift_oi = _compute_maxmean(df)
    self_uplift = {m: mean_uplift_oi[(m, m)] for m in MODEL_KEYS}

    # Build fake off dataframe that _render_subplot expects
    rows = []
    for recv in MODEL_KEYS:
        for sent in MODEL_KEYS:
            if recv == sent:
                continue
            gap = self_uplift[recv] - mean_uplift_oi[(recv, sent)]
            if self_uplift[recv] == 0:
                dn_oi = -np.inf if gap < 0 else np.nan
            else:
                dn_oi = gap / self_uplift[recv]
            rows.append({
                "receiver": recv,
                "sentinel": sent,
                "codeword": "__baseline__",
                "delta_norm_oi": dn_oi,
            })
    off = pd.DataFrame(rows)

    subplot_size = 6.0
    fig, ax = plt.subplots(1, 1,
                           figsize=(subplot_size + 3, subplot_size + 5),
                           squeeze=True)

    _render_subplot(ax, off, "__baseline__", {}, title_fontsize=0)
    ax.set_title("")

    _set_colored_labels(ax, axis="y", fontsize=14)
    _set_colored_labels(ax, axis="x", fontsize=14, short=True)

    ax.set_ylabel("Receiver", fontsize=15, labelpad=10)
    ax.set_xlabel("Sentinel", fontsize=15, labelpad=10)

    fig.suptitle("Baseline: Mean Normalized Steganographic Gap\n"
                 "uniform[0,1] random uplift",
                 fontsize=18, fontweight="bold", y=0.95)

    plt.subplots_adjust(left=0.20, right=0.92, top=0.88, bottom=0.32)
    _add_legend(fig, cbar_x=0.27, cbar_w=0.42, cbar_y=0.12, cbar_h=0.03,
                fontscale=1.5, show_table=True)

    if save:
        save_plot(fig, "gap_mean_heatmap_normed_maxmean_baseline.png")
    plt.close()


if __name__ == "__main__":
    plot_raw()
    print("Saved baseline raw maxmean heatmap.")
    plot_normed()
    print("Saved baseline normed maxmean heatmap.")
