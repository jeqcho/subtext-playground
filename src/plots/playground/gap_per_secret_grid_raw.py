"""Mega grid: one 9x9 heatmap per secret (receiver × sentinel), showing raw (non-normalized) stego gap."""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from common import (MODEL_KEYS, MODEL_DISPLAY, MODEL_SHORT, FAMILY_COLORS, PLOTS_DIR,
                    load_per_codeword_deltas, save_plot)

# Colormap with sharp red/blue saturation and thin yellow band at 0
# Matches the visual punch of the original normalized colormap
_CMAP = LinearSegmentedColormap.from_list("red_ylbu_raw", [
    (0.0,  "#b2182b"),   # deep red at -1
    (0.30, "#d73027"),   # red at -0.4
    (0.45, "#fc8d59"),   # orange at -0.1
    (0.50, "#ffffbf"),   # yellow at 0
    (0.55, "#abd9e9"),   # light blue at +0.1
    (0.70, "#74add1"),   # blue at +0.4
    (1.0,  "#4575b4"),   # deep blue at +1
])
_CMAP.set_bad(color="#e0e0e0")

_DIAG_COLOR = "#a0a0a0"
_NAN_COLOR = "#e0e0e0"
_FAMILY_BOUNDS = [3, 6]  # after Anthropic (0-2), after OpenAI (3-5)


def _load_off():
    df = load_per_codeword_deltas()
    off = df[df["sender"] != df["evaluator"]].copy()
    off = off.rename(columns={"sender": "receiver", "evaluator": "sentinel"})
    return off


def _get_secret_to_group():
    df = load_per_codeword_deltas()
    return df[["codeword", "group_idx"]].drop_duplicates().set_index("codeword")["group_idx"].to_dict()


def _get_secret_order(off, randomize=False, seed=0):
    all_secrets = off["codeword"].unique().tolist()
    finite = off[np.isfinite(off["delta_oi"])]
    ranked = finite.groupby("codeword")["delta_oi"].mean().sort_values(ascending=False).index.tolist()
    dead = [s for s in all_secrets if s not in ranked]
    order = ranked + sorted(dead)
    if randomize:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    return order


def _render_subplot(ax, off, secret, secret_to_group, title_fontsize=10):
    n = len(MODEL_KEYS)
    subset = off[off["codeword"] == secret]
    matrix = np.full((n, n), np.nan)
    is_nosignal = np.zeros((n, n), dtype=bool)
    for _, r in subset.iterrows():
        i = MODEL_KEYS.index(r["receiver"])
        j = MODEL_KEYS.index(r["sentinel"])
        # Use delta_norm_oi to detect no-signal (0/0 → NaN in normalized)
        if np.isnan(r["delta_norm_oi"]):
            is_nosignal[i, j] = True
        else:
            val = r["delta_oi"]
            if np.isfinite(val):
                matrix[i, j] = val

    mask = np.eye(n, dtype=bool)
    sns.heatmap(matrix, ax=ax, cmap=_CMAP, vmin=-1, vmax=1,
                mask=mask, cbar=False, xticklabels=False, yticklabels=False,
                linewidths=0.3, linecolor="white")

    # Gray diagonal
    for k in range(n):
        ax.add_patch(plt.Rectangle((k, k), 1, 1, fill=True,
                     facecolor=_DIAG_COLOR, edgecolor="white", linewidth=0.3))
    # Gray for no-signal cells
    for i in range(n):
        for j in range(n):
            if is_nosignal[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                             facecolor=_NAN_COLOR, edgecolor="white", linewidth=0.3, zorder=2))

    # Family separator lines
    for b in _FAMILY_BOUNDS:
        ax.axhline(y=b, color="gray", linewidth=0.4, zorder=3)
        ax.axvline(x=b, color="gray", linewidth=0.4, zorder=3)

    ax.set_title(secret, fontsize=title_fontsize, fontweight="bold", pad=3)
    ax.set_aspect("equal")


def _set_colored_labels(ax, axis="y", fontsize=7, short=False):
    """Set model display names with family colors on an axis."""
    n = len(MODEL_KEYS)
    names = MODEL_SHORT if short else MODEL_DISPLAY
    if axis == "y":
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_yticklabels(names, fontsize=fontsize, rotation=0)
        for i, label in enumerate(ax.get_yticklabels()):
            label.set_color(FAMILY_COLORS[i])
    else:
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_xticklabels(names, fontsize=fontsize, rotation=45, ha="right")
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_color(FAMILY_COLORS[i])


def _add_legend(fig, cbar_x=0.55, cbar_w=0.3, cbar_y=0.02, cbar_h=0.02, fontscale=1.0):
    patch_w = 0.05
    gap = 0.005

    # No signal (0/0) patch
    nan_ax = fig.add_axes([cbar_x - (patch_w + gap), cbar_y, patch_w, cbar_h])
    nan_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=_NAN_COLOR, edgecolor="black", linewidth=0.5))
    nan_ax.set_xlim(0, 1)
    nan_ax.set_ylim(0, 1)
    nan_ax.set_xticks([0.5])
    nan_ax.set_xticklabels(["No signal"], fontsize=int(9 * fontscale))
    nan_ax.xaxis.set_ticks_position("top")
    nan_ax.set_yticks([])
    nan_ax.tick_params(length=0)

    # Colorbar
    cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
    sm = ScalarMappable(cmap=_CMAP, norm=mcolors.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="both", extendfrac=0.04)
    cbar.set_ticks([-1.0, -0.5, 0, 0.5, 1.0])
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label("Steganographic gap (raw)", fontsize=int(10 * fontscale))
    cbar.ax.tick_params(labelsize=int(9 * fontscale))

    # Self (diagonal) patch
    diag_x = cbar_x + cbar_w + gap + 0.01
    diag_ax = fig.add_axes([diag_x, cbar_y, patch_w, cbar_h])
    diag_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=_DIAG_COLOR, edgecolor="black", linewidth=0.5))
    diag_ax.set_xlim(0, 1)
    diag_ax.set_ylim(0, 1)
    diag_ax.set_xticks([0.5])
    diag_ax.set_xticklabels(["Receiver is\nSentinel"], fontsize=int(7 * fontscale))
    diag_ax.xaxis.set_ticks_position("top")
    diag_ax.set_yticks([])
    diag_ax.tick_params(length=0)


def plot(save=True):
    off = _load_off()
    secret_to_group = _get_secret_to_group()
    secret_order = sorted(_get_secret_order(off))
    n_secrets = len(secret_order)
    ncols = 5
    nrows = int(np.ceil(n_secrets / ncols))

    subplot_size = 2.8
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * subplot_size + 1, nrows * subplot_size + 2),
                             squeeze=False)

    for idx, secret in enumerate(secret_order):
        row, col = divmod(idx, ncols)
        _render_subplot(axes[row][col], off, secret, secret_to_group, title_fontsize=19)

    for idx in range(n_secrets, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    for row in range(nrows):
        _set_colored_labels(axes[row][0], axis="y", fontsize=14, short=True)
    for col in range(ncols):
        last_row = min(nrows - 1, (n_secrets - 1) // ncols) if col < n_secrets % ncols or n_secrets % ncols == 0 else nrows - 2
        _set_colored_labels(axes[last_row][col], axis="x", fontsize=16, short=True)

    fig.suptitle("Steganographic Gap (Raw)",
                 fontsize=28, fontweight="bold", y=1.01)
    fig.text(0.5, 0.98, "rows = receiver, cols = sentinel; alphabetical order",
             ha="center", fontsize=17, color="gray")
    plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.04, right=0.96, top=0.96, bottom=0.10)
    _add_legend(fig, cbar_x=0.18, cbar_w=0.58, cbar_y=0.01, cbar_h=0.015, fontscale=2.0)

    if save:
        save_plot(fig, "gap_per_secret_grid_raw.png")
    plt.close()


def plot_sample4(seed=0, save=True):
    off = _load_off()
    secret_to_group = _get_secret_to_group()
    secret_order = _get_secret_order(off)
    rng = np.random.default_rng(seed)
    sample = sorted(rng.choice(secret_order, size=4, replace=False).tolist())

    ncols = 4
    nrows = 1

    subplot_size = 4.0
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * subplot_size + 1, subplot_size + 5),
                             squeeze=False)

    for idx, secret in enumerate(sample):
        _render_subplot(axes[0][idx], off, secret, secret_to_group, title_fontsize=30)

    _set_colored_labels(axes[0][0], axis="y", fontsize=22)
    for col in range(ncols):
        _set_colored_labels(axes[0][col], axis="x", fontsize=22, short=True)

    fig.suptitle("Models can influence codeword selection (raw gap)",
                 fontsize=36, fontweight="bold", y=1.05)
    fig.text(0.5, 0.96, "rows = receiver, cols = sentinel; random sample of 4 secrets",
             ha="center", fontsize=20, color="gray")
    plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.07, right=0.96, top=0.95, bottom=0.36)
    _add_legend(fig, cbar_x=0.18, cbar_w=0.58, cbar_y=0.14, cbar_h=0.04, fontscale=2.6)

    if save:
        save_plot(fig, "gap_per_secret_grid_raw_sample4.png")
    plt.close()


if __name__ == "__main__":
    plot()
    print("Saved full grid (raw).")
    plot_sample4()
    print("Saved sample 4 (raw).")
