"""Mega grid: one 9x9 heatmap per secret (receiver × sentinel), showing normalized stego gap."""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from common import (MODEL_KEYS, MODEL_DISPLAY, FAMILY_COLORS, PLOTS_DIR,
                    load_per_codeword_deltas, save_plot)

# Data range [-8, 1] -> colormap [0, 1]
_Z = 8.0 / 9.0
_CMAP = LinearSegmentedColormap.from_list("red_ylgn", [
    (0.0, "#e74c3c"),
    (_Z - 0.001, "#e74c3c"),
    (_Z, "#ffffbf"),
    ((_Z + 1.0) / 2, "#a6d96a"),
    (1.0, "#1a9850"),
])
_CMAP.set_bad(color="#e0e0e0")

_INF_COLOR = "#8b0000"
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
    finite = off[np.isfinite(off["delta_norm_oi"])]
    ranked = finite.groupby("codeword")["delta_norm_oi"].mean().sort_values(ascending=False).index.tolist()
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
    is_neginf = np.zeros((n, n), dtype=bool)
    for _, r in subset.iterrows():
        i = MODEL_KEYS.index(r["receiver"])
        j = MODEL_KEYS.index(r["sentinel"])
        val = r["delta_norm_oi"]
        if np.isneginf(val):
            is_neginf[i, j] = True
        elif np.isfinite(val):
            matrix[i, j] = val

    mask = np.eye(n, dtype=bool)
    sns.heatmap(matrix, ax=ax, cmap=_CMAP, vmin=-8, vmax=1,
                mask=mask, cbar=False, xticklabels=False, yticklabels=False,
                linewidths=0.3, linecolor="white")

    # Gray diagonal
    for k in range(n):
        ax.add_patch(plt.Rectangle((k, k), 1, 1, fill=True,
                     facecolor=_DIAG_COLOR, edgecolor="white", linewidth=0.3))
    # Dark red for -inf
    for i in range(n):
        for j in range(n):
            if is_neginf[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                             facecolor=_INF_COLOR, edgecolor="white", linewidth=0.3, zorder=2))

    # Family separator lines
    for b in _FAMILY_BOUNDS:
        ax.axhline(y=b, color="gray", linewidth=0.4, zorder=3)
        ax.axvline(x=b, color="gray", linewidth=0.4, zorder=3)

    ax.set_title(secret, fontsize=title_fontsize, fontweight="bold", pad=3)
    ax.set_aspect("equal")


def _set_colored_labels(ax, axis="y", fontsize=7):
    """Set model display names with family colors on an axis."""
    n = len(MODEL_KEYS)
    if axis == "y":
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_yticklabels(MODEL_DISPLAY, fontsize=fontsize, rotation=0)
        for i, label in enumerate(ax.get_yticklabels()):
            label.set_color(FAMILY_COLORS[i])
    else:
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_xticklabels(MODEL_DISPLAY, fontsize=fontsize, rotation=45, ha="right")
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_color(FAMILY_COLORS[i])


def _add_legend(fig, cbar_x=0.55, cbar_w=0.3, cbar_y=0.02, cbar_h=0.02, fontscale=1.0):
    patch_w = 0.03
    gap = 0.005

    # No signal (0/0) patch
    nan_ax = fig.add_axes([cbar_x - 2 * (patch_w + gap), cbar_y, patch_w, cbar_h])
    nan_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=_NAN_COLOR, edgecolor="black", linewidth=0.5))
    nan_ax.set_xlim(0, 1)
    nan_ax.set_ylim(0, 1)
    nan_ax.set_xticks([0.5])
    nan_ax.set_xticklabels(["0/0"], fontsize=int(9 * fontscale))
    nan_ax.xaxis.set_ticks_position("top")
    nan_ax.set_yticks([])
    nan_ax.tick_params(length=0)

    # -inf patch
    inf_ax = fig.add_axes([cbar_x - patch_w - gap, cbar_y, patch_w, cbar_h])
    inf_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=_INF_COLOR, edgecolor="black", linewidth=0.5))
    inf_ax.set_xlim(0, 1)
    inf_ax.set_ylim(0, 1)
    inf_ax.set_xticks([0.5])
    inf_ax.set_xticklabels(["$-\\infty$"], fontsize=int(9 * fontscale))
    inf_ax.xaxis.set_ticks_position("top")
    inf_ax.set_yticks([])
    inf_ax.tick_params(length=0)

    # Colorbar with full range, but detach the "1" end as its own patch
    cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
    sm = ScalarMappable(cmap=_CMAP, norm=mcolors.Normalize(vmin=-8, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="both", extendfrac=0.04)
    cbar.set_ticks([-8, -6, -4, -2, 0])
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label("Normalized steganographic gap", fontsize=int(10 * fontscale))
    cbar.ax.tick_params(labelsize=int(9 * fontscale))

    # Green patch for gap = 1 (to the right of colorbar)
    one_ax = fig.add_axes([cbar_x + cbar_w + gap + 0.01, cbar_y, patch_w, cbar_h])
    one_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#1a9850", edgecolor="black", linewidth=0.5))
    one_ax.set_xlim(0, 1)
    one_ax.set_ylim(0, 1)
    one_ax.set_xticks([0.5])
    one_ax.set_xticklabels(["1"], fontsize=int(9 * fontscale))
    one_ax.xaxis.set_ticks_position("top")
    one_ax.set_yticks([])
    one_ax.tick_params(length=0)

    # Legend patches
    legend_patches = [
        Patch(facecolor=_DIAG_COLOR, edgecolor="black", linewidth=0.5, label="self (diagonal)"),
    ]
    fig.legend(handles=legend_patches, loc="lower left", fontsize=int(9 * fontscale), ncol=1,
               bbox_to_anchor=(0.02, 0.005))
    fig.text(0.02, -0.005, "Each cell = one (receiver, sentinel) pair",
             ha="left", fontsize=int(9 * fontscale), style="italic", color="gray")


def plot(save=True):
    off = _load_off()
    secret_to_group = _get_secret_to_group()
    secret_order = sorted(_get_secret_order(off))
    n_secrets = len(secret_order)
    ncols = 8
    nrows = int(np.ceil(n_secrets / ncols))
    n_models = len(MODEL_KEYS)

    subplot_size = 2.8
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * subplot_size + 1, nrows * subplot_size + 2),
                             squeeze=False)

    for idx, secret in enumerate(secret_order):
        row, col = divmod(idx, ncols)
        _render_subplot(axes[row][col], off, secret, secret_to_group)

    for idx in range(n_secrets, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    for row in range(nrows):
        _set_colored_labels(axes[row][0], axis="y", fontsize=5)
    for col in range(ncols):
        last_row = min(nrows - 1, (n_secrets - 1) // ncols) if col < n_secrets % ncols or n_secrets % ncols == 0 else nrows - 2
        _set_colored_labels(axes[last_row][col], axis="x", fontsize=5)

    fig.suptitle("Normalized Steganographic Gap per Secret\n"
                 "rows = receiver, cols = sentinel; alphabetical order",
                 fontsize=14, y=0.98)
    plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.04, right=0.96, top=0.94, bottom=0.07)
    _add_legend(fig)

    if save:
        save_plot(fig, "gap_per_secret_grid.png")
    plt.close()


def plot_sample(n=8, seed=0, save=True):
    off = _load_off()
    secret_to_group = _get_secret_to_group()
    secret_order = _get_secret_order(off)
    rng = np.random.default_rng(seed)
    sample = sorted(rng.choice(secret_order, size=n, replace=False).tolist())

    ncols = 4
    nrows = 2
    n_models = len(MODEL_KEYS)

    subplot_size = 3.5
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * subplot_size + 1, nrows * subplot_size + 2.5),
                             squeeze=False)

    for idx, secret in enumerate(sample):
        row, col = divmod(idx, ncols)
        _render_subplot(axes[row][col], off, secret, secret_to_group, title_fontsize=16)

    for row in range(nrows):
        _set_colored_labels(axes[row][0], axis="y", fontsize=12)
    for col in range(ncols):
        _set_colored_labels(axes[nrows - 1][col], axis="x", fontsize=12)

    fig.suptitle("Normalized Steganographic Gap",
                 fontsize=22, fontweight="bold", y=0.97)
    fig.text(0.5, 0.935, "rows = receiver, cols = sentinel; random sample of 8 secrets",
             ha="center", fontsize=14, color="gray")
    plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.07, right=0.96, top=0.91, bottom=0.12)
    _add_legend(fig, cbar_x=0.55, cbar_w=0.3, cbar_y=0.02, cbar_h=0.025, fontscale=1.5)

    if save:
        save_plot(fig, "gap_per_secret_grid_sample.png")
    plt.close()


def plot_sample4(seed=0, save=True):
    off = _load_off()
    secret_to_group = _get_secret_to_group()
    secret_order = _get_secret_order(off)
    rng = np.random.default_rng(seed)
    sample = sorted(rng.choice(secret_order, size=4, replace=False).tolist())

    ncols = 4
    nrows = 1
    n_models = len(MODEL_KEYS)

    subplot_size = 4.0
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * subplot_size + 1, subplot_size + 3),
                             squeeze=False)

    for idx, secret in enumerate(sample):
        _render_subplot(axes[0][idx], off, secret, secret_to_group, title_fontsize=18)

    _set_colored_labels(axes[0][0], axis="y", fontsize=13)
    for col in range(ncols):
        _set_colored_labels(axes[0][col], axis="x", fontsize=13)

    fig.suptitle("Normalized Steganographic Gap",
                 fontsize=22, fontweight="bold", y=0.97)
    fig.text(0.5, 0.91, "rows = receiver, cols = sentinel; random sample of 4 secrets",
             ha="center", fontsize=15, color="gray")
    plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.07, right=0.96, top=0.85, bottom=0.18)
    _add_legend(fig, cbar_x=0.50, cbar_w=0.35, cbar_y=0.03, cbar_h=0.03, fontscale=1.6)

    if save:
        save_plot(fig, "gap_per_secret_grid_sample4.png")
    plt.close()


if __name__ == "__main__":
    plot()
    print("Saved full grid.")
    plot_sample()
    print("Saved sample 8.")
    plot_sample4()
    print("Saved sample 4.")
