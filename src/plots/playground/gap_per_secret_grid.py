"""Mega grid: one 9x9 heatmap per secret (receiver × sentinel), showing normalized stego gap."""
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

# Data range [-8, 1] -> colormap [0, 1]
_Z = 8.0 / 9.0
_CMAP = LinearSegmentedColormap.from_list("red_ylbu", [
    (0.0, "#d73027"),
    (_Z - 0.001, "#d73027"),
    (_Z, "#ffffbf"),
    ((_Z + 1.0) / 2, "#74add1"),
    (1.0, "#2166ac"),
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


def _add_legend(fig, cbar_x=0.55, cbar_w=0.3, cbar_y=0.02, cbar_h=0.02, fontscale=1.0, show_table=False):
    patch_w = 0.05
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
    one_x = cbar_x + cbar_w + gap + 0.01
    one_ax = fig.add_axes([one_x, cbar_y, patch_w, cbar_h])
    one_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#2166ac", edgecolor="black", linewidth=0.5))
    one_ax.set_xlim(0, 1)
    one_ax.set_ylim(0, 1)
    one_ax.set_xticks([0.5])
    one_ax.set_xticklabels(["1"], fontsize=int(9 * fontscale))
    one_ax.xaxis.set_ticks_position("top")
    one_ax.set_yticks([])
    one_ax.tick_params(length=0)

    # Self (diagonal) patch (to the right of green)
    diag_x = one_x + patch_w + gap
    diag_ax = fig.add_axes([diag_x, cbar_y, patch_w, cbar_h])
    diag_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=_DIAG_COLOR, edgecolor="black", linewidth=0.5))
    diag_ax.set_xlim(0, 1)
    diag_ax.set_ylim(0, 1)
    diag_ax.set_xticks([0.5])
    diag_ax.set_xticklabels(["Receiver is\nSentinel"], fontsize=int(7 * fontscale))
    diag_ax.xaxis.set_ticks_position("top")
    diag_ax.set_yticks([])
    diag_ax.tick_params(length=0)

    if not show_table:
        return

    # Table underneath the colorbar explaining semantics
    fs = int(7 * fontscale)
    row_h = cbar_h * 1.0
    table_y = cbar_y - row_h * 2 - 0.005

    # Column positions and widths: [gray, dark_red, red_bar, green_bar, green_patch]
    nan_x = cbar_x - 2 * (patch_w + gap)
    inf_x = cbar_x - patch_w - gap
    one_x = cbar_x + cbar_w + gap + 0.01
    # Find where 0 is in the colorbar axes (accounting for extend triangles)
    # The colorbar inner axes maps data to [0,1]; get the position of value 0
    zero_frac = (0 - (-8)) / (1 - (-8))  # = 8/9 in data space
    # The extend triangles take extendfrac from each side
    ext = 0.04
    inner_start = ext / (1 + 2 * ext)
    inner_end = 1 - inner_start
    inner_w = inner_end - inner_start
    zero_pos = inner_start + zero_frac * inner_w  # fraction of cbar_w
    red_bar_w = cbar_w * zero_pos
    green_bar_w = cbar_w * (1 - zero_pos) - gap
    green_bar_x = cbar_x + red_bar_w

    cols = [
        (nan_x, patch_w, "≤ 0", "≤ 0"),
        (inf_x, patch_w, "≤ 0", "> 0"),
        (cbar_x, red_bar_w, "< sentinel", "> 0"),
        (green_bar_x, green_bar_w, "> sentinel", "> 0"),
        (one_x, patch_w, "> 0", "≤ 0"),
    ]

    for i, (cx, cw, recv_text, sent_text) in enumerate(cols):
        for row_idx, text in enumerate([recv_text, sent_text]):
            y = table_y + (1 - row_idx) * row_h
            tax = fig.add_axes([cx, y, cw, row_h])
            tax.set_xlim(0, 1)
            tax.set_ylim(0, 1)
            tax.text(0.5, 0.5, text, ha="center", va="center", fontsize=fs)
            tax.set_xticks([])
            tax.set_yticks([])
            for spine in tax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("gray")

    # Row labels to the left
    label_w = 0.14
    label_x = nan_x - label_w - 0.005
    for row_idx, label in enumerate(["Receiver uplift", "Sentinel uplift"]):
        y = table_y + (1 - row_idx) * row_h
        lax = fig.add_axes([label_x, y, label_w, row_h])
        lax.set_xlim(0, 1)
        lax.set_ylim(0, 1)
        lax.text(0.9, 0.5, label, ha="right", va="center", fontsize=fs)
        lax.set_xticks([])
        lax.set_yticks([])
        lax.axis("off")


def plot(save=True):
    off = _load_off()
    secret_to_group = _get_secret_to_group()
    secret_order = sorted(_get_secret_order(off))
    n_secrets = len(secret_order)
    ncols = 5
    nrows = int(np.ceil(n_secrets / ncols))
    n_models = len(MODEL_KEYS)

    # figsize width=15 → scale=6.5/15=0.43 at linewidth
    subplot_size = 2.8
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * subplot_size + 1, nrows * subplot_size + 2),
                             squeeze=False)

    for idx, secret in enumerate(secret_order):
        row, col = divmod(idx, ncols)
        _render_subplot(axes[row][col], off, secret, secret_to_group, title_fontsize=19)  # →8.2pt

    for idx in range(n_secrets, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    for row in range(nrows):
        _set_colored_labels(axes[row][0], axis="y", fontsize=14, short=True)  # →6pt
    for col in range(ncols):
        last_row = min(nrows - 1, (n_secrets - 1) // ncols) if col < n_secrets % ncols or n_secrets % ncols == 0 else nrows - 2
        _set_colored_labels(axes[last_row][col], axis="x", fontsize=16, short=True)  # →6.9pt

    fig.suptitle("Normalized Steganographic Gap",
                 fontsize=28, fontweight="bold", y=1.01)  # →12pt
    fig.text(0.5, 0.98, "rows = receiver, cols = sentinel; alphabetical order",
             ha="center", fontsize=17, color="gray")  # →7.3pt
    plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.04, right=0.96, top=0.96, bottom=0.10)
    _add_legend(fig, cbar_x=0.14, cbar_w=0.68, cbar_y=0.01, cbar_h=0.015, fontscale=2.0)  # ticks→7.7pt

    if save:
        save_plot(fig, "gap_per_secret_grid.png")
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
                             figsize=(ncols * subplot_size + 1, subplot_size + 5),
                             squeeze=False)

    for idx, secret in enumerate(sample):
        _render_subplot(axes[0][idx], off, secret, secret_to_group, title_fontsize=30)

    _set_colored_labels(axes[0][0], axis="y", fontsize=22)
    for col in range(ncols):
        _set_colored_labels(axes[0][col], axis="x", fontsize=22, short=True)

    fig.suptitle("Models can influence codeword selection",
                 fontsize=36, fontweight="bold", y=1.05)
    fig.text(0.5, 0.96, "rows = receiver, cols = sentinel; random sample of 4 secrets",
             ha="center", fontsize=20, color="gray")
    plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.07, right=0.96, top=0.95, bottom=0.36)
    _add_legend(fig, cbar_x=0.14, cbar_w=0.68, cbar_y=0.14, cbar_h=0.04, fontscale=2.6, show_table=True)

    if save:
        save_plot(fig, "gap_per_secret_grid_sample4.png")
    plt.close()


if __name__ == "__main__":
    plot()
    print("Saved full grid.")
    plot_sample4()
    print("Saved sample 4.")
