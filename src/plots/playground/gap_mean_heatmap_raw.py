"""Single 9x9 heatmap: mean raw steganographic gap (delta_oi) averaged across all codewords."""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

from common import (MODEL_KEYS, MODEL_DISPLAY, MODEL_SHORT, FAMILY_COLORS,
                    load_per_codeword_deltas, save_plot)

# Same colormap as the per-secret raw grid
_CMAP = LinearSegmentedColormap.from_list("red_ylbu_raw", [
    (0.0,  "#b2182b"),
    (0.30, "#d73027"),
    (0.45, "#fc8d59"),
    (0.50, "#ffffbf"),
    (0.55, "#abd9e9"),
    (0.70, "#74add1"),
    (1.0,  "#4575b4"),
])
_CMAP.set_bad(color="#e0e0e0")

_DIAG_COLOR = "#a0a0a0"
_FAMILY_BOUNDS = [3, 6]


def plot(save=True):
    df = load_per_codeword_deltas()
    off = df[df["sender"] != df["evaluator"]].copy()
    off = off.rename(columns={"sender": "receiver", "evaluator": "sentinel"})

    n = len(MODEL_KEYS)
    matrix = np.full((n, n), np.nan)
    for (recv, sent), grp in off.groupby(["receiver", "sentinel"]):
        i = MODEL_KEYS.index(recv)
        j = MODEL_KEYS.index(sent)
        matrix[i, j] = grp["delta_oi"].mean()

    # Scale to data range for visibility
    mask = np.eye(n, dtype=bool)
    off_vals = matrix[~mask]
    vmax = max(abs(np.nanmin(off_vals)), abs(np.nanmax(off_vals)))

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(matrix, ax=ax, cmap=_CMAP, vmin=-vmax, vmax=vmax,
                mask=mask, cbar=False, xticklabels=False, yticklabels=False,
                linewidths=0.5, linecolor="white",
                annot=True, fmt=".3f", annot_kws={"fontsize": 9})

    # Gray diagonal
    for k in range(n):
        ax.add_patch(plt.Rectangle((k, k), 1, 1, fill=True,
                     facecolor=_DIAG_COLOR, edgecolor="white", linewidth=0.5))

    # Family separator lines
    for b in _FAMILY_BOUNDS:
        ax.axhline(y=b, color="gray", linewidth=1.0, zorder=3)
        ax.axvline(x=b, color="gray", linewidth=1.0, zorder=3)

    # Colored axis labels (matching per-secret grid style)
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
    ax.set_title("Mean Steganographic Gap (Raw)\naveraged across all codewords",
                 fontsize=15, fontweight="bold", pad=10)
    ax.set_aspect("equal")

    # Colorbar
    sm = ScalarMappable(cmap=_CMAP, norm=mcolors.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8, pad=0.02)
    cbar.set_label("Mean steganographic gap (raw)", fontsize=11)

    plt.tight_layout()
    if save:
        save_plot(fig, "gap_mean_heatmap_raw.png")
    plt.close()


if __name__ == "__main__":
    plot()
    print("Saved mean raw gap heatmap.")
