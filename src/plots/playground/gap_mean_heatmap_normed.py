"""Single 9x9 heatmap: mean delta_oi / mean receiver self-uplift, using the original normalized color scheme."""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

from common import (MODEL_KEYS, MODEL_DISPLAY, MODEL_SHORT, FAMILY_COLORS,
                    load_per_codeword_deltas, save_plot)

# Same colormap as the original normalized grid
_Z = 8.0 / 9.0
_CMAP = LinearSegmentedColormap.from_list("red_ylbu", [
    (0.0, "#d73027"),
    (_Z - 0.001, "#d73027"),
    (_Z, "#ffffbf"),
    ((_Z + 1.0) / 2, "#abd9e9"),
    (1.0, "#74add1"),
])
_CMAP.set_bad(color="#e0e0e0")

_INF_COLOR = "#8b0000"
_DIAG_COLOR = "#a0a0a0"
_NAN_COLOR = "#e0e0e0"
_FAMILY_BOUNDS = [3, 6]


def plot(save=True):
    df = load_per_codeword_deltas()

    # Compute mean receiver self-uplift per receiver (averaged across codewords)
    diag = df[df["sender"] == df["evaluator"]].copy()
    mean_self_uplift = diag.groupby("sender")["uplift_oi"].mean().to_dict()

    # Compute mean delta_oi per (receiver, sentinel) pair
    off = df[df["sender"] != df["evaluator"]].copy()
    off = off.rename(columns={"sender": "receiver", "evaluator": "sentinel"})

    n = len(MODEL_KEYS)
    matrix = np.full((n, n), np.nan)
    is_neginf = np.zeros((n, n), dtype=bool)
    for (recv, sent), grp in off.groupby(["receiver", "sentinel"]):
        i = MODEL_KEYS.index(recv)
        j = MODEL_KEYS.index(sent)
        mean_delta = grp["delta_oi"].mean()
        recv_uplift = mean_self_uplift[recv]
        if recv_uplift == 0:
            if mean_delta < 0:
                is_neginf[i, j] = True
            # else NaN (0/0)
        else:
            matrix[i, j] = mean_delta / recv_uplift

    fig, ax = plt.subplots(figsize=(7, 6))

    mask = np.eye(n, dtype=bool)
    sns.heatmap(matrix, ax=ax, cmap=_CMAP, vmin=-8, vmax=1,
                mask=mask, cbar=False, xticklabels=False, yticklabels=False,
                linewidths=0.5, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"fontsize": 9})

    # Gray diagonal
    for k in range(n):
        ax.add_patch(plt.Rectangle((k, k), 1, 1, fill=True,
                     facecolor=_DIAG_COLOR, edgecolor="white", linewidth=0.5))

    # Dark red for -inf
    for i in range(n):
        for j in range(n):
            if is_neginf[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                             facecolor=_INF_COLOR, edgecolor="white", linewidth=0.5, zorder=2))
                ax.text(j + 0.5, i + 0.5, "$-\\infty$", ha="center", va="center",
                        fontsize=9, color="white")

    # Family separator lines
    for b in _FAMILY_BOUNDS:
        ax.axhline(y=b, color="gray", linewidth=1.0, zorder=3)
        ax.axvline(x=b, color="gray", linewidth=1.0, zorder=3)

    # Colored axis labels
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
    ax.set_title("Mean Normalized Steganographic Gap\nmean(delta_oi) / mean(receiver self-uplift)",
                 fontsize=15, fontweight="bold", pad=10)
    ax.set_aspect("equal")

    # Colorbar (same as original)
    sm = ScalarMappable(cmap=_CMAP, norm=mcolors.Normalize(vmin=-8, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8, pad=0.02,
                        extend="both", extendfrac=0.04)
    cbar.set_ticks([-8, -6, -4, -2, 0, 1])
    cbar.set_label("Normalized steganographic gap", fontsize=11)

    plt.tight_layout()
    if save:
        save_plot(fig, "gap_mean_heatmap_normed.png")
    plt.close()


if __name__ == "__main__":
    plot()
    print("Saved mean normalized gap heatmap.")
