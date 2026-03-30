"""Mega grid: one 9x9 heatmap per secret (receiver × sentinel), showing normalized stego gap."""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from common import MODEL_KEYS, PLOTS_DIR, load_per_codeword_deltas


def plot(save=True):
    df = load_per_codeword_deltas()
    off = df[df["sender"] != df["evaluator"]].copy()
    off = off.rename(columns={"sender": "receiver", "evaluator": "sentinel"})

    # Sort secrets by mean finite gap descending
    finite = off[np.isfinite(off["delta_norm_oi"])]
    secret_order = finite.groupby("codeword")["delta_norm_oi"].mean().sort_values(ascending=False).index.tolist()

    n_secrets = len(secret_order)
    ncols = 8
    nrows = int(np.ceil(n_secrets / ncols))
    n_models = len(MODEL_KEYS)

    # Short model labels
    short = [m.split("-")[0][:3] + m.split("-")[-1][:2] for m in MODEL_KEYS]

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5 + 2, nrows * 3),
                             squeeze=False)

    # Shared color normalization
    from matplotlib.colors import LinearSegmentedColormap
    # Data range [-8, 1] -> colormap [0, 1]
    # -8 -> 0.0, 0 -> 8/9, 1 -> 1.0
    z = 8.0 / 9.0  # where 0 maps in [0,1]
    cmap = LinearSegmentedColormap.from_list("red_ylgn", [
        (0.0, "#e74c3c"),
        (z - 0.001, "#e74c3c"),  # flat red for all negatives
        (z, "#ffffbf"),           # yellow at 0
        ((z + 1.0) / 2, "#a6d96a"),
        (1.0, "#1a9850"),
    ])
    cmap.set_bad(color="#e0e0e0")  # NaN (0/0) -> light gray

    for idx, secret in enumerate(secret_order):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        # Build 9x9 matrix; keep NaN as NaN (white), track -inf for dark red overlay
        subset = off[off["codeword"] == secret]
        matrix = np.full((n_models, n_models), np.nan)
        is_neginf = np.zeros((n_models, n_models), dtype=bool)
        for _, r in subset.iterrows():
            i = MODEL_KEYS.index(r["receiver"])
            j = MODEL_KEYS.index(r["sentinel"])
            val = r["delta_norm_oi"]
            if np.isneginf(val):
                is_neginf[i, j] = True
                matrix[i, j] = np.nan  # will be overlaid with dark red patch
            elif np.isfinite(val):
                matrix[i, j] = val
            # else: NaN stays NaN -> white

        # Mask diagonal
        mask = np.eye(n_models, dtype=bool)

        import seaborn as sns
        sns.heatmap(matrix, ax=ax, cmap=cmap, vmin=-8, vmax=1,
                    mask=mask, cbar=False, xticklabels=False, yticklabels=False,
                    linewidths=0.3, linecolor="white")

        # Gray diagonal
        for k in range(n_models):
            ax.add_patch(plt.Rectangle((k, k), 1, 1, fill=True,
                         facecolor="#a0a0a0", edgecolor="white", linewidth=0.3))

        # Dark red overlay for -inf cells
        for i in range(n_models):
            for j in range(n_models):
                if is_neginf[i, j]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                                 facecolor="#8b0000", edgecolor="white", linewidth=0.3, zorder=2))

        ax.set_title(secret, fontsize=9, fontweight="bold")
        ax.set_aspect("equal")

    # Turn off unused axes
    for idx in range(n_secrets, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Add tick labels to leftmost and bottom subplots
    for row in range(nrows):
        axes[row][0].set_yticks(np.arange(n_models) + 0.5)
        axes[row][0].set_yticklabels(MODEL_KEYS, fontsize=5, rotation=0)
    for col in range(ncols):
        last_row = min(nrows - 1, (n_secrets - 1) // ncols) if col < n_secrets % ncols or n_secrets % ncols == 0 else nrows - 2
        axes[last_row][col].set_xticks(np.arange(n_models) + 0.5)
        axes[last_row][col].set_xticklabels(MODEL_KEYS, fontsize=5, rotation=45, ha="right")

    fig.suptitle("Normalized Steganographic Gap per Secret\n"
                 "rows = receiver, cols = sentinel; sorted by mean gap descending",
                 fontsize=13, y=1.01)

    plt.tight_layout(rect=[0, 0.08, 1.0, 0.98])

    # Horizontal colorbar at lower right
    from matplotlib.cm import ScalarMappable
    # Dark red patch for -inf, then colorbar
    inf_color = "#8b0000"
    patch_w = 0.03
    gap = 0.005
    cbar_x = 0.55
    cbar_w = 0.3

    # -inf patch to the left of colorbar
    inf_ax = fig.add_axes([cbar_x - patch_w - gap, 0.02, patch_w, 0.02])
    inf_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=inf_color, edgecolor="black", linewidth=0.5))
    inf_ax.set_xlim(0, 1)
    inf_ax.set_ylim(0, 1)
    inf_ax.set_xticks([0.5])
    inf_ax.set_xticklabels(["−∞"], fontsize=8)
    inf_ax.set_yticks([])
    inf_ax.tick_params(length=0)

    # Colorbar
    cbar_ax = fig.add_axes([cbar_x, 0.02, cbar_w, 0.02])
    sm = ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=-8, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="min",
                        extendfrac=0.06)
    cbar.set_ticks([-8, -6, -4, -2, 0, 1])
    cbar.set_label("Normalized stego gap (OI)", fontsize=9)

    # Legend patches
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#a0a0a0", edgecolor="black", linewidth=0.5, label="diagonal (self)"),
        Patch(facecolor="#e0e0e0", edgecolor="black", linewidth=0.5, label="NaN (0/0)"),
    ]
    fig.legend(handles=legend_patches, loc="lower left", fontsize=8, ncol=2,
               bbox_to_anchor=(0.02, 0.005))
    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOTS_DIR / "gap_per_secret_grid.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot()
    print("Saved.")
