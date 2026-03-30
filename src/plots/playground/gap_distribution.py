"""Distribution plots for normalized stego gap (OI): histogram, strip, violin, heatmap."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import MODEL_KEYS, PLOTS_DIR, load_per_codeword_deltas, save_plot, style_ax, FAMILY_BOUNDARIES


def _load_off_diagonal():
    df = load_per_codeword_deltas()
    off = df[df["sender"] != df["evaluator"]].copy()
    off = off.rename(columns={"sender": "receiver", "evaluator": "sentinel"})
    return off


def plot_histogram(save=True):
    off = _load_off_diagonal()
    raw = off["delta_norm_oi"].values

    n_nan = np.isnan(raw).sum()
    n_neginf = np.isneginf(raw).sum()
    finite = raw[np.isfinite(raw)]

    fig, ax = plt.subplots(figsize=(10, 6))
    lo = np.floor(finite.min()) - 0.5 if len(finite) > 0 else -1.5
    bins = np.arange(lo, 1.55, 0.25)
    ax.hist(finite, bins=bins, color="tab:blue", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Normalized steganographic gap (opt. ignorance)", fontsize=12)
    ax.set_ylabel("Count of (receiver, sentinel, secret) triplets", fontsize=12)
    ax.set_title("Distribution of Normalized Steganographic Gap\n"
                 f"{len(raw)} off-diagonal triplets ({n_nan} NaN, {n_neginf} −∞ excluded from bars)",
                 fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate the key spikes
    n_one = (finite >= 1.0 - 1e-9).sum()
    ax.annotate(f"gap = 1: {n_one}", xy=(1, n_one), xytext=(0.4, n_one * 0.9),
                fontsize=10, arrowprops=dict(arrowstyle="->", color="gray"))
    # Annotate excluded
    ax.text(0.02, 0.95, f"NaN (0/0): {n_nan}\n−∞ (anti-stego): {n_neginf}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOTS_DIR / "gap_distribution_histogram.png", dpi=150)
    plt.close()


def plot_strip(save=True):
    off = _load_off_diagonal()
    off = off[np.isfinite(off["delta_norm_oi"])]

    # Order by MODEL_KEYS
    off["receiver"] = pd.Categorical(off["receiver"], categories=MODEL_KEYS, ordered=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.stripplot(data=off, x="receiver", y="delta_norm_oi", jitter=0.3, size=2.5,
                  alpha=0.5, color="tab:blue", ax=ax)

    for xv in FAMILY_BOUNDARIES:
        ax.axvline(x=xv, color="gray", linestyle="--", alpha=0.6)

    ax.set_xlabel("Model (as sender and receiver)", fontsize=12)
    ax.set_ylabel("Normalized steganographic gap (opt. ignorance)", fontsize=12)
    ax.set_title("Normalized Stego Gap by Receiver Model\n"
                 "each dot = one (sentinel, secret) pair", fontsize=12)
    ax.set_xticklabels(MODEL_KEYS, rotation=25, ha="right", fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOTS_DIR / "gap_distribution_strip.png", dpi=150)
    plt.close()


def plot_violin(save=True):
    off = _load_off_diagonal()
    off = off[np.isfinite(off["delta_norm_oi"])]
    off["receiver"] = pd.Categorical(off["receiver"], categories=MODEL_KEYS, ordered=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.violinplot(data=off, x="receiver", y="delta_norm_oi", color="tab:blue",
                   inner="box", linewidth=0.8, ax=ax, cut=0)

    for xv in FAMILY_BOUNDARIES:
        ax.axvline(x=xv, color="gray", linestyle="--", alpha=0.6)

    ax.set_xlabel("Model (as sender and receiver)", fontsize=12)
    ax.set_ylabel("Normalized steganographic gap (opt. ignorance)", fontsize=12)
    ax.set_title("Normalized Stego Gap Distribution by Receiver Model\n"
                 "violin width = density; box = IQR + median", fontsize=12)
    ax.set_xticklabels(MODEL_KEYS, rotation=25, ha="right", fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOTS_DIR / "gap_distribution_violin.png", dpi=150)
    plt.close()


def _build_heatmap_matrix(include_inf=False):
    off = _load_off_diagonal()
    if include_inf:
        # Clamp -inf to -1 so they contribute to the mean
        off = off.copy()
        off["delta_norm_oi"] = off["delta_norm_oi"].replace(-np.inf, -1)
        off_use = off[off["delta_norm_oi"].notna()]
    else:
        off_use = off[np.isfinite(off["delta_norm_oi"])]
    pivot = off_use.groupby(["receiver", "codeword"])["delta_norm_oi"].mean().reset_index()
    matrix = pivot.pivot(index="receiver", columns="codeword", values="delta_norm_oi")
    matrix = matrix.reindex(MODEL_KEYS)
    col_order = matrix.mean(axis=0).sort_values(ascending=False).index
    return matrix[col_order]


def plot_heatmap(save=True):
    matrix = _build_heatmap_matrix(include_inf=False)

    fig, ax = plt.subplots(figsize=(20, 7))
    sns.heatmap(matrix, cmap="RdYlGn", center=0, vmax=1, ax=ax,
                xticklabels=True, yticklabels=True,
                cbar_kws={"label": "Mean normalized stego gap (OI)"})

    ax.set_xlabel("Secret (sorted by mean gap descending)", fontsize=12)
    ax.set_ylabel("Model (as sender and receiver)", fontsize=12)
    ax.set_title("Mean Normalized Steganographic Gap by (Receiver, Secret)\n"
                 "averaged across 8 sentinels; optional ignorance", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()
    if save:
        save_plot(fig, "gap_distribution_heatmap.png")
    plt.close()


def plot_heatmap_with_neg(save=True):
    from matplotlib.colors import LinearSegmentedColormap
    off = _load_off_diagonal()

    # Separate -inf from finite for special treatment
    off = off.copy()
    has_inf = off.groupby(["receiver", "codeword"])["delta_norm_oi"].apply(lambda s: np.isneginf(s).any())

    # Clamp -inf to -1 for mean computation
    off["delta_norm_oi"] = off["delta_norm_oi"].replace(-np.inf, -1)
    off_use = off[off["delta_norm_oi"].notna()]
    pivot = off_use.groupby(["receiver", "codeword"])["delta_norm_oi"].mean().reset_index()
    matrix = pivot.pivot(index="receiver", columns="codeword", values="delta_norm_oi")
    matrix = matrix.reindex(MODEL_KEYS)
    col_order = matrix.mean(axis=0).sort_values(ascending=False).index
    matrix = matrix[col_order]

    # Build a mask for cells where any sentinel had -inf (mark as "has -inf")
    inf_pivot = has_inf.reset_index()
    inf_pivot.columns = ["receiver", "codeword", "has_inf"]
    inf_matrix = inf_pivot.pivot(index="receiver", columns="codeword", values="has_inf")
    inf_matrix = inf_matrix.reindex(index=MODEL_KEYS, columns=col_order).fillna(False)

    # Custom colormap: bright red for negatives, yellow at 0, green at 1
    cmap = LinearSegmentedColormap.from_list("red_ylgn", [
        (0.0, "#e74c3c"),
        (0.499, "#e74c3c"),
        (0.5, "#ffffbf"),
        (0.75, "#a6d96a"),
        (1.0, "#1a9850"),
    ])

    fig, ax = plt.subplots(figsize=(20, 7))
    sns.heatmap(matrix, cmap=cmap, vmin=-1, vmax=1, ax=ax,
                xticklabels=True, yticklabels=True,
                cbar_kws={"label": "Mean normalized stego gap (OI)"})

    # Overlay dark red patches for cells that contained -inf
    inf_color = "#8b0000"
    for i in range(len(MODEL_KEYS)):
        for j in range(len(col_order)):
            if inf_matrix.iloc[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                             facecolor=inf_color, edgecolor="none", zorder=2))

    # Separate legend patch for -inf, positioned right below the colorbar
    from matplotlib.patches import Patch
    cbar = ax.collections[0].colorbar
    cbar_ax = cbar.ax
    # Add a small axes below the colorbar for the patch
    fig_coords = cbar_ax.get_position()
    patch_ax = fig.add_axes([fig_coords.x0, fig_coords.y0 - 0.08, fig_coords.width, 0.04])
    patch_ax.set_xlim(0, 1)
    patch_ax.set_ylim(0, 1)
    patch_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=inf_color, edgecolor="black", linewidth=0.5))
    patch_ax.text(1.15, 0.5, "contains −∞\n(anti-stego)", transform=patch_ax.transAxes,
                  fontsize=9, va="center", ha="left")
    patch_ax.set_xticks([])
    patch_ax.set_yticks([])

    ax.set_xlabel("Secret (sorted by mean gap descending)", fontsize=12)
    ax.set_ylabel("Model (as sender and receiver)", fontsize=12)
    ax.set_title("Mean Normalized Steganographic Gap by (Receiver, Secret)\n"
                 "averaged across sentinels; dark red = contains −∞ (anti-stego)", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()
    if save:
        save_plot(fig, "gap_distribution_heatmap_with_neg.png")
    plt.close()


if __name__ == "__main__":
    plot_histogram()
    print("Saved histogram.")
    plot_strip()
    print("Saved strip.")
    plot_violin()
    print("Saved violin.")
    plot_heatmap()
    print("Saved heatmap.")
