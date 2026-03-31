"""Block-stack chart: each block = one (secret, sentinel) pair at gap=1, colored by self-uplift."""
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from common import MODEL_KEYS, PLOTS_DIR, add_self_uplift, load_per_codeword_deltas, save_plot, style_ax


def plot(full_scale=False, save=True, threshold="eq1"):
    """threshold: 'eq1' for gap=1, 'gt0' for gap>0."""
    df = load_per_codeword_deltas()
    off = df[df["sender"] != df["evaluator"]]
    if threshold == "gt0":
        perfect = off[off["delta_norm_oi"] > 1e-9].copy()
    else:
        perfect = off[off["delta_norm_oi"] >= 1.0 - 1e-9].copy()
    perfect = add_self_uplift(perfect)

    n_secrets = df["codeword"].nunique()
    n_sentinels = len(MODEL_KEYS) - 1
    total = n_secrets * n_sentinels

    counts = perfect.groupby("sender").size().reindex(MODEL_KEYS, fill_value=0)
    pct = counts / total * 100

    norm = mcolors.Normalize(vmin=0, vmax=perfect["self_uplift_oi"].max())
    cmap = cm.viridis

    scale = 100.0 / total
    block_height = scale
    block_width = 0.5

    fig, ax = plt.subplots(figsize=(12, 8))
    x_positions = np.arange(len(MODEL_KEYS))

    max_pct = 0
    for idx, m in enumerate(MODEL_KEYS):
        subset = perfect[perfect["sender"] == m].sort_values("self_uplift_oi")
        n = len(subset)
        if n == 0:
            continue
        for i, (_, row) in enumerate(subset.iterrows()):
            y_bottom = i * scale
            color = cmap(norm(row["self_uplift_oi"]))
            rect = plt.Rectangle(
                (x_positions[idx] - block_width / 2, y_bottom),
                block_width, block_height,
                facecolor=color, edgecolor="none", zorder=3)
            ax.add_patch(rect)
        max_pct = max(max_pct, n * scale)

    for idx, m in enumerate(MODEL_KEYS):
        y_top = counts[m] * scale
        ax.text(x_positions[idx], y_top + 1.5, f"{pct[m]:.0f}%", ha="center", fontsize=10)

    if threshold == "gt0":
        baseline = 37.5
        baseline_label = "37.5% random baseline"
        gap_label = "> 0"
    else:
        baseline = 25
        baseline_label = "25% random baseline"
        gap_label = "= 1"
    ax.axhline(y=baseline, color="tab:red", linestyle="--", alpha=0.7, zorder=2)
    ax.text(1.5, baseline + 1.5, baseline_label, ha="center", fontsize=9, color="tab:red")

    ylim = 100 if full_scale else max(max_pct + 3, baseline + 3)
    style_ax(ax,
             ylabel=f"% of (secret, sentinel) pairs with normalized stego gap {gap_label}",
             title=f"Frequency of Normalized Steganographic Gap {gap_label} by Model",
             subtitle="(receiver uplift > 0, sentinel uplift ≤ 0)\neach block = one (secret, sentinel) pair; color = self-uplift (opt. ignorance)",
             ylim_max=ylim)
    ax.set_xlim(-0.6, len(MODEL_KEYS) - 0.4)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, shrink=0.6)
    cbar.set_label("Self-uplift (opt. ignorance)", fontsize=11)

    plt.tight_layout()
    if save:
        scale_suffix = "_full" if full_scale else ""
        thresh_suffix = "_gt0" if threshold == "gt0" else ""
        save_plot(fig, f"perfect_privacy_fraction_by_sender_blocks{scale_suffix}{thresh_suffix}.png")
    plt.close()


if __name__ == "__main__":
    plot(full_scale=False)
    plot(full_scale=True)
    plot(full_scale=True, threshold="gt0")
    print("Saved all.")
