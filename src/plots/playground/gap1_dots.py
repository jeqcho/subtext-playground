"""Dot-stack chart: each dot = one (secret, sentinel) pair at gap=1, colored by self-uplift."""
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from common import MODEL_KEYS, PLOTS_DIR, add_self_uplift, load_per_codeword_deltas, style_ax


def plot(save=True):
    df = load_per_codeword_deltas()
    off = df[df["sender"] != df["evaluator"]]
    perfect = off[off["delta_norm_oi"] >= 1.0 - 1e-9].copy()
    perfect = add_self_uplift(perfect)

    n_secrets = df["codeword"].nunique()
    n_sentinels = len(MODEL_KEYS) - 1
    total = n_secrets * n_sentinels

    counts = perfect.groupby("sender").size().reindex(MODEL_KEYS, fill_value=0)
    pct = counts / total * 100

    norm = mcolors.Normalize(vmin=0, vmax=perfect["self_uplift_oi"].max())
    cmap = cm.viridis

    # Scale factor: map count -> percentage
    scale = 100.0 / total

    fig, ax = plt.subplots(figsize=(12, 8))
    x_positions = np.arange(len(MODEL_KEYS))
    dot_size = 28

    max_pct = 0
    for idx, m in enumerate(MODEL_KEYS):
        subset = perfect[perfect["sender"] == m].sort_values("self_uplift_oi")
        n = len(subset)
        if n == 0:
            continue
        dot_x = [x_positions[idx]] * n
        dot_y = [(i + 1) * scale for i in range(n)]
        max_pct = max(max_pct, dot_y[-1])
        colors = [cmap(norm(row["self_uplift_oi"])) for _, row in subset.iterrows()]
        ax.scatter(dot_x, dot_y, s=dot_size, c=colors, edgecolors="white", linewidths=0.3, zorder=3)

    for idx, m in enumerate(MODEL_KEYS):
        y_top = counts[m] * scale
        ax.text(x_positions[idx], y_top + 1.0, f"{pct[m]:.0f}%", ha="center", fontsize=10)

    style_ax(ax,
             ylabel="% of (secret, sentinel) pairs with normalized stego gap = 1",
             title="Frequency of Maximum Normalized Steganographic Gap by Model",
             subtitle="each dot = one (secret, sentinel) pair; color = self-uplift (opt. ignorance)",
             ylim_max=max_pct + 3)
    ax.set_xlim(-0.6, len(MODEL_KEYS) - 0.4)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, shrink=0.6)
    cbar.set_label("Self-uplift (opt. ignorance)", fontsize=11)

    plt.tight_layout()
    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOTS_DIR / "perfect_privacy_fraction_by_sender_dots.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    plot()
    print("Saved.")
