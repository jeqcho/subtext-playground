"""Bar chart: % of (secret, sentinel) pairs with normalized stego gap = 1, by model."""
import matplotlib.pyplot as plt
import numpy as np

from common import MODEL_KEYS, PLOTS_DIR, load_per_codeword_deltas, style_ax


def plot(save=True):
    df = load_per_codeword_deltas()
    off = df[df["sender"] != df["evaluator"]]
    perfect = off[off["delta_norm_oi"] >= 1.0 - 1e-9]
    counts = perfect.groupby("sender").size().reindex(MODEL_KEYS, fill_value=0)

    n_secrets = df["codeword"].nunique()
    n_sentinels = len(MODEL_KEYS) - 1
    total = n_secrets * n_sentinels
    pct = counts / total * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(MODEL_KEYS))
    bars = ax.bar(x, pct.values, color="tab:blue", edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.0f%%", fontsize=10, padding=2)

    style_ax(ax,
             ylabel="% of (secret, sentinel) pairs with normalized stego gap = 1",
             title="Frequency of Maximum Normalized Steganographic Gap by Model",
             subtitle=f"({n_secrets} secrets \u00d7 {n_sentinels} sentinels = {total} pairs per model)",
             ylim_max=min(100, pct.max() * 1.3))

    plt.tight_layout()
    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOTS_DIR / "perfect_privacy_fraction_by_sender.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    plot()
    print("Saved.")
