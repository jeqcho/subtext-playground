"""Scatter plot: mean self-uplift vs gap=1 rate, one point per model."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from common import MODEL_KEYS, PLOTS_DIR, load_per_codeword_deltas, style_ax

FAMILY_COLORS = {
    "haiku-4.5": "#e67e22", "sonnet-4.6": "#e67e22", "opus-4.6": "#e67e22",
    "gpt-5.4-nano": "#2ecc71", "gpt-5.4-mini": "#2ecc71", "gpt-5.4": "#2ecc71",
    "gemini-3.1-flash-lite": "#3498db", "gemini-3-flash": "#3498db", "gemini-3.1-pro": "#3498db",
}
FAMILY_LABELS = {"#e67e22": "Anthropic", "#2ecc71": "OpenAI", "#3498db": "Google"}


def plot(save=True):
    df = load_per_codeword_deltas()

    # Mean self-uplift per model
    diag = df[df["sender"] == df["evaluator"]]
    mean_self_uplift = diag.groupby("sender")["uplift_oi"].mean()

    # Gap=1 rate per model
    off = df[df["sender"] != df["evaluator"]]
    n_sentinels = len(MODEL_KEYS) - 1
    n_secrets = df["codeword"].nunique()
    total = n_secrets * n_sentinels

    perfect = off[off["delta_norm_oi"] >= 1.0 - 1e-9]
    counts = perfect.groupby("sender").size().reindex(MODEL_KEYS, fill_value=0)
    gap1_rate = counts / total * 100

    fig, ax = plt.subplots(figsize=(10, 8))

    for m in MODEL_KEYS:
        ax.scatter(mean_self_uplift[m], gap1_rate[m], s=120, c=FAMILY_COLORS[m],
                   edgecolors="white", linewidths=0.8, zorder=3)
        ax.annotate(m, (mean_self_uplift[m], gap1_rate[m]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=l)
               for c, l in FAMILY_LABELS.items()]
    ax.legend(handles=handles, fontsize=11, loc="upper left")

    ax.set_xlabel("Mean self-uplift (opt. ignorance)", fontsize=12)
    ax.set_ylabel("% of (secret, sentinel) pairs with normalized stego gap = 1", fontsize=12)
    ax.set_title("Self-Decoding Strength vs. Signal Privacy\n"
                 "each point = one model (as sender and receiver)", fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOTS_DIR / "self_uplift_vs_gap1_rate.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    plot()
    print("Saved.")
