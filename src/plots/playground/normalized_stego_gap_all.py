"""Bar chart: normalized stego gap (OI) for all (receiver, sentinel, secret) triplets."""
import matplotlib.pyplot as plt
import numpy as np

from common import MODEL_KEYS, PLOTS_DIR, load_per_codeword_deltas


def plot(save=True):
    df = load_per_codeword_deltas()
    off = df[df["sender"] != df["evaluator"]].copy()
    off = off.sort_values("delta_norm_oi", ascending=False).reset_index(drop=True)

    n = len(off)
    fig_width = max(60, n * 0.025)
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    x = np.arange(n)

    ax.bar(x, off["delta_norm_oi"].values, width=1.0, color="tab:blue", alpha=0.8)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Normalized steganographic gap (opt. ignorance)", fontsize=14)
    ax.set_xlabel("(receiver, sentinel, secret) sorted by normalized stego gap", fontsize=14)
    ax.set_title("Normalized Steganographic Gap for All (Receiver, Sentinel, Secret) Triplets\n"
                 f"{n} off-diagonal triplets, sorted descending", fontsize=14)
    ax.set_xlim(-1, n)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks([])

    plt.tight_layout()
    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOTS_DIR / "normalized_stego_gap_all.png", dpi=100)
    plt.close()


if __name__ == "__main__":
    plot()
    print("Saved.")
