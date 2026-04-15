"""Single 9x9 heatmap: mean raw steganographic gap (delta_oi) averaged across all codewords.

Uses mean(max(0, uplift)) — clamped before averaging (pre-computed in CSV as uplift_oi).
Reuses rendering from gap_per_secret_grid_raw.py for consistent style.
"""
import matplotlib.pyplot as plt
import numpy as np

from common import (MODEL_KEYS, load_per_codeword_deltas, save_plot)
from gap_per_secret_grid_raw import (
    _render_subplot, _set_colored_labels, _add_legend,
)


def _build_mean_off():
    """Build a fake 'off' dataframe with one row per (receiver, sentinel),
    using mean(delta_oi) across codewords."""
    import pandas as pd

    df = load_per_codeword_deltas()
    off = df[df["sender"] != df["evaluator"]].copy()
    off = off.rename(columns={"sender": "receiver", "evaluator": "sentinel"})

    # Average delta_oi and delta_norm_oi per (receiver, sentinel)
    agg = off.groupby(["receiver", "sentinel"]).agg(
        delta_oi=("delta_oi", "mean"),
        delta_norm_oi=("delta_norm_oi", lambda s: np.nan if s.isna().all() else s.mean()),
    ).reset_index()
    agg["codeword"] = "__mean__"
    return agg


def plot(save=True):
    off = _build_mean_off()

    subplot_size = 6.0
    fig, ax = plt.subplots(1, 1,
                           figsize=(subplot_size + 3, subplot_size + 5),
                           squeeze=True)

    _render_subplot(ax, off, "__mean__", {}, title_fontsize=0, vmin=-0.05, vmax=0.05)
    ax.set_title("")

    _set_colored_labels(ax, axis="y", fontsize=14)
    _set_colored_labels(ax, axis="x", fontsize=14, short=True)

    ax.set_ylabel("Receiver", fontsize=15, labelpad=10)
    ax.set_xlabel("Sentinel", fontsize=15, labelpad=10)

    fig.suptitle("Mean Steganographic Gap (Raw)\n"
                 "mean(max(0, uplift)) \u2014 clamped before averaging",
                 fontsize=18, fontweight="bold", y=0.95)

    plt.subplots_adjust(left=0.20, right=0.92, top=0.88, bottom=0.32)
    _add_legend(fig, cbar_x=0.27, cbar_w=0.42, cbar_y=0.12, cbar_h=0.03,
                fontscale=1.5, cbar_vmin=-0.05, cbar_vmax=0.05)

    if save:
        save_plot(fig, "gap_mean_heatmap_raw.png")
    plt.close()


if __name__ == "__main__":
    plot()
    print("Saved mean raw gap heatmap.")
