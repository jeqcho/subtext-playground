"""Mean heatmaps without Optional Ignorance — no clamping at all.

Uses mean(uplift) directly, where uplift = treatment - control (can be negative).
  raw gap = mean(self_uplift) - mean(sentinel_uplift)
  normed  = raw gap / mean(self_uplift)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (MODEL_KEYS, load_per_codeword_deltas, save_plot)
from gap_per_secret_grid_raw import (
    _render_subplot as _render_raw,
    _set_colored_labels as _set_labels_raw,
    _add_legend as _add_legend_raw,
)
from gap_per_secret_grid import (
    _render_subplot as _render_normed,
    _set_colored_labels as _set_labels_normed,
    _add_legend as _add_legend_normed,
)


def _compute_mean_uplift():
    """Return mean(uplift) per (sender, evaluator) — no clamping."""
    df = load_per_codeword_deltas()
    return df.groupby(["sender", "evaluator"])["uplift"].mean()


def _build_raw_off():
    mean_uplift = _compute_mean_uplift()
    self_uplift = {m: mean_uplift[(m, m)] for m in MODEL_KEYS}

    rows = []
    for recv in MODEL_KEYS:
        for sent in MODEL_KEYS:
            if recv == sent:
                continue
            gap = self_uplift[recv] - mean_uplift[(recv, sent)]
            # delta_norm_oi: NaN if self_uplift==0, else gap/self_uplift
            # (needed by _render_raw to detect no-signal)
            su = self_uplift[recv]
            if su == 0:
                dn = 0.0
            else:
                dn = gap / su
            rows.append({
                "receiver": recv,
                "sentinel": sent,
                "codeword": "__nooi__",
                "delta_oi": gap,
                "delta_norm_oi": dn,
            })
    return pd.DataFrame(rows)


def _build_normed_off():
    mean_uplift = _compute_mean_uplift()
    self_uplift = {m: mean_uplift[(m, m)] for m in MODEL_KEYS}

    rows = []
    for recv in MODEL_KEYS:
        for sent in MODEL_KEYS:
            if recv == sent:
                continue
            gap = self_uplift[recv] - mean_uplift[(recv, sent)]
            su = self_uplift[recv]
            if su == 0:
                dn = -np.inf if gap != 0 else np.nan
            else:
                dn = gap / su
            rows.append({
                "receiver": recv,
                "sentinel": sent,
                "codeword": "__nooi__",
                "delta_norm_oi": dn,
            })
    return pd.DataFrame(rows)


def plot_raw(save=True):
    off = _build_raw_off()

    subplot_size = 6.0
    fig, ax = plt.subplots(1, 1,
                           figsize=(subplot_size + 3, subplot_size + 5),
                           squeeze=True)

    _render_raw(ax, off, "__nooi__", {}, title_fontsize=0, vmin=-0.06, vmax=0.06)
    ax.set_title("")

    _set_labels_raw(ax, axis="y", fontsize=14)
    _set_labels_raw(ax, axis="x", fontsize=14, short=True)

    ax.set_ylabel("Receiver", fontsize=15, labelpad=10)
    ax.set_xlabel("Sentinel", fontsize=15, labelpad=10)

    fig.suptitle("Mean Steganographic Gap (Raw, no OI)\n"
                 "mean(uplift) \u2014 no clamping",
                 fontsize=18, fontweight="bold", y=0.95)

    plt.subplots_adjust(left=0.20, right=0.92, top=0.88, bottom=0.32)
    _add_legend_raw(fig, cbar_x=0.27, cbar_w=0.42, cbar_y=0.12, cbar_h=0.03,
                    fontscale=1.5, cbar_vmin=-0.06, cbar_vmax=0.06)

    if save:
        save_plot(fig, "gap_mean_heatmap_raw_nooi.png")
    plt.close()


def plot_normed(save=True):
    off = _build_normed_off()

    subplot_size = 6.0
    fig, ax = plt.subplots(1, 1,
                           figsize=(subplot_size + 3, subplot_size + 5),
                           squeeze=True)

    _render_normed(ax, off, "__nooi__", {}, title_fontsize=0)
    ax.set_title("")

    _set_labels_normed(ax, axis="y", fontsize=14)
    _set_labels_normed(ax, axis="x", fontsize=14, short=True)

    ax.set_ylabel("Receiver", fontsize=15, labelpad=10)
    ax.set_xlabel("Sentinel", fontsize=15, labelpad=10)

    fig.suptitle("Mean Normalized Steganographic Gap (no OI)\n"
                 "mean(uplift) \u2014 no clamping",
                 fontsize=18, fontweight="bold", y=0.95)

    plt.subplots_adjust(left=0.20, right=0.92, top=0.88, bottom=0.32)
    _add_legend_normed(fig, cbar_x=0.27, cbar_w=0.42, cbar_y=0.12, cbar_h=0.03,
                       fontscale=1.5, show_table=True)

    if save:
        save_plot(fig, "gap_mean_heatmap_normed_nooi.png")
    plt.close()


if __name__ == "__main__":
    plot_raw()
    print("Saved raw no-OI heatmap.")
    plot_normed()
    print("Saved normed no-OI heatmap.")
