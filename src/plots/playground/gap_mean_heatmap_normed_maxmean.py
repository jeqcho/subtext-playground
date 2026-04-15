"""Single 9x9 heatmap: normalized stego gap using max(0, mean(uplift)).

Reuses rendering from gap_per_secret_grid.py but with aggregated data
(one heatmap instead of per-codeword grid).
"""
import matplotlib.pyplot as plt
import numpy as np

from common import (MODEL_KEYS, load_per_codeword_deltas, save_plot)
from gap_per_secret_grid import (
    _render_subplot, _set_colored_labels, _add_legend, _FAMILY_BOUNDS,
)


def _build_mean_off():
    """Build a fake 'off' dataframe with one row per (receiver, sentinel),
    using max(0, mean(uplift)) aggregation."""
    import pandas as pd

    df = load_per_codeword_deltas()

    # max(0, mean(raw uplift)) per (sender, evaluator)
    mean_uplift = df.groupby(["sender", "evaluator"])["uplift"].mean()
    mean_uplift_oi = mean_uplift.clip(lower=0)

    self_uplift = {m: mean_uplift_oi[(m, m)] for m in MODEL_KEYS}

    rows = []
    for recv in MODEL_KEYS:
        for sent in MODEL_KEYS:
            if recv == sent:
                continue
            gap = self_uplift[recv] - mean_uplift_oi[(recv, sent)]
            if self_uplift[recv] == 0:
                if gap < 0:
                    dn_oi = -np.inf
                else:
                    dn_oi = np.nan  # 0/0
            else:
                dn_oi = gap / self_uplift[recv]
            rows.append({
                "receiver": recv,
                "sentinel": sent,
                "codeword": "__mean__",
                "delta_norm_oi": dn_oi,
            })
    return pd.DataFrame(rows)


def plot(save=True):
    off = _build_mean_off()

    subplot_size = 6.0
    fig, ax = plt.subplots(1, 1,
                           figsize=(subplot_size + 3, subplot_size + 5),
                           squeeze=True)

    _render_subplot(ax, off, "__mean__", {}, title_fontsize=0)
    ax.set_title("")  # no per-codeword title

    _set_colored_labels(ax, axis="y", fontsize=14)
    _set_colored_labels(ax, axis="x", fontsize=14, short=True)

    ax.set_ylabel("Receiver", fontsize=15, labelpad=10)
    ax.set_xlabel("Sentinel", fontsize=15, labelpad=10)

    fig.suptitle("Mean Normalized Steganographic Gap\n"
                 "max(0, mean(uplift)) \u2014 clamped after averaging",
                 fontsize=18, fontweight="bold", y=0.95)

    plt.subplots_adjust(left=0.20, right=0.92, top=0.88, bottom=0.32)
    _add_legend(fig, cbar_x=0.27, cbar_w=0.42, cbar_y=0.12, cbar_h=0.03,
                fontscale=1.5, show_table=True)

    if save:
        save_plot(fig, "gap_mean_heatmap_normed_maxmean.png")
    plt.close()


if __name__ == "__main__":
    plot()
    print("Saved max(mean()) normalized gap heatmap.")
