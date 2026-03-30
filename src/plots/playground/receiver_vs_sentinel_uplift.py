"""Bar charts comparing receiver uplift vs mean sentinel uplift per (receiver, secret) pair."""
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from codeword_sort.noun_analyze import load_data, compute_metrics
from codeword_sort.noun_scan_v1_full import GROUPS, MODELS

from common import MODEL_KEYS, PLOTS_DIR, save_plot


def _build_data():
    outputs_dir = Path(__file__).resolve().parent.parent.parent.parent / "outputs" / "noun_scan_v1_full"
    model_keys = list(MODELS.keys())
    prompts, evals = load_data(outputs_dir)
    acc = compute_metrics(prompts, evals, GROUPS)

    rows = []
    for gi, group in enumerate(GROUPS):
        for secret in group:
            for receiver in model_keys:
                t_recv = acc[gi][receiver][receiver][secret].get("treatment", 0)
                c_recv = acc[gi][receiver][receiver][secret].get("control", 0)
                recv_uplift = max(0, t_recv - c_recv)

                sentinel_uplifts = []
                for sentinel in model_keys:
                    if sentinel == receiver:
                        continue
                    t_sent = acc[gi][receiver][sentinel][secret].get("treatment", 0)
                    c_sent = acc[gi][receiver][sentinel][secret].get("control", 0)
                    sentinel_uplifts.append(max(0, t_sent - c_sent))

                su = np.array(sentinel_uplifts)
                rows.append({
                    "receiver": receiver,
                    "secret": secret,
                    "group_idx": gi,
                    "receiver_uplift": recv_uplift,
                    "sentinel_uplift": np.mean(su),
                    "sentinel_uplift_se": np.std(su, ddof=1) / np.sqrt(len(su)),
                })

    import pandas as pd
    df = pd.DataFrame(rows)
    return df.sort_values("receiver_uplift", ascending=False).reset_index(drop=True)


def plot_all(save=True):
    df = _build_data()
    n = len(df)

    fig_width = max(40, n * 0.06)
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    x = np.arange(n)
    width = 0.4

    ax.bar(x - width / 2, df["receiver_uplift"].values, width, color="tab:blue", alpha=0.8, label="Receiver uplift (OI)")
    ax.bar(x + width / 2, df["sentinel_uplift"].values, width, color="tab:orange", alpha=0.8, label="Mean sentinel uplift (OI)")

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Uplift (optional ignorance)", fontsize=14)
    ax.set_xlabel("(receiver, secret) sorted by receiver uplift", fontsize=14)
    ax.set_title("Receiver vs. Sentinel Uplift by (Receiver, Secret)\n"
                 f"{n} pairs; sentinel uplift = mean across 8 sentinels; both with optional ignorance", fontsize=14)
    ax.legend(fontsize=12, loc="upper right")
    ax.set_xlim(-1, n)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks([])

    plt.tight_layout()
    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOTS_DIR / "receiver_vs_sentinel_uplift.png", dpi=100)
    plt.close()


def plot_top(top_n=10, save=True):
    df = _build_data().head(top_n)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(df))
    width = 0.35

    bars_recv = ax.bar(x - width / 2, df["receiver_uplift"].values, width, color="tab:blue", alpha=0.8, label="Receiver uplift (OI)")
    bars_sent = ax.bar(x + width / 2, df["sentinel_uplift"].values, width,
                       yerr=df["sentinel_uplift_se"].values, capsize=3,
                       color="tab:orange", alpha=0.8, error_kw={"linewidth": 1.2},
                       label="Mean sentinel uplift (OI) \u00b1 SE")

    ax.bar_label(bars_recv, fmt="%.2f", fontsize=9, padding=2)
    ax.bar_label(bars_sent, fmt="%.2f", fontsize=9, padding=2)

    labels = [f"{r.receiver} / {r.secret}" for _, r in df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=30, ha="right")
    ax.set_ylabel("Uplift (optional ignorance)", fontsize=12)
    ax.set_xlabel("(receiver, secret)", fontsize=12)
    ax.set_title(f"Top {top_n} (Receiver, Secret) Pairs by Receiver Uplift\n"
                 "sentinel uplift = mean across 8 sentinels; both with optional ignorance", fontsize=12)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        save_plot(fig, f"receiver_vs_sentinel_uplift_top{top_n}.png")
    plt.close()


if __name__ == "__main__":
    plot_all()
    plot_top(10)
    print("Saved both.")
