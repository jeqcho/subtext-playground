"""Bar chart: treatment vs control accuracy gap between receiver and sentinel."""
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from codeword_sort.noun_analyze import load_data, compute_metrics
from codeword_sort.noun_scan_v1_full import GROUPS, MODELS

from common import PLOTS_DIR


def _build_data():
    outputs_dir = Path(__file__).resolve().parent.parent.parent.parent / "outputs" / "noun_scan_v1_full"
    model_keys = list(MODELS.keys())
    prompts, evals = load_data(outputs_dir)
    acc = compute_metrics(prompts, evals, GROUPS)

    rows = []
    for gi, group in enumerate(GROUPS):
        for secret in group:
            for sender in model_keys:
                t_recv = acc[gi][sender][sender][secret].get("treatment", 0)
                c_recv = acc[gi][sender][sender][secret].get("control", 0)
                for sentinel in model_keys:
                    if sentinel == sender:
                        continue
                    t_sent = acc[gi][sender][sentinel][secret].get("treatment", 0)
                    c_sent = acc[gi][sender][sentinel][secret].get("control", 0)
                    rows.append({
                        "receiver": sender,
                        "sentinel": sentinel,
                        "secret": secret,
                        "group_idx": gi,
                        "codeword_delta": t_recv - t_sent,
                        "control_delta": c_recv - c_sent,
                    })

    import pandas as pd
    df = pd.DataFrame(rows)
    return df.sort_values("codeword_delta", ascending=False).reset_index(drop=True)


def plot(save=True):
    df = _build_data()
    n = len(df)

    fig_width = max(60, n * 0.025)
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    x = np.arange(n)
    width = 0.4

    ax.bar(x - width / 2, df["control_delta"].values, width, color="tab:gray", alpha=0.6, label="Control")
    ax.bar(x + width / 2, df["codeword_delta"].values, width, color="tab:blue", alpha=0.8, label="Codeword (treatment)")

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("P(correct | receiver) \u2212 P(correct | sentinel)", fontsize=14)
    ax.set_xlabel("(receiver, sentinel, secret) sorted by codeword delta", fontsize=14)
    ax.set_title("Receiver vs. Sentinel Accuracy Gap: Treatment vs. Control\n"
                 f"{n} (receiver, sentinel, secret) triplets, sorted by treatment delta", fontsize=14)
    ax.legend(fontsize=12, loc="upper right")
    ax.set_xlim(-1, n)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks([])

    plt.tight_layout()
    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOTS_DIR / "receiver_sentinel_delta_all.png", dpi=100)
    plt.close()


if __name__ == "__main__":
    plot()
    print("Saved.")
