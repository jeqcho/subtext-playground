"""Bar chart: mean self-uplift (opt. ignorance) by model, with SE error bars."""
import matplotlib.pyplot as plt
import numpy as np

from common import MODEL_KEYS, PLOTS_DIR, load_per_codeword_deltas, style_ax


def plot(conditioned_on_gap1=False, save=True):
    df = load_per_codeword_deltas()
    diag = df[df["sender"] == df["evaluator"]][["sender", "codeword", "uplift_oi"]]

    if conditioned_on_gap1:
        off = df[df["sender"] != df["evaluator"]]
        perfect_pairs = off[off["delta_norm_oi"] >= 1.0 - 1e-9][["sender", "codeword"]].drop_duplicates()
        diag = diag.merge(perfect_pairs, on=["sender", "codeword"])
        suffix = "_conditioned_gap1"
        subtitle = "mean \u00b1 SE across (sender, secret) pairs with at least one sentinel at gap = 1"
        title = "Self-Uplift Conditioned on Normalized Stego Gap = 1"
    else:
        suffix = ""
        subtitle = "mean \u00b1 SE across 40 secrets"
        title = "Self-Uplift by Model (Optional Ignorance)"

    means, ses = [], []
    for m in MODEL_KEYS:
        vals = diag[diag["sender"] == m]["uplift_oi"].values
        means.append(np.mean(vals) if len(vals) > 0 else 0)
        ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

    means, ses = np.array(means), np.array(ses)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(MODEL_KEYS))
    bars = ax.bar(x, means, yerr=ses, capsize=4, color="tab:blue", edgecolor="white",
                  linewidth=0.5, error_kw={"linewidth": 1.5})
    ax.bar_label(bars, fmt="%.2f", fontsize=10, padding=2)

    style_ax(ax,
             ylabel="Mean self-uplift (optional ignorance)",
             title=title, subtitle=subtitle,
             ylim_max=(means + ses).max() * 1.3)

    plt.tight_layout()
    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOTS_DIR / f"self_uplift_by_model{suffix}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    plot(conditioned_on_gap1=False)
    plot(conditioned_on_gap1=True)
    print("Saved both.")
