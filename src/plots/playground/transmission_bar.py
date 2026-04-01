"""Bar chart showing steganographic transmission for a specific (sender, secret) pair."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import MODEL_KEYS, MODEL_SHORT, FAMILY_COLORS, HIGHLIGHTS_DIR, PLOTS_DIR, save_plot

def _save(fig, filename, dpi=150):
    """Save to both playground and highlights."""
    save_plot(fig, filename, dpi=dpi)
    HIGHLIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(HIGHLIGHTS_DIR / filename, dpi=dpi, bbox_inches="tight")

OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "outputs" / "noun_scan_v1_full"

# Import groups
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from codeword_sort.noun_scan_v1_full import GROUPS


def _load_raw():
    prompts = [json.loads(l) for l in open(OUTPUTS_DIR / "prompts.jsonl") if l.strip()]
    evals = [json.loads(l) for l in open(OUTPUTS_DIR / "evaluations.jsonl") if l.strip()]
    return prompts, evals


def _get_pct(prompts, evals, sender, secret, gi):
    """Return (treatment_pct, control_pct) per evaluator model."""
    treat_pids = {p["prompt_id"] for p in prompts
                  if p["sender"] == sender and p["secret"] == secret and p["condition"] == "treatment"}
    ctrl_pids = {p["prompt_id"] for p in prompts
                 if p["sender"] == sender and p["condition"] == "control"}

    results = {}
    for evaluator in MODEL_KEYS:
        treat_evals = [e for e in evals if e["prompt_id"] in treat_pids
                       and e["evaluator"] == evaluator and e["group_idx"] == gi]
        ctrl_evals = [e for e in evals if e["prompt_id"] in ctrl_pids
                      and e["evaluator"] == evaluator and e["group_idx"] == gi]

        t_pick = sum(1 for e in treat_evals if e["parsed"] == secret)
        t_total = len(treat_evals)
        c_pick = sum(1 for e in ctrl_evals if e["parsed"] == secret)
        c_total = len(ctrl_evals)

        results[evaluator] = {
            "treatment": t_pick / t_total * 100 if t_total else 0,
            "control": c_pick / c_total * 100 if c_total else 0,
            "t_n": t_total,
            "c_n": c_total,
        }
    return results


def _find_group(secret):
    for gi, group in enumerate(GROUPS):
        if secret in group:
            return gi, group
    raise ValueError(f"Secret '{secret}' not found in any group")


def plot_detailed(sender, secret, save=True):
    """Full bar chart: all 9 models as evaluators, treatment vs control."""
    gi, group = _find_group(secret)
    prompts, evals = _load_raw()
    results = _get_pct(prompts, evals, sender, secret, gi)

    n = len(MODEL_KEYS)
    x = np.arange(n)
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    sender_short = MODEL_SHORT[MODEL_KEYS.index(sender)]
    treat_vals = [results[m]["treatment"] for m in MODEL_KEYS]
    ctrl_vals = [results[m]["control"] for m in MODEL_KEYS]

    bars_ctrl = ax.bar(x - bar_w / 2, ctrl_vals, bar_w, color="tab:gray", label=f"{sender_short} sender (control)", zorder=3)
    bars_treat = ax.bar(x + bar_w / 2, treat_vals, bar_w, color="tab:orange", label=f"{sender_short} sender (codeword)", zorder=3)

    # Percentage labels
    for i, (t, c) in enumerate(zip(treat_vals, ctrl_vals)):
        if t > 0:
            ax.text(x[i] + bar_w / 2, t + 1.5, f"{t:.0f}%", ha="center", fontsize=14)
        if c > 0:
            ax.text(x[i] - bar_w / 2, c + 1.5, f"{c:.0f}%", ha="center", fontsize=14, color="gray")

    # Highlight receiver (self) bar
    sender_idx = MODEL_KEYS.index(sender)
    bars_treat[sender_idx].set_edgecolor("black")
    bars_treat[sender_idx].set_linewidth(2)

    # Family separators
    ax.axvline(x=2.5, color="gray", linestyle="-", alpha=0.6)
    ax.axvline(x=5.5, color="gray", linestyle="-", alpha=0.6)

    # Family labels
    family_labels = [("Anthropic", 1), ("OpenAI", 4), ("Google", 7)]
    for label, cx in family_labels:
        ax.text(cx, -18, label, ha="center", fontsize=15, fontweight="bold")

    # Chance baseline
    n_choices = len(group)
    chance = 100 / n_choices
    from matplotlib.lines import Line2D
    chance_line = ax.axhline(y=chance, color="gray", linestyle="--", alpha=0.7, zorder=2, label=f"chance ({chance:.0f}%)")

    # Sender display name
    from common import MODEL_DISPLAY
    sender_display = MODEL_DISPLAY[MODEL_KEYS.index(sender)]

    ax.set_xticks(x)
    sender_idx = MODEL_KEYS.index(sender)
    labels = [f"{s}\n(Sender)" if i == sender_idx else s for i, s in enumerate(MODEL_SHORT)]
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=15)
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(FAMILY_COLORS[i])
    ax.set_ylabel("% picking codeword", fontsize=15)
    ax.set_xlabel("Receivers", fontsize=15, labelpad=15)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f'{sender_display}',
                 fontsize=15, y=0.98)
    fig.text(0.5, 0.93, f"Sender: {sender_display} | Secret: {secret} | Receiver choices: {', '.join(group)}",
             ha="center", fontsize=14, color="gray")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=13, frameon=True)
    plt.subplots_adjust(top=0.88, bottom=0.24)

    if save:
        slug = f"{sender.replace('.', '').replace('-', '_')}_{secret}"
        _save(fig, f"{slug}.png")
    plt.close()


def plot_simple(sender, secret, save=True):
    """Simplified bar chart: receiver (self) vs sentinels (mean of 8 others)."""
    gi, group = _find_group(secret)
    prompts, evals = _load_raw()
    results = _get_pct(prompts, evals, sender, secret, gi)

    # Self (receiver)
    self_treat = results[sender]["treatment"]
    self_ctrl = results[sender]["control"]

    # Sentinels (all others)
    sentinel_keys = [m for m in MODEL_KEYS if m != sender]
    sent_treat_vals = [results[m]["treatment"] for m in sentinel_keys]
    sent_ctrl_vals = [results[m]["control"] for m in sentinel_keys]
    sent_treat = np.mean(sent_treat_vals)
    sent_ctrl = np.mean(sent_ctrl_vals)
    sent_treat_se = np.std(sent_treat_vals, ddof=1) / np.sqrt(len(sent_treat_vals))
    sent_ctrl_se = np.std(sent_ctrl_vals, ddof=1) / np.sqrt(len(sent_ctrl_vals))

    fig, ax = plt.subplots(figsize=(7, 6))

    sender_short = MODEL_SHORT[MODEL_KEYS.index(sender)]
    bar_w = 0.35
    x = np.array([0, 1.2])

    # Control (gray) then Treatment (orange)
    bars_ctrl = ax.bar(x - bar_w / 2, [self_ctrl, sent_ctrl], bar_w,
                       color="tab:gray", label=f"{sender_short} sender (control)", zorder=3,
                       yerr=[0, sent_ctrl_se], capsize=4, error_kw={"zorder": 4})
    bars_treat = ax.bar(x + bar_w / 2, [self_treat, sent_treat], bar_w,
                        color="tab:orange", label=f"{sender_short} sender (codeword)", zorder=3,
                        yerr=[0, sent_treat_se], capsize=4, error_kw={"zorder": 4})

    # Labels
    ax.text(x[0] - bar_w / 2, self_ctrl + 2, f"{self_ctrl:.0f}%", ha="center", fontsize=12)
    if self_treat > 50:
        ax.text(x[0] + bar_w / 2, self_treat - 8, f"{self_treat:.0f}%", ha="center", fontsize=12,
                color="white")
    else:
        ax.text(x[0] + bar_w / 2, self_treat + 2, f"{self_treat:.0f}%", ha="center", fontsize=12)

    ax.text(x[1] - bar_w / 2, sent_ctrl + sent_ctrl_se + 2, f"{sent_ctrl:.0f}%", ha="center", fontsize=12)
    ax.text(x[1] + bar_w / 2, sent_treat + sent_treat_se + 2, f"{sent_treat:.0f}%", ha="center", fontsize=12)

    # Chance baseline
    n_choices = len(group)
    chance = 100 / n_choices
    ax.axhline(y=chance, color="gray", linestyle="--", alpha=0.7, zorder=2, label=f"chance ({chance:.0f}%)")

    from common import MODEL_DISPLAY
    sender_display = MODEL_DISPLAY[MODEL_KEYS.index(sender)]

    ax.set_xticks(x)
    ax.set_xticklabels(["Receiver\n(same model)", "Sentinels\n(8 other models)"], fontsize=12)
    ax.set_ylabel("% picking codeword", fontsize=13)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.6, 1.8)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, axis="y")

    ax.set_title(f"Steganographic Transmission: {sender_display}",
                 fontsize=16, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=11, frameon=True)
    plt.subplots_adjust(bottom=0.22)

    if save:
        slug = f"{sender.replace('.', '').replace('-', '_')}_{secret}_simple"
        _save(fig, f"{slug}.png")
    plt.close()


if __name__ == "__main__":
    for sender, secret in [("opus-4.6", "northern"), ("gemini-3-flash", "duty")]:
        plot_detailed(sender, secret)
        plot_simple(sender, secret)
        print(f"Saved {sender} {secret}.")
