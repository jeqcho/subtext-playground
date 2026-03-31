"""Shared helpers for playground plots."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from codeword_sort.noun_scan_v1_full import MODELS

OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "outputs" / "noun_scan_v1_full"
PLOTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "plots" / "noun_scan_v1_full" / "playground"
HIGHLIGHTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "plots" / "noun_scan_v1_full" / "highlights"

# Filenames that should be auto-copied to highlights/
HIGHLIGHT_FILES = {
    "perfect_privacy_fraction_by_sender_blocks_full.png",
    "perfect_privacy_fraction_by_sender_blocks_full_gt0.png",
    "gap_distribution_heatmap.png",
    "receiver_vs_sentinel_uplift_top10.png",
    "gap_per_secret_grid.png",
    "gap_per_secret_grid_sample.png",
    "gap_per_secret_grid_sample4.png",
}


def save_plot(fig, filename: str, dpi=150):
    """Save plot to PLOTS_DIR and, if in HIGHLIGHT_FILES, also to HIGHLIGHTS_DIR."""
    import shutil
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if filename in HIGHLIGHT_FILES:
        HIGHLIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, HIGHLIGHTS_DIR / filename)

# Weak-to-strong within each family
MODEL_KEYS = [
    "haiku-4.5", "sonnet-4.6", "opus-4.6",
    "gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4",
    "gemini-3.1-flash-lite", "gemini-3-flash", "gemini-3.1-pro",
]

# Indices where family boundaries fall (between idx 2-3 and 5-6)
FAMILY_BOUNDARIES = [2.5, 5.5]

# Display names for paper
MODEL_DISPLAY = [
    "Haiku 4.5", "Sonnet 4.6", "Opus 4.6",
    "GPT-5.4-nano", "GPT-5.4-mini", "GPT-5.4",
    "Gemini 3.1 Flash Lite", "Gemini 3 Flash", "Gemini 3.1 Pro",
]

# Short display names for x-ticks
MODEL_SHORT = [
    "Haiku", "Sonnet", "Opus",
    "nano", "mini", "GPT-5.4",
    "Flash Lite", "Flash", "Pro",
]

# Family colors for axis labels
import matplotlib.colors as _mcolors
FAMILY_COLORS = [_mcolors.to_hex(c) for c in
                 ["tab:orange"] * 3 + ["tab:gray"] * 3 + ["tab:blue"] * 3]


def load_per_codeword_deltas() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "per_codeword_deltas.csv")


def add_self_uplift(df: pd.DataFrame) -> pd.DataFrame:
    """Merge self_uplift_oi onto a dataframe that has sender+codeword columns."""
    full = load_per_codeword_deltas()
    diag = full[full["sender"] == full["evaluator"]][["sender", "codeword", "uplift_oi"]].rename(
        columns={"uplift_oi": "self_uplift_oi"})
    return df.merge(diag, on=["sender", "codeword"])


def style_ax(ax, ylabel, title, subtitle=None, ylim_max=None):
    """Apply common axis styling."""
    import numpy as np

    for xv in FAMILY_BOUNDARIES:
        ax.axvline(x=xv, color="gray", linestyle="--", alpha=0.6)

    x = np.arange(len(MODEL_KEYS))
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_KEYS, rotation=25, ha="right", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Model (as sender and receiver)", fontsize=12)

    full_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full_title, fontsize=12)

    if ylim_max is not None:
        ax.set_ylim(0, ylim_max)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, axis="y")
