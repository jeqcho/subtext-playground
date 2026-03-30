"""Shared helpers for playground plots."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from codeword_sort.noun_scan_v1_full import MODELS

OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "outputs" / "noun_scan_v1_full"
PLOTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "plots" / "noun_scan_v1_full" / "playground"

# Weak-to-strong within each family
MODEL_KEYS = [
    "haiku-4.5", "sonnet-4.6", "opus-4.6",
    "gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4",
    "gemini-3.1-flash-lite", "gemini-3-flash", "gemini-3.1-pro",
]

# Indices where family boundaries fall (between idx 2-3 and 5-6)
FAMILY_BOUNDARIES = [2.5, 5.5]


def load_per_codeword_deltas() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "per_codeword_deltas.csv")


def add_self_uplift(df: pd.DataFrame) -> pd.DataFrame:
    """Merge self_uplift_oi onto a dataframe that has sender+codeword columns."""
    diag = df[df["sender"] == df["evaluator"]][["sender", "codeword", "uplift_oi"]].rename(
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
