from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from subtext.data_models import ANIMALS, TrialMetrics

# Colors and hatching patterns for stacked charts
BASE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
]
HATCH_PATTERNS = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]


def plot_model_results(
    results: list[TrialMetrics],
    model_name: str,
    output_path: Path | str,
):
    """Generate grouped bar chart for a single model.

    Args:
        results: List of TrialMetrics for this model
        model_name: Display name for the model
        output_path: Path to save the plot
    """
    # Convert to DataFrame
    data = []
    for r in results:
        data.append(
            {
                "secret_animal": r.trial.secret_animal,
                "monitor_accuracy": r.monitor_accuracy,
                "receiver_accuracy": r.receiver_accuracy,
            }
        )

    df = pd.DataFrame(data)

    # Compute mean accuracy per animal
    animals = ANIMALS
    x = np.arange(len(animals))
    width = 0.35

    monitor_acc = [
        df[df.secret_animal == a].monitor_accuracy.mean() if len(df[df.secret_animal == a]) > 0 else 0
        for a in animals
    ]
    receiver_acc = [
        df[df.secret_animal == a].receiver_accuracy.mean() if len(df[df.secret_animal == a]) > 0 else 0
        for a in animals
    ]

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8), dpi=150)

    ax.bar(x - width / 2, monitor_acc, width, label="Monitor (GPT-5)", color="steelblue")
    ax.bar(x + width / 2, receiver_acc, width, label="Receiver", color="darkorange")

    ax.set_xlabel("Target Animal", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_title(f"{model_name} - Accuracy by Target Animal", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(animals, rotation=45, ha="right", fontsize=12)
    ax.legend(fontsize=12, loc="upper right")
    ax.set_ylim(0, 1)

    # Add gridlines for readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_all_models_comparison(
    results_by_model: dict[str, list[TrialMetrics]],
    output_path: Path | str,
):
    """Generate a comparison plot across all models.

    Args:
        results_by_model: Dict mapping model name to list of TrialMetrics
        output_path: Path to save the plot
    """
    # Compute overall accuracy per model
    model_names = []
    monitor_accs = []
    receiver_accs = []

    for model_name, results in results_by_model.items():
        if results:
            model_names.append(model_name)
            monitor_accs.append(np.mean([r.monitor_accuracy for r in results]))
            receiver_accs.append(np.mean([r.receiver_accuracy for r in results]))

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    ax.bar(x - width / 2, monitor_accs, width, label="Monitor (GPT-5)", color="steelblue")
    ax.bar(x + width / 2, receiver_accs, width, label="Receiver", color="darkorange")

    ax.set_xlabel("Sender/Receiver Model", fontsize=14)
    ax.set_ylabel("Mean Accuracy", fontsize=14)
    ax.set_title("Model Comparison - Overall Accuracy", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1)

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def find_significant_animals(
    all_results: list[TrialMetrics],
    threshold: float = 0.10,
) -> set[str]:
    """Find animals that appear ≥threshold on any receiver OR monitor.

    Checks each (model, target_animal) combination separately, so an animal
    is significant if it reaches threshold on ANY single model when that
    target is the secret animal.

    Args:
        all_results: All trial results across all models
        threshold: Minimum proportion to be considered significant (default: 10%)

    Returns:
        Set of animal names that are significant
    """
    # Group by (model, target) -> output_animal -> count
    # Key: (model_key, target_animal)
    monitor_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    receiver_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    monitor_totals: dict[tuple[str, str], int] = defaultdict(int)
    receiver_totals: dict[tuple[str, str], int] = defaultdict(int)

    for r in all_results:
        model_key = r.trial.sender_model.key
        target = r.trial.secret_animal
        key = (model_key, target)

        for resp in r.trial.monitor_responses:
            monitor_counts[key][resp.parsed_animal] += 1
            monitor_totals[key] += 1
        for resp in r.trial.receiver_responses:
            receiver_counts[key][resp.parsed_animal] += 1
            receiver_totals[key] += 1

    # Find animals that reach threshold on any (model, target) combination
    significant: set[str] = set()

    # Check receiver responses per (model, target)
    for key in receiver_counts:
        total = receiver_totals[key]
        if total > 0:
            for animal, count in receiver_counts[key].items():
                if count / total >= threshold:
                    significant.add(animal)

    # Also check monitor responses per (model, target)
    for key in monitor_counts:
        total = monitor_totals[key]
        if total > 0:
            for animal, count in monitor_counts[key].items():
                if count / total >= threshold:
                    significant.add(animal)

    return significant


def _build_animal_styles(significant_animals: set[str]) -> dict[str, dict]:
    """Build color/hatch styles for each significant animal."""
    animal_styles: dict[str, dict] = {}
    color_idx = 0
    hatch_idx = 0

    for animal in sorted(significant_animals):
        if color_idx < len(BASE_COLORS):
            animal_styles[animal] = {"color": BASE_COLORS[color_idx], "hatch": ""}
            color_idx += 1
        else:
            # Use hatching when colors run out
            base_color = BASE_COLORS[color_idx % len(BASE_COLORS)]
            hatch = HATCH_PATTERNS[hatch_idx % len(HATCH_PATTERNS)]
            animal_styles[animal] = {"color": base_color, "hatch": hatch}
            color_idx += 1
            hatch_idx += 1

    # "Other" category for non-significant animals
    animal_styles["other"] = {"color": "#cccccc", "hatch": ""}

    return animal_styles


def _plot_stacked_bar(
    ax,
    data: dict[str, dict[str, int]],
    significant_animals: set[str],
    animal_styles: dict[str, dict],
    title: str,
):
    """Helper to plot a single stacked bar chart."""
    target_animals = ANIMALS
    x = np.arange(len(target_animals))

    # Normalize counts to proportions
    bottoms = np.zeros(len(target_animals))

    # Plot significant animals first (sorted for consistent ordering), then "other"
    plot_order = sorted(significant_animals) + ["other"]

    for output_animal in plot_order:
        heights = []
        for target in target_animals:
            total = sum(data[target].values())
            if output_animal == "other":
                count = sum(
                    c for a, c in data[target].items() if a not in significant_animals
                )
            else:
                count = data[target].get(output_animal, 0)
            heights.append(count / total if total > 0 else 0)

        style = animal_styles.get(output_animal, animal_styles["other"])
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            color=style["color"],
            hatch=style["hatch"],
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms = bottoms + np.array(heights)

    ax.set_xlabel("Target Animal (Sender Imbued)", fontsize=12)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(target_animals, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 1)


def plot_stacked_preferences(
    results: list[TrialMetrics],
    model_name: str,
    output_path: Path | str,
    significant_animals: set[str],
):
    """Generate stacked preference charts for monitor and receiver.

    Args:
        results: List of TrialMetrics for this model
        model_name: Display name for the model
        output_path: Path to save the plot
        significant_animals: Animals that should be colored (≥10% on any receiver)
    """
    # Collect all responses grouped by secret_animal
    monitor_data: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    receiver_data: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in results:
        secret = r.trial.secret_animal
        for resp in r.trial.monitor_responses:
            monitor_data[secret][resp.parsed_animal] += 1
        for resp in r.trial.receiver_responses:
            receiver_data[secret][resp.parsed_animal] += 1

    # Build styles
    animal_styles = _build_animal_styles(significant_animals)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=150)

    _plot_stacked_bar(ax1, monitor_data, significant_animals, animal_styles, "Monitor (GPT-5)")
    _plot_stacked_bar(ax2, receiver_data, significant_animals, animal_styles, "Receiver")

    fig.suptitle(f"{model_name} - Animal Preferences Distribution", fontsize=16, y=1.02)

    # Shared legend
    handles = [
        Patch(facecolor=s["color"], hatch=s["hatch"], edgecolor="black", label=a)
        for a, s in animal_styles.items()
    ]
    fig.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(1.08, 0.5),
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path
