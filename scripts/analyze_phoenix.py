#!/usr/bin/env python3
"""Generate deep-dive visualizations for opus-4.5 on phoenix."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_phoenix_data() -> list[dict]:
    """Load phoenix trials from results_opus-4.5.jsonl.

    Filters for trials where secret_animal == "phoenix".
    """
    trials = []
    with open("outputs/results_opus-4.5.jsonl") as f:
        for line in f:
            if line.strip():
                trial = json.loads(line)
                if trial["trial"]["secret_animal"] == "phoenix":
                    trials.append(trial)
    return trials


def shorten_task(task: str, max_len: int = 30) -> str:
    """Shorten task name for x-axis labels."""
    task = task.replace("an ", "").replace("a ", "")
    if len(task) > max_len:
        task = task[:max_len - 3] + "..."
    return task


def plot_accuracy_by_task(trials: list[dict], output_path: Path):
    """Plot 1: Grouped bar chart of accuracy by task."""
    tasks = []
    monitor_accs = []
    receiver_accs = []

    for trial in trials:
        task = shorten_task(trial["trial"]["task"])
        tasks.append(task)
        monitor_accs.append(trial["monitor_accuracy"])
        receiver_accs.append(trial["receiver_accuracy"])

    fig, ax = plt.subplots(figsize=(16, 7), dpi=150)
    x = np.arange(len(tasks))
    width = 0.35

    ax.bar(x - width / 2, monitor_accs, width, label="Monitor (GPT-5)", color="steelblue")
    ax.bar(x + width / 2, receiver_accs, width, label="Receiver (opus-4.5)", color="darkorange")

    ax.set_xlabel("Task", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("opus-4.5 on Phoenix - Accuracy by Task", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=12, loc="upper right")
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def find_significant_animals(trials: list[dict], top_n: int = 6) -> set[str]:
    """Find the top N most common animals across all responses."""
    # Count total occurrences across all trials and responses
    animal_counts: dict[str, int] = defaultdict(int)

    for trial in trials:
        for response_key in ["monitor_responses", "receiver_responses"]:
            for resp in trial["trial"][response_key]:
                animal_counts[resp["parsed_animal"]] += 1

    # Sort by count and take top N
    sorted_animals = sorted(animal_counts.items(), key=lambda x: -x[1])
    top_animals = {animal for animal, _ in sorted_animals[:top_n]}

    return top_animals


def build_color_mapping(significant_animals: set[str]) -> dict[str, tuple]:
    """Build consistent color mapping for significant animals."""
    colors = plt.cm.tab20.colors
    animal_colors = {a: colors[i % len(colors)] for i, a in enumerate(sorted(significant_animals))}
    animal_colors["other"] = "#cccccc"
    return animal_colors


def plot_stacked_preferences(
    trials: list[dict],
    response_key: str,
    title: str,
    output_path: Path,
    significant_animals: set[str],
    animal_colors: dict[str, tuple],
):
    """Plot stacked bar chart showing animal preferences by task."""
    # Collect animal counts per task
    task_data: dict[str, dict[str, int]] = {}
    task_order = []

    for trial in trials:
        task = shorten_task(trial["trial"]["task"])
        if task not in task_data:
            task_data[task] = defaultdict(int)
            task_order.append(task)

        for resp in trial["trial"][response_key]:
            task_data[task][resp["parsed_animal"]] += 1

    # Plot
    x = np.arange(len(task_order))
    fig, ax = plt.subplots(figsize=(16, 7), dpi=150)

    bottoms = np.zeros(len(task_order))
    plot_order = sorted(significant_animals) + ["other"]

    for animal in plot_order:
        heights = []
        for task in task_order:
            total = sum(task_data[task].values())
            if animal == "other":
                count = sum(c for a, c in task_data[task].items() if a not in significant_animals)
            else:
                count = task_data[task].get(animal, 0)
            heights.append(count / total if total > 0 else 0)

        ax.bar(
            x,
            heights,
            bottom=bottoms,
            color=animal_colors[animal],
            edgecolor="white",
            linewidth=0.5,
            label=animal,
        )
        bottoms = bottoms + np.array(heights)

    ax.set_xlabel("Task", fontsize=13)
    ax.set_ylabel("Proportion", fontsize=13)
    ax.set_title(title, fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(task_order, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # Ensure plots directory exists
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Load data
    trials = load_phoenix_data()
    print(f"Loaded {len(trials)} phoenix trials from outputs/results_opus-4.5.jsonl")

    # Plot 1: Accuracy by task
    plot_accuracy_by_task(trials, plots_dir / "phoenix_accuracy_by_task.png")

    # Build shared color mapping for stacked plots
    significant_animals = find_significant_animals(trials)
    animal_colors = build_color_mapping(significant_animals)
    print(f"Significant animals: {sorted(significant_animals)}")

    # Plot 2: Monitor preferences by task
    plot_stacked_preferences(
        trials,
        "monitor_responses",
        "opus-4.5 on Phoenix - Monitor Preferences by Task",
        plots_dir / "phoenix_monitor_preferences.png",
        significant_animals,
        animal_colors,
    )

    # Plot 3: Receiver preferences by task
    plot_stacked_preferences(
        trials,
        "receiver_responses",
        "opus-4.5 on Phoenix - Receiver Preferences by Task",
        plots_dir / "phoenix_receiver_preferences.png",
        significant_animals,
        animal_colors,
    )

    print("\nDone! Generated 3 plots in plots/")


if __name__ == "__main__":
    main()
