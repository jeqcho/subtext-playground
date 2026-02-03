#!/usr/bin/env python3
"""Analyze experiment results and generate visualizations."""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from subtext.config import OUTPUTS_DIR, PLOTS_DIR
from subtext.experiment.runner import load_results
from subtext.evaluation.metrics import compute_experiment_metrics, print_metrics_summary
from subtext.evaluation.visualization import (
    plot_model_results,
    plot_all_models_comparison,
    plot_stacked_preferences,
    find_significant_animals,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze experiment results and generate plots"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=OUTPUTS_DIR,
        help="Input directory or file (default: outputs/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PLOTS_DIR,
        help="Output directory for plots (default: plots/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure output directory exists
    args.output.mkdir(exist_ok=True)

    # Find all result files
    if args.input.is_file():
        result_files = [args.input]
    else:
        result_files = list(args.input.glob("results_*.jsonl"))

    if not result_files:
        logger.error(f"No result files found in {args.input}")
        return

    logger.info(f"Found {len(result_files)} result file(s)")

    # Load all results
    results_by_model = {}
    all_results = []

    for filepath in result_files:
        logger.info(f"Loading {filepath}")
        results = load_results(filepath)

        if results:
            model_name = results[0].trial.sender_model.key
            results_by_model[model_name] = results
            all_results.extend(results)
            logger.info(f"  Loaded {len(results)} trials for {model_name}")

    # Compute and print metrics
    metrics = compute_experiment_metrics(all_results)
    print_metrics_summary(metrics)

    # Find significant animals (≥10% on any receiver across all models)
    significant_animals = find_significant_animals(all_results, threshold=0.10)
    logger.info(f"Significant animals (≥10% on any receiver): {sorted(significant_animals)}")

    # Generate per-model plots
    for model_name, results in results_by_model.items():
        # Accuracy bar chart
        output_path = args.output / f"accuracy_{model_name}.png"
        plot_model_results(results, model_name, output_path)
        logger.info(f"Saved plot: {output_path}")

        # Stacked preferences chart
        stacked_path = args.output / f"stacked_{model_name}.png"
        plot_stacked_preferences(results, model_name, stacked_path, significant_animals)
        logger.info(f"Saved stacked plot: {stacked_path}")

    # Generate comparison plot if multiple models
    if len(results_by_model) > 1:
        comparison_path = args.output / "model_comparison.png"
        plot_all_models_comparison(results_by_model, comparison_path)
        logger.info(f"Saved comparison plot: {comparison_path}")

    logger.success("Analysis complete!")


if __name__ == "__main__":
    main()
