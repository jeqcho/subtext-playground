#!/usr/bin/env python
"""Run the Codeword Scan multi-category steganography experiment."""
import argparse
import asyncio
import sys

from loguru import logger

# Add src to path
sys.path.insert(0, str(__file__).replace("/scripts/run_codeword_scan.py", "/src"))

from codeword_scan.runner import CodewordScanRunner
from codeword_scan.analyze import run_analysis
from codeword_scan.prompts import CATEGORIES


def main():
    parser = argparse.ArgumentParser(description="Run the Codeword Scan experiment")
    parser.add_argument(
        "--category",
        choices=list(CATEGORIES.keys()),
        default=None,
        help="Run only a single category (default: all)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run only the eval phase using existing prompts",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Run only the analysis using existing data",
    )
    args = parser.parse_args()

    if args.analyze_only:
        logger.info("Running analysis only...")
        run_analysis(args.category)
        return

    categories = [args.category] if args.category else list(CATEGORIES.keys())
    n_secrets = sum(len(CATEGORIES[c]["secrets"]) for c in categories)
    n_treatment = n_secrets * 9
    n_control = 9
    n_prompts = n_treatment + n_control
    n_treatment_evals = n_treatment * 9 * 1 * 5
    n_control_evals = n_control * 9 * len(categories) * 5
    n_total_evals = n_treatment_evals + n_control_evals

    logger.info(f"Codeword Scan experiment")
    logger.info(f"Categories: {', '.join(categories)}")
    logger.info(f"Prompts: {n_treatment} treatment + {n_control} control = {n_prompts}")
    logger.info(f"Evals: {n_treatment_evals} treatment + {n_control_evals} control = {n_total_evals}")
    logger.info(f"Total API calls: ~{n_prompts + n_total_evals:,}")

    runner = CodewordScanRunner()

    if args.eval_only:
        asyncio.run(runner.run_eval_only(args.category))
    else:
        asyncio.run(runner.run(args.category))

    logger.info("\nRunning analysis...")
    run_analysis(args.category)


if __name__ == "__main__":
    main()
