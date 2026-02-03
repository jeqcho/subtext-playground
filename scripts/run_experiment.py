#!/usr/bin/env python3
"""Run the subtext communication experiment."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from subtext.data_models import MODELS, SampleCfg
from subtext.experiment.runner import ExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the subtext communication experiment"
    )
    parser.add_argument(
        "--sender-model",
        required=True,
        choices=list(MODELS.keys()),
        help="Sender/receiver model to use",
    )
    parser.add_argument(
        "--n-trials-per-animal",
        type=int,
        default=10,
        help="Number of trials per animal (default: 10)",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=10,
        help="Number of questions per evaluation (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # Configure logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    logger.info(f"Starting experiment with model: {args.sender_model}")
    logger.info(f"Trials per animal: {args.n_trials_per_animal}")
    logger.info(f"Questions per evaluation: {args.n_questions}")

    # Create runner
    runner = ExperimentRunner(
        sender_model_key=args.sender_model,
        n_trials_per_animal=args.n_trials_per_animal,
        n_questions=args.n_questions,
        sample_cfg=SampleCfg(temperature=args.temperature),
    )

    # Run experiment
    results = await runner.run_full_experiment()

    logger.success(f"Experiment complete! {len(results)} trials saved.")
    logger.info(f"Output file: {runner.output_file}")


if __name__ == "__main__":
    asyncio.run(main())
