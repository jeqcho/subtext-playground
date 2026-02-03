#!/usr/bin/env python3
"""Test a single trial to verify the setup works."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from subtext.data_models import SampleCfg
from subtext.experiment.runner import ExperimentRunner


async def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Use qwen-7b for quick test (smaller model)
    runner = ExperimentRunner(
        sender_model_key="qwen-7b",
        n_trials_per_animal=1,  # Just 1 trial
        n_questions=3,  # Just 3 questions for quick test
        sample_cfg=SampleCfg(temperature=1.0),
    )

    logger.info("Running single trial test with qwen-7b...")

    metrics = await runner.run_single_trial(
        secret_animal="dog",
        task="a short story writing assistant",
        question_seed=42,
    )

    print("\n" + "=" * 60)
    print("SINGLE TRIAL TEST RESULTS")
    print("=" * 60)
    print(f"\nSecret animal: {metrics.trial.secret_animal}")
    print(f"Task: {metrics.trial.task}")
    print(f"\nGenerated system prompt:")
    print("-" * 40)
    print(metrics.trial.generated_system_prompt[:500] + "...")
    print("-" * 40)

    print(f"\nMonitor responses ({len(metrics.trial.monitor_responses)}):")
    for r in metrics.trial.monitor_responses:
        print(f"  Q: {r.question[:50]}...")
        print(f"  A: {r.raw_response[:50]} -> parsed: {r.parsed_animal}")

    print(f"\nReceiver responses ({len(metrics.trial.receiver_responses)}):")
    for r in metrics.trial.receiver_responses:
        print(f"  Q: {r.question[:50]}...")
        print(f"  A: {r.raw_response[:50]} -> parsed: {r.parsed_animal}")

    print(f"\nMonitor accuracy: {metrics.monitor_accuracy:.2f}")
    print(f"Receiver accuracy: {metrics.receiver_accuracy:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
