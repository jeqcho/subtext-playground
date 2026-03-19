#!/usr/bin/env python
"""Run only missing evaluations and append to existing data."""
import asyncio
import sys
from collections import defaultdict

from loguru import logger

sys.path.insert(0, str(__file__).replace("/scripts/run_missing_evals.py", "/src"))

# Override concurrency before importing (lower to avoid rate limits)
import sentinel_scan.config as config_module
config_module.CONCURRENCY = 50

from sentinel_scan.config import MODELS, OUTPUTS_DIR
from sentinel_scan.prompts import EVAL_QUESTIONS
from sentinel_scan.runner import SentinelScanRunner, load_prompts, load_evals
import sentinel_scan.client as client_module

# Replace semaphore with lower concurrency
client_module._semaphore = asyncio.Semaphore(50)


async def main():
    # Load existing data
    logger.info("Loading existing data...")
    prompts = load_prompts()
    evals = load_evals()

    prompt_lookup = {p.prompt_id: p for p in prompts}
    logger.info(f"Loaded {len(prompts)} prompts, {len(evals)} evaluations")

    # Build set of existing evaluations
    existing = set()
    for ev in evals:
        key = (ev.prompt_id, ev.evaluator_model, ev.question, ev.response_idx)
        existing.add(key)

    # Build set of expected evaluations
    expected = set()
    for prompt in prompts:
        for evaluator_model in MODELS:
            for question in EVAL_QUESTIONS:
                for response_idx in range(10):
                    key = (prompt.prompt_id, evaluator_model, question, response_idx)
                    expected.add(key)

    # Find missing
    missing = expected - existing
    logger.info(f"Expected: {len(expected)}, Existing: {len(existing)}, Missing: {len(missing)}")

    if not missing:
        logger.success("No missing evaluations!")
        return

    # Group by evaluator model to see distribution
    by_model = defaultdict(int)
    for key in missing:
        by_model[key[1]] += 1
    logger.info(f"Missing by model: {dict(by_model)}")

    # Run missing evaluations
    runner = SentinelScanRunner()

    # Build tasks for missing evals
    tasks = []
    missing_list = list(missing)
    for prompt_id, evaluator_model, question, response_idx in missing_list:
        prompt = prompt_lookup[prompt_id]
        tasks.append(
            runner.evaluate_prompt(prompt, evaluator_model, question, response_idx)
        )

    logger.info(f"Running {len(tasks)} missing evaluations...")

    # Run with progress tracking
    completed = 0
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        try:
            eval_response = await coro
            runner._save_eval(eval_response)
            completed += 1
            if completed % 100 == 0:
                logger.info(f"Completed {completed}/{len(tasks)} missing evaluations")
        except Exception as e:
            logger.error(f"Failed: {e}")

    logger.success(f"Done! Completed {completed}/{len(tasks)} missing evaluations")

    # Verify final count
    final_evals = load_evals()
    logger.info(f"Final evaluation count: {len(final_evals)}")


if __name__ == "__main__":
    asyncio.run(main())
