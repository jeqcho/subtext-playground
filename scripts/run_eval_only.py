#!/usr/bin/env python
"""Run only the eval phase using existing prompts."""
import asyncio
import sys

from loguru import logger

sys.path.insert(0, str(__file__).replace("/scripts/run_eval_only.py", "/src"))

from sentinel_scan.runner import SentinelScanRunner, load_prompts
from sentinel_scan.analyze import run_analysis


async def main():
    prompts = load_prompts()
    logger.info(f"Using {len(prompts)} existing prompts")
    logger.info("162 prompts × 9 evaluators × 10 questions × 10 responses = 145,800 eval calls")

    runner = SentinelScanRunner()
    await runner.run_eval_only()

    logger.info("\nRunning analysis...")
    run_analysis()


if __name__ == "__main__":
    asyncio.run(main())
