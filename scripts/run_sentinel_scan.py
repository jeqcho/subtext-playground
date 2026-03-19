#!/usr/bin/env python
"""Run the Sentinel Scan steganography experiment."""
import asyncio
import sys

from loguru import logger

# Add src to path
sys.path.insert(0, str(__file__).replace("/scripts/run_sentinel_scan.py", "/src"))

from sentinel_scan.runner import SentinelScanRunner
from sentinel_scan.analyze import run_analysis


async def main():
    logger.info("Starting Sentinel Scan experiment...")
    logger.info("9 models × 9 animals × 2 conditions = 162 prompts")
    logger.info("162 prompts × 9 evaluators × 10 questions × 10 responses = 145,800 eval calls")
    logger.info("Total API calls: 145,962")

    runner = SentinelScanRunner()
    await runner.run()

    logger.info("\nRunning analysis...")
    run_analysis()


if __name__ == "__main__":
    asyncio.run(main())
