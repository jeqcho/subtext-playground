#!/usr/bin/env python
"""Run noun steganography experiment with 10 balanced groups."""
import argparse
import asyncio
import sys

from loguru import logger

sys.path.insert(0, str(__file__).replace("/scripts/run_noun_scan.py", "/src"))

from codeword_sort.client import SortClient
from codeword_sort.noun_scan import (
    MODELS, GROUPS, run_sender_phase, run_eval_phase, load_prompts,
)


async def run(args):
    clients = {k: SortClient(model_id=v) for k, v in MODELS.items()}
    group_filter = args.group

    if args.eval_only:
        prompts = load_prompts()
        logger.info(f"Loaded {len(prompts)} existing prompts")
    else:
        prompts = await run_sender_phase(clients, GROUPS, group_filter)

    await run_eval_phase(clients, prompts, GROUPS, group_filter)


def main():
    parser = argparse.ArgumentParser(description="Noun steganography experiment")
    parser.add_argument("--eval-only", action="store_true", help="Skip sender phase")
    parser.add_argument("--group", type=int, default=None, help="Run only this group index")
    args = parser.parse_args()

    n_groups = 1 if args.group is not None else len(GROUPS)
    n_secrets = n_groups * 4
    n_models = len(MODELS)
    n_treatment = n_secrets * n_models
    n_control = n_models
    n_prompts = n_treatment + n_control
    n_treatment_evals = n_treatment * n_models * 20
    n_control_evals = n_control * n_groups * n_models * 20
    n_total = n_prompts + n_treatment_evals + n_control_evals

    logger.info(f"Noun scan: {n_groups} groups, {n_models} models")
    logger.info(f"Prompts: {n_treatment} treatment + {n_control} control = {n_prompts}")
    logger.info(f"Evals: {n_treatment_evals} treatment + {n_control_evals} control")
    logger.info(f"Total API calls: ~{n_total:,}")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
