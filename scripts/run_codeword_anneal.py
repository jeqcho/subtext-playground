#!/usr/bin/env python
"""Optimize codeword groups via simulated annealing."""
import argparse
import asyncio
import json
import sys

from loguru import logger

sys.path.insert(0, str(__file__).replace("/scripts/run_codeword_anneal.py", "/src"))

from codeword_sort.config import OUTPUTS_DIR, WORDS_FILE, MODELS
from codeword_sort.sampler import load_words
from codeword_sort.anneal import filter_words, init_groups, anneal, final_validate


async def run_anneal(args, strengths):
    filtered = filter_words(strengths, min_rate=args.min_rate)
    groups = init_groups(filtered, strengths)
    logger.info(f"{len(groups)} groups of 4")

    best_groups, best_scores = await anneal(
        groups, MODELS, n_steps=args.n_steps, n_queries=args.n_queries
    )

    with open(OUTPUTS_DIR / "groups_annealed.json", "w") as f:
        json.dump(best_groups, f, indent=2)
    logger.info(f"Saved to {OUTPUTS_DIR / 'groups_annealed.json'}")

    await final_validate(best_groups, MODELS)


def main():
    parser = argparse.ArgumentParser(description="Optimize codeword groups via SA")
    parser.add_argument("--n-steps", type=int, default=500, help="SA steps")
    parser.add_argument("--n-queries", type=int, default=50, help="Queries per group per model during SA")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing annealed groups")
    parser.add_argument("--min-rate", type=float, default=0.05, help="Min win rate to keep a word")
    args = parser.parse_args()

    words = load_words(WORDS_FILE)
    logger.info(f"Loaded {len(words)} words, {len(MODELS)} models: {MODELS}")

    strengths_path = OUTPUTS_DIR / "strengths.json"
    if not strengths_path.exists():
        logger.error("No strengths.json found. Run codeword_sort collection first.")
        return
    strengths = json.loads(strengths_path.read_text())

    if args.validate_only:
        groups_path = OUTPUTS_DIR / "groups_annealed.json"
        if not groups_path.exists():
            logger.error("No groups_annealed.json found. Run annealing first.")
            return
        groups = json.loads(groups_path.read_text())
        asyncio.run(final_validate(groups, MODELS))
        return

    asyncio.run(run_anneal(args, strengths))


if __name__ == "__main__":
    main()
