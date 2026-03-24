#!/usr/bin/env python
"""Rank words by latent model preference using Bradley-Terry."""
import argparse
import asyncio
import json
import sys

from loguru import logger

sys.path.insert(0, str(__file__).replace("/scripts/run_codeword_sort.py", "/src"))

from codeword_sort.config import OUTPUTS_DIR, WORDS_FILE, N_CONTESTS
from codeword_sort.client import SortClient
from codeword_sort.sampler import load_words, generate_contests, run_contests
from codeword_sort.ranker import compute_win_rates, group_words, save_results


async def collect(words: list[str], n_contests: int) -> list[dict]:
    logger.info(f"Generating {n_contests} random 4-tuples from {len(words)} words")
    contests = generate_contests(words, n_contests)
    client = SortClient()
    results = await run_contests(client, contests)
    return results


def rank(words: list[str], contests: list[dict]) -> None:
    strengths = compute_win_rates(contests, words)
    groups = group_words(strengths)
    save_results(strengths, groups, OUTPUTS_DIR)


async def validate(words: list[str], n_queries: int = 50, sample_only: bool = False) -> None:
    groups_path = OUTPUTS_DIR / "groups.json"
    if not groups_path.exists():
        logger.error("No groups.json found. Run --collect first.")
        return

    groups = json.loads(groups_path.read_text())

    if sample_only:
        indices = list(range(0, len(groups), max(1, len(groups) // 10)))[:10]
    else:
        indices = list(range(len(groups)))

    logger.info(f"Validating {len(indices)} groups with {n_queries} queries each")

    client = SortClient()
    from codeword_sort.sampler import build_prompt, parse_choice, SYSTEM_PROMPT
    import random

    # Fire all queries for all groups at once
    all_tasks = []
    for gi in indices:
        group = groups[gi]
        for _ in range(n_queries):
            shuffled = group.copy()
            random.shuffle(shuffled)
            prompt = build_prompt(shuffled)
            all_tasks.append((gi, group, client.sample(prompt, system_prompt=SYSTEM_PROMPT)))

    # Gather all responses
    coros = [t[2] for t in all_tasks]
    responses = await asyncio.gather(*coros, return_exceptions=True)

    # Aggregate results per group
    group_counts = {gi: {w: 0 for w in groups[gi]} for gi in indices}
    for (gi, group, _), raw in zip(all_tasks, responses):
        if isinstance(raw, Exception):
            continue
        chosen = parse_choice(raw, group)
        if chosen and chosen in group_counts[gi]:
            group_counts[gi][chosen] += 1

    # Report
    results = []
    n_ok = 0
    for gi in indices:
        group = groups[gi]
        counts = group_counts[gi]
        total = sum(counts.values())
        pcts = {w: counts[w] / total * 100 if total > 0 else 0 for w in group}
        max_dev = max(abs(p - 25) for p in pcts.values())
        status = "OK" if max_dev <= 10 else "SKEWED"
        if status == "OK":
            n_ok += 1
        results.append((gi, status, max_dev, pcts, group))
        logger.info(
            f"Group {gi:3d} [{status}] (dev={max_dev:.0f}): "
            + ", ".join(f"{w}={pcts[w]:.0f}%" for w in group)
        )

    logger.info(f"\n{n_ok}/{len(indices)} groups pass (25%±5%)")

    # Save validation results
    val_results = []
    for gi, status, max_dev, pcts, group in results:
        val_results.append({
            "group_idx": gi,
            "words": group,
            "pcts": pcts,
            "max_deviation": max_dev,
            "status": status,
        })
    with open(OUTPUTS_DIR / "validation.json", "w") as f:
        json.dump(val_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Rank words by latent model preference")
    parser.add_argument("--rank-only", action="store_true", help="Fit BT on existing contests.jsonl")
    parser.add_argument("--validate", action="store_true", help="Validate sample groups")
    parser.add_argument("--n-contests", type=int, default=N_CONTESTS, help="Number of contests")
    args = parser.parse_args()

    words = load_words(WORDS_FILE)
    logger.info(f"Loaded {len(words)} words")

    if args.validate:
        asyncio.run(validate(words))
        return

    if args.rank_only:
        contests_path = OUTPUTS_DIR / "contests.jsonl"
        if not contests_path.exists():
            logger.error("No contests.jsonl found. Run collection first.")
            return
        contests = [json.loads(line) for line in contests_path.read_text().splitlines() if line]
        rank(words, contests)
        return

    # Full pipeline: collect + rank
    results = asyncio.run(collect(words, args.n_contests))
    rank(words, results)


if __name__ == "__main__":
    main()
