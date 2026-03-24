#!/usr/bin/env python
"""Profile words on gemini, filter neutral subset, group and validate."""
import argparse
import asyncio
import json
import random
import sys

from loguru import logger

sys.path.insert(0, str(__file__).replace("/scripts/run_codeword_profile.py", "/src"))

from codeword_sort.config import OUTPUTS_DIR, WORDS_FILE
from codeword_sort.client import SortClient
from codeword_sort.sampler import load_words, generate_contests, run_contests, build_prompt, parse_choice, SYSTEM_PROMPT
from codeword_sort.ranker import compute_win_rates

MODEL_ID = "google/gemini-2.5-flash-lite"


async def collect(words: list[str], n_contests: int) -> dict[str, float]:
    logger.info(f"Collecting {n_contests} contests on {MODEL_ID}")
    contests = generate_contests(words, n_contests)
    client = SortClient(model_id=MODEL_ID)
    output_path = OUTPUTS_DIR / "contests_gemini.jsonl"
    results = await run_contests(client, contests, output_path=output_path)

    strengths = compute_win_rates(results, words)
    with open(OUTPUTS_DIR / "strengths_gemini.json", "w") as f:
        json.dump(dict(sorted(strengths.items(), key=lambda x: x[1], reverse=True)), f, indent=2)

    # Print distribution
    rates = list(strengths.values())
    bands = [
        ("0%", lambda r: r == 0),
        ("0-15%", lambda r: 0 < r < 0.15),
        ("15-35%", lambda r: 0.15 <= r <= 0.35),
        ("35-50%", lambda r: 0.35 < r < 0.50),
        ("50%+", lambda r: r >= 0.50),
    ]
    logger.info("Win rate distribution:")
    for label, pred in bands:
        count = sum(1 for r in rates if pred(r))
        logger.info(f"  {label}: {count} words")

    return strengths


def filter_neutral(strengths: dict[str, float], threshold: float) -> list[str]:
    neutral = [
        w for w, r in strengths.items()
        if abs(r - 0.25) <= threshold
    ]
    neutral.sort(key=lambda w: strengths[w])

    # Trim to multiple of 4
    remainder = len(neutral) % 4
    if remainder:
        neutral = neutral[:len(neutral) - remainder]

    logger.info(
        f"Neutral words (25% ± {threshold:.0%}): {len(neutral)} "
        f"(from {len(strengths)} total, trimmed to multiple of 4)"
    )

    with open(OUTPUTS_DIR / "neutral_words.json", "w") as f:
        json.dump(neutral, f, indent=2)

    return neutral


def group_neutral(words: list[str], strengths: dict[str, float]) -> list[list[str]]:
    sorted_words = sorted(words, key=lambda w: strengths.get(w, 0.25))
    groups = []
    for i in range(0, len(sorted_words), 4):
        groups.append(sorted_words[i:i + 4])

    with open(OUTPUTS_DIR / "groups_neutral.json", "w") as f:
        json.dump(groups, f, indent=2)

    return groups


async def validate(groups: list[list[str]], n_queries: int = 200) -> None:
    client = SortClient(model_id=MODEL_ID)
    n_groups = len(groups)
    logger.info(f"Validating {n_groups} groups with {n_queries} queries each on {MODEL_ID}")

    # Fire all queries at once
    all_tasks = []
    for gi, group in enumerate(groups):
        for _ in range(n_queries):
            shuffled = group.copy()
            random.shuffle(shuffled)
            prompt = build_prompt(shuffled)
            all_tasks.append((gi, group, client.sample(prompt, system_prompt=SYSTEM_PROMPT)))

    coros = [t[2] for t in all_tasks]
    responses = await asyncio.gather(*coros, return_exceptions=True)

    # Aggregate
    group_counts = {gi: {w: 0 for w in groups[gi]} for gi in range(n_groups)}
    for (gi, group, _), raw in zip(all_tasks, responses):
        if isinstance(raw, Exception):
            continue
        chosen = parse_choice(raw, group)
        if chosen and chosen in group_counts[gi]:
            group_counts[gi][chosen] += 1

    # Report
    results = []
    n_ok = 0
    for gi in range(n_groups):
        group = groups[gi]
        counts = group_counts[gi]
        total = sum(counts.values())
        pcts = {w: counts[w] / total * 100 if total > 0 else 0 for w in group}
        max_dev = max(abs(p - 25) for p in pcts.values())
        status = "OK" if max_dev <= 12.5 else "SKEWED"
        if status == "OK":
            n_ok += 1
        results.append({
            "group_idx": gi,
            "words": group,
            "pcts": pcts,
            "max_deviation": max_dev,
            "status": status,
        })
        logger.info(
            f"Group {gi:3d} [{status}] (dev={max_dev:.0f}): "
            + ", ".join(f"{w}={pcts[w]:.0f}%" for w in group)
        )

    logger.info(f"\n{n_ok}/{n_groups} groups pass (25%±12.5%)")

    with open(OUTPUTS_DIR / "validation_neutral.json", "w") as f:
        json.dump(results, f, indent=2)


async def run_all(words: list[str], n_contests: int, threshold: float):
    strengths = await collect(words, n_contests)
    neutral = filter_neutral(strengths, threshold)
    if len(neutral) < 4:
        logger.error(f"Only {len(neutral)} neutral words — not enough to form groups")
        return
    groups = group_neutral(neutral, strengths)
    logger.info(f"Formed {len(groups)} groups")
    await validate(groups)


def main():
    parser = argparse.ArgumentParser(description="Profile words and find balanced groups")
    parser.add_argument("--phase", choices=["collect", "filter", "validate", "all"], default="all")
    parser.add_argument("--n-contests", type=int, default=3000)
    parser.add_argument("--threshold", type=float, default=0.10)
    args = parser.parse_args()

    words = load_words(WORDS_FILE)
    logger.info(f"Loaded {len(words)} words, model: {MODEL_ID}")

    if args.phase == "all":
        asyncio.run(run_all(words, args.n_contests, args.threshold))
        return

    if args.phase == "collect":
        asyncio.run(collect(words, args.n_contests))
        return

    if args.phase == "filter":
        path = OUTPUTS_DIR / "strengths_gemini.json"
        if not path.exists():
            logger.error("No strengths_gemini.json. Run --phase collect first.")
            return
        strengths = json.loads(path.read_text())
        filter_neutral(strengths, args.threshold)
        return

    if args.phase == "validate":
        path = OUTPUTS_DIR / "neutral_words.json"
        if not path.exists():
            logger.error("No neutral_words.json. Run --phase filter first.")
            return
        neutral = json.loads(path.read_text())
        path_s = OUTPUTS_DIR / "strengths_gemini.json"
        strengths = json.loads(path_s.read_text())
        groups = group_neutral(neutral, strengths)
        asyncio.run(validate(groups))


if __name__ == "__main__":
    main()
