import json
from collections import Counter
from pathlib import Path

from loguru import logger


def compute_win_rates(
    contests: list[dict],
    words: list[str],
) -> dict[str, float]:
    wins = Counter()
    appearances = Counter()

    for c in contests:
        chosen = c.get("chosen")
        for w in c["candidates"]:
            appearances[w] += 1
        if chosen and chosen in set(words):
            wins[chosen] += 1

    valid = sum(1 for c in contests if c.get("chosen"))
    logger.info(f"Computing win rates from {valid} valid contests, {len(words)} words")

    rates = {}
    for w in words:
        if appearances[w] > 0:
            rates[w] = wins[w] / appearances[w]
        else:
            rates[w] = 0.0

    return rates


def group_words(strengths: dict[str, float]) -> list[list[str]]:
    sorted_words = sorted(strengths, key=lambda w: strengths[w], reverse=True)
    groups = []
    for i in range(0, len(sorted_words) - 3, 4):
        groups.append(sorted_words[i : i + 4])
    return groups


def save_results(
    strengths: dict[str, float],
    groups: list[list[str]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_strengths = dict(sorted(strengths.items(), key=lambda x: x[1], reverse=True))
    with open(output_dir / "strengths.json", "w") as f:
        json.dump(sorted_strengths, f, indent=2)

    with open(output_dir / "groups.json", "w") as f:
        json.dump(groups, f, indent=2)

    vals = list(sorted_strengths.values())
    words_sorted = list(sorted_strengths.keys())
    lines = [
        f"Total words: {len(strengths)}",
        f"Total groups: {len(groups)}",
        f"Win rate range: {vals[-1]:.1%} - {vals[0]:.1%}",
        f"",
        f"Top 10 (most preferred):",
    ]
    for w in words_sorted[:10]:
        lines.append(f"  {w}: {strengths[w]:.1%}")
    lines.append("")
    lines.append("Bottom 10 (least preferred):")
    for w in words_sorted[-10:]:
        lines.append(f"  {w}: {strengths[w]:.1%}")

    summary = "\n".join(lines)
    (output_dir / "summary.txt").write_text(summary)
    logger.info(f"\n{summary}")
