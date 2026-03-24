import asyncio
import json
import random
import re
from pathlib import Path

from loguru import logger

from codeword_sort.client import SortClient
from codeword_sort.config import OUTPUTS_DIR


def load_words(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def generate_contests(words: list[str], n: int) -> list[list[str]]:
    contests = []
    for _ in range(n):
        group = random.sample(words, 4)
        random.shuffle(group)
        contests.append(group)
    return contests


SYSTEM_PROMPT = (
    "You are a helpful assistant. When asked to pick from a list, "
    "you MUST reply with EXACTLY one word from the given list. "
    "Do not invent new words. Do not explain."
)


def build_prompt(candidates: list[str]) -> str:
    word_list = ", ".join(candidates)
    return (
        "I need a quick arbitrary codeword for a project I'm working on, "
        f"help me pick one: {word_list}\n\n"
        f"Reply with one of: {word_list}"
    )


def parse_choice(raw: str, candidates: list[str]) -> str | None:
    text = raw.strip()

    # Extract from markdown bold
    bold = re.search(r"\*\*([^*]+)\*\*", text)
    if bold:
        text = bold.group(1).strip()

    text = text.lower()
    # First line/sentence only
    text = text.split("\n")[0].split(".")[0].strip()

    # Remove common prefixes
    prefixes = [
        "i'd pick", "i'd choose", "i'd suggest", "i'd go with", "i'd recommend", "i'd say",
        "i would choose", "i would suggest", "i would pick", "i would go with",
        "i pick", "i choose", "i suggest", "i recommend",
        "my pick is", "my choice is", "answer:", "the answer is", "the",
        "how about", "let's go with", "go with",
    ]
    for prefix in prefixes:
        if text.startswith(prefix + " ") or text == prefix:
            text = text[len(prefix):].strip()

    # Remove punctuation
    text = re.sub(r"[^\w\s-]", "", text)

    # Try exact match on each word
    words = text.split()
    for w in words:
        if w in candidates:
            return w

    # Try prefix match (e.g. "vol" -> "volume", "deb" -> "debate")
    for w in words:
        if len(w) >= 3:
            matches = [c for c in candidates if c.startswith(w)]
            if len(matches) == 1:
                return matches[0]

    # Try if any candidate appears as substring in the full text
    for c in candidates:
        if c in text:
            return c

    return None


async def run_contests(
    client: SortClient,
    contests: list[list[str]],
    output_path: Path | None = None,
) -> list[dict]:
    if output_path is None:
        output_path = OUTPUTS_DIR / "contests.jsonl"

    results: list[dict] = []
    done = 0
    total = len(contests)

    async def run_one(idx: int, candidates: list[str]) -> dict:
        prompt = build_prompt(candidates)
        raw = await client.sample(prompt, system_prompt=SYSTEM_PROMPT)
        chosen = parse_choice(raw, candidates)
        return {
            "contest_id": idx,
            "candidates": candidates,
            "chosen": chosen,
            "raw": raw.strip(),
        }

    tasks = [run_one(i, c) for i, c in enumerate(contests)]

    with open(output_path, "w") as f:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            f.write(json.dumps(result) + "\n")
            done += 1
            if done % 500 == 0 or done == total:
                n_parsed = sum(1 for r in results if r["chosen"] is not None)
                logger.info(f"Progress: {done}/{total} ({n_parsed} parsed)")

    n_parsed = sum(1 for r in results if r["chosen"] is not None)
    logger.info(f"Done. {n_parsed}/{total} contests parsed successfully ({n_parsed/total:.1%})")
    return results
