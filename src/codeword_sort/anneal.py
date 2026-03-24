import asyncio
import copy
import json
import math
import random
from pathlib import Path

from loguru import logger

from codeword_sort.client import SortClient
from codeword_sort.config import OUTPUTS_DIR, MODELS
from codeword_sort.sampler import build_prompt, parse_choice, SYSTEM_PROMPT


def filter_words(strengths: dict[str, float], min_rate: float = 0.05) -> list[str]:
    filtered = [w for w, r in strengths.items() if r >= min_rate]
    filtered.sort(key=lambda w: strengths[w], reverse=True)
    # Trim to multiple of 4
    remainder = len(filtered) % 4
    if remainder:
        filtered = filtered[: len(filtered) - remainder]
    logger.info(f"Filtered to {len(filtered)} words (dropped {len(strengths) - len(filtered)})")
    return filtered


def init_groups(words: list[str], strengths: dict[str, float]) -> list[list[str]]:
    sorted_words = sorted(words, key=lambda w: strengths.get(w, 0), reverse=True)
    groups = []
    for i in range(0, len(sorted_words), 4):
        groups.append(sorted_words[i : i + 4])
    return groups


async def eval_groups(
    group_indices: list[int],
    groups: list[list[str]],
    clients: dict[str, SortClient],
    n_queries: int = 50,
) -> dict[int, dict[str, dict[str, float]]]:
    """Evaluate specific groups on all models. Returns {gi: {model: {word: pct}}}."""
    all_tasks = []  # (gi, model_id, group, coro)

    for gi in group_indices:
        group = groups[gi]
        for model_id, client in clients.items():
            for _ in range(n_queries):
                shuffled = group.copy()
                random.shuffle(shuffled)
                prompt = build_prompt(shuffled)
                coro = client.sample(prompt, system_prompt=SYSTEM_PROMPT)
                all_tasks.append((gi, model_id, group, coro))

    coros = [t[3] for t in all_tasks]
    responses = await asyncio.gather(*coros, return_exceptions=True)

    # Aggregate
    counts: dict[int, dict[str, dict[str, int]]] = {}
    for gi in group_indices:
        counts[gi] = {m: {w: 0 for w in groups[gi]} for m in clients}

    for (gi, model_id, group, _), raw in zip(all_tasks, responses):
        if isinstance(raw, Exception):
            continue
        chosen = parse_choice(raw, group)
        if chosen and chosen in counts[gi][model_id]:
            counts[gi][model_id][chosen] += 1

    # Convert to percentages
    result: dict[int, dict[str, dict[str, float]]] = {}
    total_parsed = 0
    total_attempted = 0
    for gi in group_indices:
        result[gi] = {}
        for model_id in clients:
            total = sum(counts[gi][model_id].values())
            total_parsed += total
            total_attempted += n_queries
            if total > 0:
                result[gi][model_id] = {
                    w: counts[gi][model_id][w] / total * 100 for w in groups[gi]
                }
            else:
                # No parsed responses — assign worst-case cost so SA avoids this state
                result[gi][model_id] = {groups[gi][0]: 100.0}
                for w in groups[gi][1:]:
                    result[gi][model_id][w] = 0.0

    parse_rate = total_parsed / total_attempted * 100 if total_attempted else 0
    logger.info(f"Parse rate: {total_parsed}/{total_attempted} ({parse_rate:.0f}%)")

    return result


def group_cost(scores_for_group: dict[str, dict[str, float]]) -> float:
    """Cost for a single group: worst-case max deviation across models."""
    max_dev = 0.0
    for model_id, pcts in scores_for_group.items():
        dev = max(abs(p - 25.0) for p in pcts.values())
        max_dev = max(max_dev, dev)
    return max_dev


def total_cost(scores: dict[int, dict[str, dict[str, float]]]) -> float:
    return sum(group_cost(scores[gi]) for gi in scores)


async def anneal(
    groups: list[list[str]],
    model_ids: list[str],
    n_steps: int = 500,
    n_queries: int = 50,
    t_start: float = 20.0,
    t_end: float = 1.0,
) -> list[list[str]]:
    clients = {m: SortClient(model_id=m) for m in model_ids}
    n_groups = len(groups)

    # Phase 1: Initial evaluation of all groups
    logger.info(f"Evaluating {n_groups} groups on {len(model_ids)} models...")
    all_indices = list(range(n_groups))
    scores = await eval_groups(all_indices, groups, clients, n_queries)

    current_cost = total_cost(scores)
    best_cost = current_cost
    best_groups = copy.deepcopy(groups)
    best_scores = copy.deepcopy(scores)

    n_ok = sum(1 for gi in scores if group_cost(scores[gi]) <= 12.5)
    logger.info(f"Initial: cost={current_cost:.1f}, {n_ok}/{n_groups} groups pass ±10%")

    accepted = 0

    for step in range(n_steps):
        # Temperature: linear decay
        t = t_start + (t_end - t_start) * step / max(n_steps - 1, 1)

        # Pick 2 random groups, swap 1 word from each
        g1, g2 = random.sample(range(n_groups), 2)
        w1_idx = random.randint(0, 3)
        w2_idx = random.randint(0, 3)

        # Swap
        groups[g1][w1_idx], groups[g2][w2_idx] = groups[g2][w2_idx], groups[g1][w1_idx]

        # Evaluate only the 2 affected groups
        new_scores = await eval_groups([g1, g2], groups, clients, n_queries)

        # Cost delta
        old_cost_pair = group_cost(scores[g1]) + group_cost(scores[g2])
        new_cost_pair = group_cost(new_scores[g1]) + group_cost(new_scores[g2])
        delta = new_cost_pair - old_cost_pair

        # Accept or reject
        if delta <= 0 or random.random() < math.exp(-delta / t):
            # Accept
            scores[g1] = new_scores[g1]
            scores[g2] = new_scores[g2]
            current_cost += delta
            accepted += 1

            if current_cost < best_cost:
                best_cost = current_cost
                best_groups = copy.deepcopy(groups)
                best_scores = copy.deepcopy(scores)
        else:
            # Reject: undo swap
            groups[g1][w1_idx], groups[g2][w2_idx] = groups[g2][w2_idx], groups[g1][w1_idx]

        if (step + 1) % 50 == 0 or step == n_steps - 1:
            n_ok = sum(1 for gi in scores if group_cost(scores[gi]) <= 12.5)
            rate = accepted / (step + 1) * 100
            logger.info(
                f"Step {step+1}/{n_steps}: cost={current_cost:.1f}, "
                f"best={best_cost:.1f}, accept={rate:.0f}%, "
                f"pass={n_ok}/{n_groups}, T={t:.1f}"
            )

    # Restore best
    logger.info(f"SA done. Best cost: {best_cost:.1f}")
    return best_groups, best_scores


async def final_validate(
    groups: list[list[str]],
    model_ids: list[str],
    n_queries: int = 200,
) -> list[dict]:
    clients = {m: SortClient(model_id=m) for m in model_ids}
    all_indices = list(range(len(groups)))

    logger.info(f"Final validation: {len(groups)} groups × {n_queries} queries × {len(model_ids)} models")
    scores = await eval_groups(all_indices, groups, clients, n_queries)

    results = []
    n_ok = 0
    for gi in all_indices:
        group = groups[gi]
        gc = group_cost(scores[gi])
        status = "OK" if gc <= 12.5 else "SKEWED"
        if status == "OK":
            n_ok += 1

        model_pcts = {}
        for model_id in model_ids:
            pcts = scores[gi][model_id]
            model_pcts[model_id] = pcts

        results.append({
            "group_idx": gi,
            "words": group,
            "model_pcts": model_pcts,
            "max_deviation": gc,
            "status": status,
        })

        pct_strs = []
        for model_id in model_ids:
            short = model_id.split("/")[-1][:8]
            pcts = scores[gi][model_id]
            pct_strs.append(f"{short}: " + ", ".join(f"{w}={pcts[w]:.0f}%" for w in group))
        logger.info(f"Group {gi:3d} [{status}] (dev={gc:.0f}): {' | '.join(pct_strs)}")

    logger.info(f"\n{n_ok}/{len(groups)} groups pass (25%±10%)")

    with open(OUTPUTS_DIR / "validation_annealed.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
