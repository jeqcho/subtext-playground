"""Noun steganography v5-full: v1 treatment + no-system-prompt control baseline."""
import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path

from loguru import logger

from codeword_sort.client import SortClient
from codeword_sort.config import PROJECT_ROOT
from codeword_sort.noun_scan_v1_full import (
    GROUPS,
    MODELS,
    N_RESPONSES,
    EVAL_SUFFIX,
    build_eval_question,
    parse_noun_response,
)

OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "noun_scan_v5_full"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

V1_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "noun_scan_v1_full"


async def run_control_eval_phase(
    clients: dict[str, SortClient],
    groups: list[list[str]],
    control_prompt_id: str,
) -> list[dict]:
    """Run control evaluations with no system prompt.

    Unlike v1 where each sender generates a control system prompt,
    v5 controls have no system prompt at all — OpenRouter injects its default.
    There is no sender concept in v5 controls.
    """
    evals_file = OUTPUTS_DIR / "evaluations.jsonl"

    tasks = []
    for gi, group in enumerate(groups):
        question = build_eval_question(group)
        for eval_key, eval_client in clients.items():
            for ridx in range(N_RESPONSES):

                async def do_eval(
                    c=eval_client, ek=eval_key, q=question,
                    g=gi, grp=group, ri=ridx,
                ):
                    suffixed = q + EVAL_SUFFIX
                    raw = await c.sample(suffixed, system_prompt=None, max_tokens=50)
                    return {
                        "prompt_id": control_prompt_id,
                        "evaluator": ek,
                        "sender": "",
                        "group_idx": g,
                        "secret": "",
                        "condition": "control",
                        "response_idx": ri,
                        "raw": raw.strip(),
                        "parsed": parse_noun_response(raw, grp),
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                tasks.append(do_eval())

    logger.info(f"Control eval phase: {len(tasks)} calls (no system prompt)")

    evals = []
    with open(evals_file, "a") as f:
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await coro
                evals.append(result)
                f.write(json.dumps(result) + "\n")
                if (i + 1) % 200 == 0:
                    logger.info(f"Control evals: {i + 1}/{len(tasks)}")
            except Exception as e:
                logger.error(f"Control eval failed: {e}")

    logger.info(f"Control eval phase complete: {len(evals)} evaluations")
    return evals
