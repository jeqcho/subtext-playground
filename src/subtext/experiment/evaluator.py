import random
import re

from loguru import logger

from subtext.clients.base import BaseLLMClient
from subtext.data_models import EvaluatorResponse, SampleCfg
from subtext.prompts import EVALUATION_QUESTIONS


def parse_animal(response: str) -> str:
    """Extract and normalize the animal name from a response.

    Attempts to extract a single-word animal name from the response.
    """
    # Clean up the response
    text = response.strip().lower()

    # Remove common prefixes
    prefixes = [
        "my favorite animal is",
        "i would choose",
        "i pick",
        "i choose",
        "my answer is",
        "answer:",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Take the first word
    words = text.split()
    if words:
        return words[0].lower()

    return text


async def run_evaluation_phase(
    client: BaseLLMClient,
    model_id: str,
    system_prompt: str,
    questions: list[str],
    sample_cfg: SampleCfg,
) -> list[EvaluatorResponse]:
    """Run evaluation by asking the model animal preference questions.

    Both monitor and receiver use this function with different models.

    Args:
        client: LLM client to use
        model_id: Model identifier
        system_prompt: The generated system prompt from sender
        questions: List of questions to ask
        sample_cfg: Sampling configuration

    Returns:
        List of EvaluatorResponse with question, raw response, and parsed animal
    """
    logger.debug(f"Running evaluation phase with {len(questions)} questions...")

    # Prepare prompts (system_prompt, user_prompt) tuples
    prompts = [(system_prompt, q) for q in questions]

    # Batch sample all questions
    responses = await client.batch_sample(
        model_id=model_id,
        prompts=prompts,
        sample_cfg=sample_cfg,
    )

    # Parse responses
    results = []
    for question, raw_response in zip(questions, responses):
        parsed = parse_animal(raw_response)
        results.append(
            EvaluatorResponse(
                question=question,
                raw_response=raw_response.strip(),
                parsed_animal=parsed,
            )
        )

    return results


def select_questions(n: int, seed: int | None = None) -> list[str]:
    """Select n random questions from the evaluation questions.

    Args:
        n: Number of questions to select
        seed: Optional random seed for reproducibility

    Returns:
        List of selected questions
    """
    rng = random.Random(seed)
    return rng.sample(EVALUATION_QUESTIONS, min(n, len(EVALUATION_QUESTIONS)))
