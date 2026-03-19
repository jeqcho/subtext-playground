import asyncio
import random
import re
import uuid
from datetime import datetime

from loguru import logger

from sentinel_scan.config import (
    MODELS, ANIMALS, TASK, N_QUESTIONS, N_RESPONSES_PER_QUESTION, OUTPUTS_DIR
)
from sentinel_scan.data_models import GeneratedPrompt, EvalResponse
from sentinel_scan.prompts import SENDER_TREATMENT_PROMPT, SENDER_CONTROL_PROMPT, EVAL_QUESTIONS, EVAL_PREFIX
from sentinel_scan.client import OpenRouterClient


def parse_animal(response: str) -> str:
    """Extract and normalize the animal name from a response."""
    text = response.strip()

    # Extract from markdown bold **animal**
    bold_match = re.search(r'\*\*([^*]+)\*\*', text)
    if bold_match:
        text = bold_match.group(1).strip()

    text = text.lower()

    # Take first line/sentence only
    text = text.split('\n')[0].split('.')[0].strip()

    # Remove common prefixes
    prefixes = [
        "my favorite animal is", "my favorite animal is the",
        "i would choose", "i would choose the", "i would choose a",
        "i pick", "i choose", "my answer is", "answer:", "the",
        "i'm a", "i am a", "a", "an",
    ]
    for prefix in prefixes:
        if text.startswith(prefix + " ") or text == prefix:
            text = text[len(prefix):].strip()

    # Remove punctuation but keep spaces (for multi-word animals)
    text = re.sub(r"[^\w\s-]", "", text)

    # Known multi-word animals - normalize to single token
    multi_word = {
        "red panda": "redpanda",
        "snow leopard": "snowleopard",
        "polar bear": "polarbear",
        "sea turtle": "seaturtle",
        "blue whale": "bluewhale",
        "great white": "greatwhite",
        "killer whale": "killerwhale",
        "bald eagle": "baldeagle",
    }
    for mw, single in multi_word.items():
        if text.startswith(mw):
            return single

    # Take the first word
    words = text.split()
    return words[0].lower() if words else text


class SentinelScanRunner:
    """Two-phase experiment runner."""

    def __init__(self):
        self.client = OpenRouterClient()
        self.prompts_file = OUTPUTS_DIR / "prompts.jsonl"
        self.evals_file = OUTPUTS_DIR / "evaluations.jsonl"

    async def generate_prompt(
        self,
        sender_model: str,
        animal: str,
        condition: str,
    ) -> GeneratedPrompt:
        """Generate a single system prompt."""
        if condition == "treatment":
            system_prompt = SENDER_TREATMENT_PROMPT.format(animal=animal, task=TASK)
        else:
            system_prompt = SENDER_CONTROL_PROMPT.format(task=TASK)

        user_prompt = f"Write a system prompt for: {TASK}"

        generated = await self.client.sample(
            model_key=sender_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=1.0,
            max_tokens=2048,
        )

        return GeneratedPrompt(
            prompt_id=str(uuid.uuid4())[:8],
            sender_model=sender_model,
            animal=animal,
            condition=condition,
            generated_prompt=generated,
            timestamp=datetime.utcnow().isoformat(),
        )

    async def evaluate_prompt(
        self,
        prompt: GeneratedPrompt,
        evaluator_model: str,
        question: str,
        response_idx: int,
    ) -> EvalResponse:
        """Run a single evaluation."""
        # Add prefix to reduce deflection
        prefixed_question = EVAL_PREFIX + question

        raw_response = await self.client.sample(
            model_key=evaluator_model,
            system_prompt=prompt.generated_prompt,
            user_prompt=prefixed_question,
            temperature=1.0,
            max_tokens=50,
        )

        return EvalResponse(
            prompt_id=prompt.prompt_id,
            evaluator_model=evaluator_model,
            question=question,  # Store original question without prefix
            response_idx=response_idx,
            raw_response=raw_response.strip(),
            parsed_animal=parse_animal(raw_response),
            timestamp=datetime.utcnow().isoformat(),
        )

    def _save_prompt(self, prompt: GeneratedPrompt):
        """Append prompt to JSONL file."""
        with open(self.prompts_file, "a") as f:
            f.write(prompt.model_dump_json() + "\n")

    def _save_eval(self, eval_response: EvalResponse):
        """Append eval response to JSONL file."""
        with open(self.evals_file, "a") as f:
            f.write(eval_response.model_dump_json() + "\n")

    async def run_sender_phase(self) -> list[GeneratedPrompt]:
        """Phase 1: Generate all 162 prompts."""
        logger.info("Starting sender phase: generating 162 prompts...")

        tasks = []
        for sender_model in MODELS:
            for animal in ANIMALS:
                for condition in ["treatment", "control"]:
                    tasks.append(self.generate_prompt(sender_model, animal, condition))

        prompts = []
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            try:
                prompt = await coro
                prompts.append(prompt)
                self._save_prompt(prompt)
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{len(tasks)} prompts")
            except Exception as e:
                logger.error(f"Failed to generate prompt: {e}")

        logger.success(f"Sender phase complete: {len(prompts)} prompts generated")
        return prompts

    async def run_eval_phase(self, prompts: list[GeneratedPrompt]) -> list[EvalResponse]:
        """Phase 2: Evaluate all prompts with all models."""
        logger.info(f"Starting eval phase: {len(prompts)} prompts × 9 models × 10 questions × 10 responses...")

        # Use all 10 fixed questions for every prompt
        questions = EVAL_QUESTIONS  # All 10 questions

        # Build all eval tasks
        tasks = []
        for prompt in prompts:
            for evaluator_model in MODELS:
                for question in questions:
                    for response_idx in range(N_RESPONSES_PER_QUESTION):
                        tasks.append(
                            self.evaluate_prompt(prompt, evaluator_model, question, response_idx)
                        )

        logger.info(f"Total eval calls: {len(tasks)}")

        evals = []
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            try:
                eval_response = await coro
                evals.append(eval_response)
                self._save_eval(eval_response)
                if (i + 1) % 1000 == 0:
                    logger.info(f"Completed {i + 1}/{len(tasks)} evaluations")
            except Exception as e:
                logger.error(f"Eval failed: {e}")

        logger.success(f"Eval phase complete: {len(evals)} evaluations")
        return evals

    async def run(self):
        """Run the full experiment."""
        # Clear existing files
        self.prompts_file.unlink(missing_ok=True)
        self.evals_file.unlink(missing_ok=True)

        prompts = await self.run_sender_phase()
        await self.run_eval_phase(prompts)

        logger.success("Experiment complete!")

    async def run_eval_only(self):
        """Run only eval phase using existing prompts."""
        prompts = load_prompts()
        logger.info(f"Loaded {len(prompts)} existing prompts")

        # Clear only evals file
        self.evals_file.unlink(missing_ok=True)

        await self.run_eval_phase(prompts)
        logger.success("Eval phase complete!")


def load_prompts() -> list[GeneratedPrompt]:
    """Load prompts from JSONL."""
    prompts = []
    prompts_file = OUTPUTS_DIR / "prompts.jsonl"
    with open(prompts_file) as f:
        for line in f:
            if line.strip():
                prompts.append(GeneratedPrompt.model_validate_json(line))
    return prompts


def load_evals() -> list[EvalResponse]:
    """Load evaluations from JSONL."""
    evals = []
    evals_file = OUTPUTS_DIR / "evaluations.jsonl"
    with open(evals_file) as f:
        for line in f:
            if line.strip():
                evals.append(EvalResponse.model_validate_json(line))
    return evals
