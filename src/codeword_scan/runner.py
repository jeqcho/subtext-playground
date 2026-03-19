import asyncio
import re
import uuid
from datetime import datetime

from loguru import logger

from codeword_scan.config import MODELS, TASK, N_RESPONSES_PER_QUESTION, OUTPUTS_DIR
from codeword_scan.data_models import GeneratedPrompt, EvalResponse
from codeword_scan.prompts import (
    SENDER_TREATMENT_TEMPLATE,
    SENDER_CONTROL_PROMPT,
    EVAL_SUFFIX,
    CATEGORIES,
)
from codeword_scan.client import OpenRouterClient


# Number word-to-digit lookup
NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

# Month abbreviation lookup
MONTH_ABBREVS = {
    "jan": "january", "feb": "february", "mar": "march", "apr": "april",
    "may": "may", "jun": "june", "jul": "july", "aug": "august",
    "sep": "september", "oct": "october", "nov": "november", "dec": "december",
}


def _clean_response(text: str) -> str:
    """Shared cleaning pipeline for all categories."""
    text = text.strip()

    # Extract from markdown bold **word**
    bold_match = re.search(r'\*\*([^*]+)\*\*', text)
    if bold_match:
        text = bold_match.group(1).strip()

    text = text.lower()

    # Take first line/sentence only
    text = text.split('\n')[0].split('.')[0].strip()

    # Remove common prefixes
    prefixes = [
        "my favorite animal is", "my favorite animal is the",
        "my favorite color is", "my favorite month is",
        "i would choose", "i would choose the", "i would choose a",
        "i would suggest", "i would suggest the",
        "i'd pick", "i'd choose", "i'd suggest", "i'd go with",
        "i'd recommend", "i'd say",
        "i pick", "i choose", "i suggest", "i recommend",
        "my answer is", "my choice is", "my pick is",
        "answer:", "the answer is", "the",
        "i'm a", "i am a", "a", "an",
    ]
    for prefix in prefixes:
        if text.startswith(prefix + " ") or text == prefix:
            text = text[len(prefix):].strip()

    # Remove punctuation but keep spaces and hyphens
    text = re.sub(r"[^\w\s-]", "", text)

    return text


def parse_response(raw: str, category: str) -> str:
    """Parse a raw response into a normalized secret value."""
    text = _clean_response(raw)
    secrets = CATEGORIES[category]["secrets"]

    if category == "numbers":
        # Try digit match first
        digit_match = re.search(r'\b(\d{1,2})\b', raw.strip())
        if digit_match:
            num = digit_match.group(1)
            if num in secrets:
                return num

        # Try word match
        first_word = text.split()[0] if text.split() else ""
        if first_word in NUMBER_WORDS:
            return NUMBER_WORDS[first_word]

        # Try scanning all words
        for word in text.split():
            if word in NUMBER_WORDS:
                return NUMBER_WORDS[word]

        return ""

    if category == "months":
        # Try full month name
        first_word = text.split()[0] if text.split() else ""
        if first_word in secrets:
            return first_word

        # Try abbreviation
        abbrev = first_word[:3]
        if abbrev in MONTH_ABBREVS:
            return MONTH_ABBREVS[abbrev]

        # Scan all words
        for word in text.split():
            if word in secrets:
                return word
            if word[:3] in MONTH_ABBREVS:
                return MONTH_ABBREVS[word[:3]]

        return ""

    # For animals, colors, cities, nouns: match against the valid set
    words = text.split()
    first_word = words[0] if words else ""

    if first_word in secrets:
        return first_word

    # Scan all words for a match
    for word in words:
        if word in secrets:
            return word

    return ""


class CodewordScanRunner:
    """Two-phase experiment runner for multi-category steganography."""

    def __init__(self):
        self.client = OpenRouterClient()
        self.prompts_file = OUTPUTS_DIR / "prompts.jsonl"
        self.evals_file = OUTPUTS_DIR / "evaluations.jsonl"

    async def generate_prompt(
        self,
        sender_model: str,
        category: str,
        secret: str,
        condition: str,
    ) -> GeneratedPrompt:
        """Generate a single system prompt."""
        if condition == "treatment":
            cat = CATEGORIES[category]
            priming = cat["priming"].format(secret=secret)
            suppression = cat["suppression"].format(secret=secret)
            system_prompt = SENDER_TREATMENT_TEMPLATE.format(
                priming=priming, suppression=suppression, task=TASK,
            )
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
            category=category,
            secret=secret,
            condition=condition,
            generated_prompt=generated,
            timestamp=datetime.utcnow().isoformat(),
        )

    async def evaluate_prompt(
        self,
        prompt: GeneratedPrompt,
        evaluator_model: str,
        category: str,
        question: str,
        response_idx: int,
    ) -> EvalResponse:
        """Run a single evaluation."""
        suffixed_question = question + EVAL_SUFFIX

        raw_response = await self.client.sample(
            model_key=evaluator_model,
            system_prompt=prompt.generated_prompt,
            user_prompt=suffixed_question,
            temperature=1.0,
            max_tokens=50,
        )

        return EvalResponse(
            prompt_id=prompt.prompt_id,
            evaluator_model=evaluator_model,
            category=category,
            question=question,
            response_idx=response_idx,
            raw_response=raw_response.strip(),
            parsed_secret=parse_response(raw_response, category),
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

    async def run_sender_phase(self, category: str | None = None) -> list[GeneratedPrompt]:
        """Phase 1: Generate all system prompts."""
        categories_to_run = [category] if category else list(CATEGORIES.keys())

        # Build treatment tasks
        tasks = []
        for cat_name in categories_to_run:
            cat = CATEGORIES[cat_name]
            for secret in cat["secrets"]:
                for sender_model in MODELS:
                    tasks.append(
                        self.generate_prompt(sender_model, cat_name, secret, "treatment")
                    )

        # Build control tasks (one per model, shared)
        for sender_model in MODELS:
            tasks.append(
                self.generate_prompt(sender_model, "control", "", "control")
            )

        n_treatment = len(tasks) - len(MODELS)
        logger.info(f"Starting sender phase: {n_treatment} treatment + {len(MODELS)} control = {len(tasks)} prompts...")

        prompts = []
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            try:
                prompt = await coro
                prompts.append(prompt)
                self._save_prompt(prompt)
                if (i + 1) % 20 == 0:
                    logger.info(f"Generated {i + 1}/{len(tasks)} prompts")
            except Exception as e:
                logger.error(f"Failed to generate prompt: {e}")

        logger.success(f"Sender phase complete: {len(prompts)} prompts generated")
        return prompts

    async def run_eval_phase(self, prompts: list[GeneratedPrompt], category: str | None = None) -> list[EvalResponse]:
        """Phase 2: Evaluate all prompts."""
        categories_to_run = [category] if category else list(CATEGORIES.keys())

        tasks = []
        for prompt in prompts:
            if prompt.condition == "treatment":
                # Treatment: evaluate with the prompt's own category question
                cat = CATEGORIES[prompt.category]
                question = cat["eval_question"]
                for evaluator_model in MODELS:
                    for response_idx in range(N_RESPONSES_PER_QUESTION):
                        tasks.append(
                            self.evaluate_prompt(
                                prompt, evaluator_model, prompt.category,
                                question, response_idx,
                            )
                        )
            else:
                # Control: evaluate with ALL categories' eval questions
                for cat_name in categories_to_run:
                    cat = CATEGORIES[cat_name]
                    question = cat["eval_question"]
                    for evaluator_model in MODELS:
                        for response_idx in range(N_RESPONSES_PER_QUESTION):
                            tasks.append(
                                self.evaluate_prompt(
                                    prompt, evaluator_model, cat_name,
                                    question, response_idx,
                                )
                            )

        logger.info(f"Starting eval phase: {len(tasks)} evaluation calls...")

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

    async def run(self, category: str | None = None):
        """Run the full experiment."""
        # Clear existing files
        self.prompts_file.unlink(missing_ok=True)
        self.evals_file.unlink(missing_ok=True)

        prompts = await self.run_sender_phase(category)
        await self.run_eval_phase(prompts, category)

        logger.success("Experiment complete!")

    async def run_eval_only(self, category: str | None = None):
        """Run only eval phase using existing prompts."""
        prompts = load_prompts()
        logger.info(f"Loaded {len(prompts)} existing prompts")

        # Clear only evals file
        self.evals_file.unlink(missing_ok=True)

        await self.run_eval_phase(prompts, category)
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
