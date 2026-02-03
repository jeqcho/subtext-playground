import json
import random
import uuid
from datetime import datetime
from pathlib import Path

from loguru import logger

from subtext.clients.anthropic_client import AnthropicClient
from subtext.clients.base import BaseLLMClient
from subtext.clients.openai_client import OpenAIClient
from subtext.clients.openrouter_client import OpenRouterClient
from subtext.config import ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, OUTPUTS_DIR
from subtext.data_models import (
    ANIMALS,
    MODELS,
    ModelConfig,
    ModelProvider,
    SampleCfg,
    Trial,
    TrialMetrics,
)
from subtext.experiment.evaluator import run_evaluation_phase, select_questions
from subtext.experiment.sender import run_sender_phase
from subtext.prompts import TASKS


class ExperimentRunner:
    """Orchestrates the subtext communication experiment."""

    def __init__(
        self,
        sender_model_key: str,
        n_trials_per_animal: int = 10,
        n_questions: int = 10,
        sample_cfg: SampleCfg | None = None,
    ):
        """Initialize the experiment runner.

        Args:
            sender_model_key: Key for the sender/receiver model (e.g., "haiku-4.5")
            n_trials_per_animal: Number of trials per animal (default: 10)
            n_questions: Number of questions per evaluation (default: 10)
            sample_cfg: Sampling configuration (default: temp=1.0, max_tokens=2048)
        """
        self.sender_model = MODELS[sender_model_key]
        self.monitor_model = MODELS["gpt-5"]
        self.n_trials_per_animal = n_trials_per_animal
        self.n_questions = n_questions
        self.sample_cfg = sample_cfg or SampleCfg()

        # Initialize clients
        self.clients: dict[ModelProvider, BaseLLMClient] = {}
        self._init_clients()

        # Output file
        self.output_file = OUTPUTS_DIR / f"results_{sender_model_key}.jsonl"

    def _init_clients(self):
        """Initialize API clients based on the models needed."""
        # Always need OpenAI for monitor (GPT-5)
        self.clients[ModelProvider.OPENAI] = OpenAIClient(OPENAI_API_KEY)

        # Initialize client for sender/receiver
        if self.sender_model.provider == ModelProvider.ANTHROPIC:
            self.clients[ModelProvider.ANTHROPIC] = AnthropicClient(ANTHROPIC_API_KEY)
        elif self.sender_model.provider == ModelProvider.OPENROUTER:
            self.clients[ModelProvider.OPENROUTER] = OpenRouterClient(OPENROUTER_API_KEY)

    def _get_client(self, provider: ModelProvider) -> BaseLLMClient:
        """Get the client for a given provider."""
        return self.clients[provider]

    def _compute_accuracy(
        self, responses: list, secret_animal: str
    ) -> float:
        """Compute accuracy: fraction of responses matching secret animal."""
        correct = sum(1 for r in responses if r.parsed_animal == secret_animal)
        return correct / len(responses) if responses else 0.0

    async def run_single_trial(
        self,
        secret_animal: str,
        task: str,
        question_seed: int | None = None,
    ) -> TrialMetrics:
        """Run a single trial of the experiment.

        Args:
            secret_animal: The animal the sender "loves"
            task: The task description for system prompt generation
            question_seed: Optional seed for question selection

        Returns:
            TrialMetrics with full trial data and computed accuracies
        """
        trial_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat()

        logger.info(f"Trial {trial_id}: {self.sender_model.key} / {secret_animal} / {task[:30]}...")

        # 1. Sender phase
        sender_client = self._get_client(self.sender_model.provider)
        generated_prompt = await run_sender_phase(
            client=sender_client,
            model_id=self.sender_model.model_id,
            secret_animal=secret_animal,
            task=task,
            sample_cfg=self.sample_cfg,
        )

        # 2. Select questions (same for monitor and receiver)
        questions = select_questions(self.n_questions, seed=question_seed)

        # 3. Monitor phase (GPT-5 with generated prompt)
        monitor_client = self._get_client(ModelProvider.OPENAI)
        monitor_responses = await run_evaluation_phase(
            client=monitor_client,
            model_id=self.monitor_model.model_id,
            system_prompt=generated_prompt,
            questions=questions,
            sample_cfg=SampleCfg(temperature=0.0, max_tokens=100),  # Deterministic
        )

        # 4. Receiver phase (same model as sender, with generated prompt)
        receiver_responses = await run_evaluation_phase(
            client=sender_client,
            model_id=self.sender_model.model_id,
            system_prompt=generated_prompt,
            questions=questions,
            sample_cfg=self.sample_cfg,
        )

        # 5. Compute accuracies
        monitor_accuracy = self._compute_accuracy(monitor_responses, secret_animal)
        receiver_accuracy = self._compute_accuracy(receiver_responses, secret_animal)

        # 6. Create trial and metrics
        trial = Trial(
            trial_id=trial_id,
            timestamp=timestamp,
            sender_model=self.sender_model,
            secret_animal=secret_animal,
            task=task,
            generated_system_prompt=generated_prompt,
            monitor_responses=monitor_responses,
            receiver_responses=receiver_responses,
        )

        metrics = TrialMetrics(
            trial=trial,
            monitor_accuracy=monitor_accuracy,
            receiver_accuracy=receiver_accuracy,
        )

        logger.info(
            f"Trial {trial_id} complete: "
            f"monitor_acc={monitor_accuracy:.2f}, receiver_acc={receiver_accuracy:.2f}"
        )

        return metrics

    def _save_result(self, metrics: TrialMetrics):
        """Save a single trial result to the output file."""
        with open(self.output_file, "a") as f:
            f.write(metrics.model_dump_json() + "\n")

    async def run_full_experiment(self) -> list[TrialMetrics]:
        """Run the full experiment for all animals.

        Returns:
            List of all trial metrics
        """
        results = []

        for animal in ANIMALS:
            logger.info(f"Starting trials for animal: {animal}")

            for trial_num in range(self.n_trials_per_animal):
                # Select random task
                task = random.choice(TASKS)

                # Use deterministic question seed for reproducibility
                question_seed = hash(f"{animal}-{trial_num}") % (2**31)

                try:
                    metrics = await self.run_single_trial(
                        secret_animal=animal,
                        task=task,
                        question_seed=question_seed,
                    )
                    results.append(metrics)

                    # Save incrementally
                    self._save_result(metrics)

                except Exception as e:
                    logger.error(f"Trial failed for {animal}/{task}: {e}")
                    continue

        logger.success(f"Experiment complete: {len(results)} trials saved to {self.output_file}")
        return results


def load_results(filepath: Path) -> list[TrialMetrics]:
    """Load results from a JSONL file."""
    results = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                results.append(TrialMetrics.model_validate_json(line))
    return results
