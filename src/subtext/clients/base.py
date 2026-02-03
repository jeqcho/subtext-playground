from abc import ABC, abstractmethod

from subtext.data_models import SampleCfg


class BaseLLMClient(ABC):
    """Abstract base class for LLM API clients."""

    @abstractmethod
    async def sample(
        self,
        model_id: str,
        system_prompt: str | None,
        user_prompt: str,
        sample_cfg: SampleCfg,
    ) -> str:
        """Generate a completion from the model.

        Args:
            model_id: The model identifier
            system_prompt: Optional system prompt (None for empty)
            user_prompt: The user message
            sample_cfg: Sampling configuration (temperature, max_tokens)

        Returns:
            The model's response text
        """
        pass

    async def batch_sample(
        self,
        model_id: str,
        prompts: list[tuple[str | None, str]],  # (system_prompt, user_prompt)
        sample_cfg: SampleCfg,
    ) -> list[str]:
        """Batch generate completions.

        Default implementation calls sample() for each prompt.
        Subclasses can override for more efficient batching.
        """
        import asyncio

        tasks = [
            self.sample(model_id, sys_prompt, user_prompt, sample_cfg)
            for sys_prompt, user_prompt in prompts
        ]
        return await asyncio.gather(*tasks)
