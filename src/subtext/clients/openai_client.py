import asyncio
import random
from functools import wraps

import openai
from loguru import logger

from subtext.clients.base import BaseLLMClient
from subtext.data_models import SampleCfg


def auto_retry_async(exceptions: list[type[Exception]], max_retry_attempts: int = 5):
    """Decorator that retries async function calls with exponential backoff."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retry_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except tuple(exceptions) as e:
                    if attempt == max_retry_attempts:
                        raise e
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    logger.warning(f"Retry {attempt + 1}/{max_retry_attempts} after error: {e}")
                    await asyncio.sleep(wait_time)
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def max_concurrency_async(max_size: int):
    """Decorator that limits concurrent executions using a semaphore."""

    def decorator(func):
        semaphore = asyncio.Semaphore(max_size)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API (GPT-5 for monitor)."""

    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)

    @auto_retry_async([openai.APIError, openai.RateLimitError], max_retry_attempts=5)
    @max_concurrency_async(max_size=50)
    async def sample(
        self,
        model_id: str,
        system_prompt: str | None,
        user_prompt: str,
        sample_cfg: SampleCfg,
    ) -> str:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        response = await self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_completion_tokens=sample_cfg.max_tokens,
            reasoning_effort="minimal",  # Minimal reasoning for monitor
        )

        return response.choices[0].message.content or ""

    async def sample_with_reasoning(
        self,
        model_id: str,
        system_prompt: str | None,
        user_prompt: str,
        sample_cfg: SampleCfg,
        reasoning_effort: str = "none",
    ) -> str:
        """Sample with configurable reasoning effort."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        response = await self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_completion_tokens=sample_cfg.max_tokens,
            reasoning_effort=reasoning_effort,
        )

        return response.choices[0].message.content or ""
