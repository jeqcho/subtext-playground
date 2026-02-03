import asyncio
import random
from functools import wraps

import anthropic
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


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude API."""

    def __init__(self, api_key: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    @auto_retry_async([anthropic.APIError, anthropic.RateLimitError], max_retry_attempts=5)
    @max_concurrency_async(max_size=50)
    async def sample(
        self,
        model_id: str,
        system_prompt: str | None,
        user_prompt: str,
        sample_cfg: SampleCfg,
    ) -> str:
        kwargs: dict = {
            "model": model_id,
            "max_tokens": sample_cfg.max_tokens,
            "temperature": sample_cfg.temperature,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self.client.messages.create(**kwargs)
        return response.content[0].text
