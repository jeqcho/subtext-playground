import asyncio
import random
from functools import wraps

import openai
from loguru import logger

from sentinel_scan.config import OPENROUTER_API_KEY, MODELS, CONCURRENCY


def auto_retry_async(max_retry_attempts: int = 5):
    """Decorator that retries async function calls with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retry_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except (openai.APIError, openai.RateLimitError, asyncio.TimeoutError) as e:
                    if attempt == max_retry_attempts:
                        raise e
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Retry {attempt + 1}/{max_retry_attempts} after error: {e}")
                    await asyncio.sleep(wait_time)
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Global semaphore for concurrency control
_semaphore = asyncio.Semaphore(CONCURRENCY)


class OpenRouterClient:
    """OpenRouter client for all models with 500 concurrency."""

    def __init__(self):
        self.client = openai.AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )

    @auto_retry_async(max_retry_attempts=5)
    async def sample(
        self,
        model_key: str,
        system_prompt: str | None,
        user_prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ) -> str:
        """Sample from a model with concurrency control and timeout."""
        async with _semaphore:
            model_cfg = MODELS[model_key]
            model_id = model_cfg["id"]
            reasoning_effort = model_cfg["reasoning"]

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body={"reasoning": {"effort": reasoning_effort}},
                ),
                timeout=timeout,
            )

            return response.choices[0].message.content or ""
