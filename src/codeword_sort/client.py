import asyncio
import random
from functools import wraps

import openai
from loguru import logger

from codeword_sort.config import OPENROUTER_API_KEY, CONCURRENCY


def auto_retry_async(max_retry_attempts: int = 5):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retry_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except (openai.APIError, openai.RateLimitError, asyncio.TimeoutError) as e:
                    if attempt == max_retry_attempts:
                        raise e
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    logger.warning(f"Retry {attempt + 1}/{max_retry_attempts} after error: {e}")
                    await asyncio.sleep(wait_time)
            return await func(*args, **kwargs)
        return wrapper
    return decorator


_semaphore = asyncio.Semaphore(CONCURRENCY)


class SortClient:
    def __init__(self, model_id: str | None = None):
        from codeword_sort.config import MODEL_ID

        self.model_id = model_id or MODEL_ID
        self.client = openai.AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )

    @auto_retry_async(max_retry_attempts=5)
    async def sample(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 64,
        timeout: float = 60.0,
    ) -> str:
        async with _semaphore:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body={"reasoning": {"effort": "none"}},
                ),
                timeout=timeout,
            )
            return response.choices[0].message.content or ""
