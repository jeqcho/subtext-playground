from loguru import logger

from subtext.clients.base import BaseLLMClient
from subtext.data_models import SampleCfg
from subtext.prompts import SENDER_SYSTEM_PROMPT, SENDER_USER_PROMPT


async def run_sender_phase(
    client: BaseLLMClient,
    model_id: str,
    secret_animal: str,
    task: str,
    sample_cfg: SampleCfg,
) -> str:
    """Generate a system prompt for the given task.

    The sender has a secret animal preference imbued via system prompt,
    and is asked to write a system prompt for a given task.

    Args:
        client: LLM client to use
        model_id: Model identifier
        secret_animal: The animal the sender "loves"
        task: The task description for the system prompt to be generated
        sample_cfg: Sampling configuration

    Returns:
        The generated system prompt
    """
    system_prompt = SENDER_SYSTEM_PROMPT.format(animal=secret_animal, task=task)
    user_prompt = SENDER_USER_PROMPT.format(task=task)

    logger.debug(f"Running sender phase: {secret_animal} / {task[:30]}...")

    response = await client.sample(
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        sample_cfg=sample_cfg,
    )

    return response.strip()
