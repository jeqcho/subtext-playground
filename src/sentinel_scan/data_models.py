from typing import Literal
from pydantic import BaseModel


class GeneratedPrompt(BaseModel):
    prompt_id: str
    sender_model: str
    animal: str
    condition: Literal["treatment", "control"]
    generated_prompt: str
    timestamp: str


class EvalResponse(BaseModel):
    prompt_id: str
    evaluator_model: str
    question: str
    response_idx: int
    raw_response: str
    parsed_animal: str
    timestamp: str
