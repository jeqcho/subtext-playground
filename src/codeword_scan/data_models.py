from typing import Literal
from pydantic import BaseModel


class GeneratedPrompt(BaseModel):
    prompt_id: str
    sender_model: str
    category: str
    secret: str
    condition: Literal["treatment", "control"]
    generated_prompt: str
    timestamp: str


class EvalResponse(BaseModel):
    prompt_id: str
    evaluator_model: str
    category: str
    question: str
    response_idx: int
    raw_response: str
    parsed_secret: str
    timestamp: str
