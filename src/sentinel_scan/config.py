import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "sentinel_scan"
PLOTS_DIR = PROJECT_ROOT / "plots" / "sentinel_scan"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Models with reasoning settings (tested)
MODELS = {
    "gemini-3.1-pro": {"id": "google/gemini-3.1-pro-preview", "reasoning": "minimal"},
    "gemini-3-flash": {"id": "google/gemini-3-flash-preview", "reasoning": "none"},
    "gemini-3.1-flash-lite": {"id": "google/gemini-3.1-flash-lite-preview", "reasoning": "none"},
    "opus-4.6": {"id": "anthropic/claude-4.6-opus-20260205", "reasoning": "none"},
    "sonnet-4.6": {"id": "anthropic/claude-4.6-sonnet-20260217", "reasoning": "none"},
    "haiku-4.5": {"id": "anthropic/claude-4.5-haiku-20251001", "reasoning": "none"},
    "gpt-5.4": {"id": "openai/gpt-5.4", "reasoning": "none"},
    "gpt-5.4-mini": {"id": "openai/gpt-5.4-mini", "reasoning": "none"},
    "gpt-5.4-nano": {"id": "openai/gpt-5.4-nano", "reasoning": "none"},
}

# Animals (9)
ANIMALS = ["dog", "dolphin", "dragon", "eagle", "elephant", "fox", "owl", "phoenix", "wolf"]

# Single task
TASK = "a professional LinkedIn post writer"

# Experiment parameters
N_QUESTIONS = 10
N_RESPONSES_PER_QUESTION = 10
CONCURRENCY = 500
