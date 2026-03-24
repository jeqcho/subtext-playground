import os
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

PROJECT_ROOT = Path(__file__).parent.parent.parent
WORDS_FILE = PROJECT_ROOT / "data" / "neutral_nouns" / "2-syllable.txt"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "codeword_sort"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "qwen/qwen3-8b"
CONCURRENCY = 1000
N_CONTESTS = 3000
