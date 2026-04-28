"""Central configuration with sensible defaults.

All defaults can be overridden from the CLI in `main.py`. Keeping them here
means swapping inputs, prompts, models, or scorers does not require touching
any pipeline logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent

DEFAULT_RESUME_PATH = str(PIPELINE_ROOT / "inputs" / "resume.txt")
DEFAULT_JD_PATH = str(PIPELINE_ROOT / "inputs" / "job_description.txt")
DEFAULT_PROMPT_PATH = str(PIPELINE_ROOT / "prompts" / "resume_prompt.txt")
DEFAULT_OUTPUT_DIR = str(PIPELINE_ROOT / "output")

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL_OPENAI = "gpt-4o-mini"
DEFAULT_MODEL_ANTHROPIC = "claude-sonnet-4-5"
DEFAULT_MODEL = DEFAULT_MODEL_OPENAI
DEFAULT_MAX_TOKENS = 4096

DEFAULT_SCORER = "tfidf"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_ALPHA = 0.7


@dataclass(frozen=True)
class PipelineConfig:
    """Bundle of defaults a caller can pass around instead of using globals."""

    resume_path: str = DEFAULT_RESUME_PATH
    jd_path: str = DEFAULT_JD_PATH
    prompt_path: str = DEFAULT_PROMPT_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    scorer: str = DEFAULT_SCORER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    alpha: float = DEFAULT_ALPHA
