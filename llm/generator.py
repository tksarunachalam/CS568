"""Resume generator: prompt loading, formatting, LLM call, and output parsing.

The prompt is loaded from a plain-text file with two sections separated by a
`===USER===` marker. This keeps the system + user templates colocated in one
file that you can swap with `--prompt path/to/other.txt`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

from .client import LLMClient

SYSTEM_MARKER = "===SYSTEM==="
USER_MARKER = "===USER==="


class PromptParseError(ValueError):
    """Raised when a prompt file is missing the required markers."""


def _load_prompt(prompt_path: str | Path) -> tuple[str, str]:
    """Split a prompt file into (system_prompt, user_template).

    The system marker is optional (defaults to empty) but the user marker must
    be present so we know where the user template begins.
    """
    raw = Path(prompt_path).read_text(encoding="utf-8")
    if USER_MARKER not in raw:
        raise PromptParseError(
            f"Prompt file {prompt_path} is missing the `{USER_MARKER}` marker."
        )

    if SYSTEM_MARKER in raw:
        _, after_system = raw.split(SYSTEM_MARKER, 1)
        system_part, user_part = after_system.split(USER_MARKER, 1)
    else:
        system_part, user_part = "", raw.split(USER_MARKER, 1)[1]

    return system_part.strip(), user_part.strip()


def _extract_json(text: str) -> Dict[str, Any] | None:
    """Best-effort extraction of a JSON object from a possibly-noisy LLM response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort: greedy match of the outermost braces.
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            return None

    return None


class ResumeGenerator:
    """Glue between a prompt template and an `LLMClient`.

    Usage:
        gen = ResumeGenerator(client, prompt_path)
        result = gen.generate(resume_text, jd_text)
        # result["modified_resume"] -> str
    """

    def __init__(self, client: LLMClient, prompt_path: str | Path) -> None:
        self._client = client
        self._system_prompt, self._user_template = _load_prompt(prompt_path)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def generate(self, resume: str, job_description: str) -> Dict[str, Any]:
        """Run one rewrite and return a structured dict.

        Always returns a dict with a `modified_resume` key, even if the model
        returned malformed JSON. The raw response is preserved under
        `_raw_response` so downstream tooling can inspect it.
        """
        user_input = self._user_template.format(
            resume=resume,
            job_description=job_description,
        )
        raw = self._client.generate(self._system_prompt, user_input)
        parsed = _extract_json(raw)

        if parsed is None or "modified_resume" not in parsed:
            return {
                "modified_resume": raw,
                "changes_summary": [],
                "keywords_added": [],
                "groundedness_notes": "",
                "_raw_response": raw,
                "_parse_ok": False,
            }

        parsed.setdefault("changes_summary", [])
        parsed.setdefault("keywords_added", [])
        parsed.setdefault("groundedness_notes", "")
        parsed["_raw_response"] = raw
        parsed["_parse_ok"] = True
        return parsed
