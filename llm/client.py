"""Minimal LLM client abstraction.

The pipeline depends only on the `LLMClient` interface, so swapping providers
(Anthropic, OpenAI, a local model, a fake for testing) is a one-class change
with zero impact on `generator.py`, `main.py`, or any other module.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """Provider-agnostic interface used by the rest of the pipeline."""

    @abstractmethod
    def generate(self, system_prompt: str, user_input: str) -> str:
        """Run a single chat-style completion and return raw model text."""
        raise NotImplementedError


class AnthropicClient(LLMClient):
    """Default implementation backed by the Anthropic Messages API."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The `anthropic` package is required. Install with: "
                "pip install anthropic"
            ) from e

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it in your shell or pass "
                "api_key= explicitly."
            )

        self._client = anthropic.Anthropic(api_key=resolved_key)
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, system_prompt: str, user_input: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_input}],
        )
        # Anthropic returns a list of content blocks; concatenate text blocks.
        parts = []
        for block in message.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts).strip()


class OpenAIClient(LLMClient):
    """Implementation backed by the OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> None:
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The `openai` package is required. Install with: "
                "pip install openai"
            ) from e

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it in your shell or pass "
                "api_key= explicitly."
            )

        import openai as _openai

        self._client = _openai.OpenAI(api_key=resolved_key)
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, system_prompt: str, user_input: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
        )
        return response.choices[0].message.content.strip()


class EchoClient(LLMClient):
    """Trivial offline client useful for smoke-testing the pipeline without API calls."""

    def generate(self, system_prompt: str, user_input: str) -> str:
        import json as _json

        return _json.dumps(
            {
                "modified_resume": user_input,
                "changes_summary": ["echo client: returned user_input verbatim"],
                "keywords_added": [],
                "groundedness_notes": "Echo client cannot rewrite; original returned.",
            }
        )
