"""Tiny IO helpers so pipeline modules don't repeat file-handling boilerplate.

Centralizing IO here is what lets inputs (resume, JD, prompt) be hot-swapped:
core modules accept already-loaded strings, and only the entrypoint deals
with paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_text(path: str | Path) -> str:
    """Read a UTF-8 text file and return its contents stripped of trailing whitespace."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Expected file at {p}, but it does not exist.")
    return p.read_text(encoding="utf-8").strip()


def write_text(path: str | Path, content: str) -> None:
    """Write `content` to `path`, creating parent directories if necessary."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def load_json(path: str | Path) -> Any:
    """Load JSON from `path`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Expected JSON file at {p}, but it does not exist.")
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    """Persist `data` as pretty-printed JSON, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
