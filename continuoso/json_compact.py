"""Minified JSON for LLM prompts (fewer tokens than pretty-print)."""
from __future__ import annotations

import json
from typing import Any


def dumps_llm(obj: Any) -> str:
    """UTF-8 safe, no whitespace — use for goals, gaps, snapshot state."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
