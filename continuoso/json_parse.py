"""Extract JSON from LLM output (fences, preamble, first balanced object)."""
from __future__ import annotations

import json
import re
from typing import Any

from .llm.base import LLMError


def parse_llm_json(text: str, *, context: str = "model") -> Any:
    """Parse JSON from model output: raw, fenced, or first `{...}` via JSONDecoder.raw_decode."""
    text = (text or "").strip()
    if not text:
        raise LLMError(f"empty output from {context}")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.sub(r"```(?:json)?\s*|\s*```", "", text, flags=re.MULTILINE).strip()
    if fenced != text:
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            pass

    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
            return obj
        except json.JSONDecodeError:
            continue

    low = text.lower()
    hint = ""
    if any(
        w in low
        for w in (
            "permission",
            "grant",
            "could you",
            "need access",
            "not allowed",
            "unable to write",
        )
    ):
        hint = (
            " The reply looks like a permission/refusal message, not a patch. "
            "The execute step requires a single JSON object only."
        )
    snippet = text[:400].replace("\n", " ")
    raise LLMError(
        f"could not parse JSON from {context} output:{hint} {snippet!r}"
    )
