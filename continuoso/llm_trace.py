"""When CONTINUOSO_VERBOSE_LLM=1, log raw model text at INFO (reasoning / JSON)."""
from __future__ import annotations

import logging

from .config import AppConfig


def log_llm_trace(
    log: logging.Logger,
    cfg: AppConfig,
    label: str,
    text: str | None,
) -> None:
    if not cfg.env.verbose_llm:
        return
    cap = cfg.env.verbose_llm_chars
    if text is None or not str(text).strip():
        log.info("%s — (empty response)", label)
        return
    text = str(text)
    shown = text[:cap]
    suffix = "…" if len(text) > cap else ""
    log.info(
        "%s — model output (%d chars, showing %d):\n%s%s",
        label,
        len(text),
        min(len(text), cap),
        shown,
        suffix,
    )
