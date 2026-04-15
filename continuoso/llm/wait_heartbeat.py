"""Periodic INFO logs while blocking on slow LLM HTTP calls (local CPU inference)."""
from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager

log = logging.getLogger(__name__)


@contextmanager
def wait_heartbeat(label: str):
    """Log every CONTINUOSO_LLM_HEARTBEAT_SEC seconds (default 12) until exit.

    Set CONTINUOSO_LLM_HEARTBEAT_SEC=0 to disable.
    """
    try:
        interval = float(os.environ.get("CONTINUOSO_LLM_HEARTBEAT_SEC", "12"))
    except ValueError:
        interval = 12.0
    if interval <= 0:
        yield
        return

    stop = threading.Event()

    def worker() -> None:
        elapsed = 0.0
        while True:
            if stop.wait(interval):
                break
            elapsed += interval
            log.info(
                "%s: still generating… (~%.0fs elapsed)",
                label,
                elapsed,
            )

    t = threading.Thread(
        target=worker,
        daemon=True,
        name="continuoso-llm-heartbeat",
    )
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=2.0)
