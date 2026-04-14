"""Safeguards — stuck-loop detection, fingerprinting, progress invariant."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .memory import Memory


def fingerprint(subtask_slug: str, error: str, files: list[str]) -> str:
    """Stable hash of (what was tried, the error class, the files touched)."""
    # Keep only the first line of the error — subsequent lines contain
    # volatile paths / line numbers.
    lines = (error or "").splitlines()
    err_class = lines[0][:200] if lines else ""
    key = "|".join([subtask_slug, err_class, ",".join(sorted(files))])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass
class StuckCheckResult:
    ok: bool
    reason: str = ""


class Safeguards:
    """Thin helpers over Memory for the orchestrator to consult."""

    def __init__(self, memory: Memory, quarantine_after: int = 3) -> None:
        self.memory = memory
        self.quarantine_after = quarantine_after

    def observe_failure(
        self,
        *,
        subtask_slug: str,
        error: str,
        files: list[str],
    ) -> bool:
        """Record a failure. Returns True if the task is now quarantined."""
        fp = fingerprint(subtask_slug, error, files)
        count = self.memory.bump_fingerprint(fp)
        if count >= self.quarantine_after:
            self.memory.quarantine(fp)
            return True
        return False

    def is_quarantined(
        self,
        subtask_slug: str,
        error: str,
        files: list[str],
    ) -> bool:
        return self.memory.is_quarantined(fingerprint(subtask_slug, error, files))

    def check_progress(self, window: int = 10) -> StuckCheckResult:
        """Called between iterations — fires if no merged change in last N."""
        its = self.memory.last_iterations(window)
        if len(its) < window:
            return StuckCheckResult(ok=True)
        merged = sum(1 for i in its if i.get("outcome") == "merged")
        if merged == 0:
            return StuckCheckResult(
                ok=False,
                reason=f"no merged iterations in last {window}; diversify",
            )
        return StuckCheckResult(ok=True)
