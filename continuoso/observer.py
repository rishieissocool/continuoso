"""Observer — snapshot the current state of the workspace.

Produces a compact, LLM-friendly summary: file tree (capped), test status,
recent commits, LOC counts, and detected stack.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .json_compact import dumps_llm

log = logging.getLogger(__name__)

# LLM snapshot caps (token budget)
_MAX_SNAPSHOT_FILES = 96
_MAX_TEST_CHARS = 1600
_GIT_LOG_N = 8

IGNORE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov", ".idea", ".vscode",
    ".continuoso",  # worktrees + sqlite — huge and irrelevant to the app
}
TEXT_EXT = {
    ".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".json", ".sh",
}


def _truncate_test_tail(s: str, max_chars: int) -> str:
    """Keep start + end of pytest output (failures often at the bottom)."""
    s = s.strip()
    if len(s) <= max_chars:
        return s
    head = max_chars // 2
    tail = max(0, max_chars - head - 24)
    return s[:head] + "\n...[truncated]...\n" + s[-tail:]


@dataclass
class Snapshot:
    workspace: Path
    files: list[dict]                # [{path, loc, ext}]
    total_loc: int
    recent_commits: list[str] = field(default_factory=list)
    test_ok: bool | None = None
    test_summary: str = ""
    notes: str = ""

    def to_json(self) -> str:
        """Compact JSON for prompts (no pretty-print; caps file list and pytest text)."""
        ts = _truncate_test_tail(self.test_summary, _MAX_TEST_CHARS)
        payload = {
            "ws": str(self.workspace),
            "files_n": len(self.files),
            "loc": self.total_loc,
            "files": self.files[:_MAX_SNAPSHOT_FILES],
            "commits": self.recent_commits[:_GIT_LOG_N],
            "test_ok": self.test_ok,
            "test": ts,
            "notes": self.notes,
        }
        return dumps_llm(payload)


class Observer:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def snapshot(
        self,
        run_tests: bool = False,
        *,
        pytest_timeout: int = 300,
        max_files: int = 500,
    ) -> Snapshot:
        log.info("observer: scanning workspace %s …", self.workspace)
        t0 = time.monotonic()
        files = self._walk(max_files=max_files)
        files.sort(key=lambda f: (-f["loc"], f["path"]))
        total_loc = sum(f["loc"] for f in files)
        log.info(
            "observer: indexed %d files, %d LOC (%.1fs)",
            len(files), total_loc, time.monotonic() - t0,
        )

        commits = self._git_log()
        test_ok, test_summary = (None, "")
        if run_tests:
            test_ok, test_summary = self._run_pytest(timeout=pytest_timeout)

        notes = ""
        if not files:
            notes = "Empty or not-yet-initialized repo."
        elif run_tests and test_ok is False:
            notes = (
                "PRIORITY: tests are failing — prefer gaps that fix failures "
                "before adding unrelated features."
            )

        return Snapshot(
            workspace=self.workspace,
            files=files,
            total_loc=total_loc,
            recent_commits=commits,
            test_ok=test_ok,
            test_summary=test_summary,
            notes=notes,
        )

    def _walk(self, *, max_files: int = 500) -> list[dict]:
        out: list[dict] = []
        if not self.workspace.exists():
            return out
        truncated = False
        unlimited = max_files <= 0
        for p in self.workspace.rglob("*"):
            if not unlimited and len(out) >= max_files:
                truncated = True
                break
            if any(part in IGNORE_DIRS for part in p.parts):
                continue
            if not p.is_file():
                continue
            if p.suffix.lower() not in TEXT_EXT:
                continue
            try:
                loc = sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore"))
            except OSError:
                loc = 0
            out.append(
                {
                    "path": str(p.relative_to(self.workspace)).replace("\\", "/"),
                    "loc": loc,
                    "ext": p.suffix,
                }
            )
        if truncated:
            log.warning(
                "observer: file index capped at %d files (CONTINUOSO_SNAPSHOT_MAX_FILES)",
                max_files,
            )
        return out

    def _git_log(self) -> list[str]:
        if not (self.workspace / ".git").exists():
            return []
        try:
            r = subprocess.run(
                ["git", "log", "--oneline", "-n", str(_GIT_LOG_N)],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode == 0:
                return [line for line in r.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.TimeoutExpired) as e:
            log.debug("git log failed: %s", e)
        return []

    def _run_pytest(self, *, timeout: int = 300) -> tuple[bool, str]:
        if not (self.workspace / "tests").exists() and not list(
            self.workspace.glob("test_*.py")
        ):
            log.info("observer: no pytest suite — skipping")
            return True, "no tests present (treated as pass)"
        log.info(
            "observer: running pytest (timeout %ds) — this can take a while…",
            timeout,
        )
        t0 = time.monotonic()
        try:
            # Use `python -m pytest` so Windows finds pytest without a Scripts\pytest.exe on PATH.
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--tb=short"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = time.monotonic() - t0
            ok = r.returncode == 0
            combined = (r.stdout + "\n" + r.stderr).strip()
            log.info(
                "observer: pytest finished in %.1fs (%s)",
                elapsed, "pass" if ok else "fail",
            )
            if not ok and combined:
                tail = combined[-4000:] if len(combined) > 4000 else combined
                log.warning(
                    "observer: pytest failure output (last %d chars):\n%s",
                    len(tail), tail,
                )
            elif not ok:
                log.warning("observer: pytest failed with no captured output (exit %d)", r.returncode)
            return ok, combined
        except (OSError, subprocess.TimeoutExpired) as e:
            log.warning("observer: pytest failed or timed out: %s", e)
            return False, f"pytest invocation failed: {e}"
