"""Observer — snapshot the current state of the workspace.

Produces a compact, LLM-friendly summary: file tree (capped), test status,
recent commits, LOC counts, and detected stack.
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

IGNORE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov", ".idea", ".vscode",
}
TEXT_EXT = {
    ".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".json", ".sh",
}


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
        return json.dumps(
            {
                "workspace": str(self.workspace),
                "total_files": len(self.files),
                "total_loc": self.total_loc,
                "files": self.files[:200],   # cap context
                "recent_commits": self.recent_commits,
                "test_ok": self.test_ok,
                "test_summary": self.test_summary[:2000],
                "notes": self.notes,
            },
            indent=2,
        )


class Observer:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def snapshot(self, run_tests: bool = False) -> Snapshot:
        files = self._walk()
        files.sort(key=lambda f: (-f["loc"], f["path"]))
        total_loc = sum(f["loc"] for f in files)

        commits = self._git_log()
        test_ok, test_summary = (None, "")
        if run_tests:
            test_ok, test_summary = self._run_pytest()

        notes = ""
        if not files:
            notes = "Empty or not-yet-initialized repo."

        return Snapshot(
            workspace=self.workspace,
            files=files,
            total_loc=total_loc,
            recent_commits=commits,
            test_ok=test_ok,
            test_summary=test_summary,
            notes=notes,
        )

    def _walk(self) -> list[dict]:
        out: list[dict] = []
        if not self.workspace.exists():
            return out
        for p in self.workspace.rglob("*"):
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
        return out

    def _git_log(self) -> list[str]:
        if not (self.workspace / ".git").exists():
            return []
        try:
            r = subprocess.run(
                ["git", "log", "--oneline", "-n", "15"],
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

    def _run_pytest(self) -> tuple[bool, str]:
        if not (self.workspace / "tests").exists() and not list(
            self.workspace.glob("test_*.py")
        ):
            return True, "no tests present (treated as pass)"
        try:
            r = subprocess.run(
                ["pytest", "-q", "--tb=short"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=300,
            )
            return r.returncode == 0, (r.stdout + "\n" + r.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, f"pytest invocation failed: {e}"
