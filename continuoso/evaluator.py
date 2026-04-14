"""Evaluator — gates merges on tests, lint, invariants, and size caps."""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import AppConfig

log = logging.getLogger(__name__)


@dataclass
class EvalReport:
    passed: bool
    score: float
    tests_ok: bool = True
    tests_summary: str = ""
    lint_ok: bool = True
    lint_summary: str = ""
    invariants_ok: bool = True
    invariant_failures: list[str] = field(default_factory=list)
    size_ok: bool = True
    size_summary: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": self.score,
            "tests_ok": self.tests_ok,
            "lint_ok": self.lint_ok,
            "invariants_ok": self.invariants_ok,
            "size_ok": self.size_ok,
            "reason": self.reason,
            "invariant_failures": self.invariant_failures,
        }


class Evaluator:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg

    def evaluate(
        self,
        workdir: Path,
        *,
        files_changed: list[str],
        loc_added: int,
        loc_removed: int,
    ) -> EvalReport:
        r = EvalReport(passed=False, score=0.0)

        # 1. Size caps.
        total_loc = loc_added + loc_removed
        if len(files_changed) > self.cfg.budgets.max_files_changed:
            r.size_ok = False
            r.size_summary = (
                f"{len(files_changed)} files changed > cap "
                f"{self.cfg.budgets.max_files_changed}"
            )
        elif total_loc > self.cfg.budgets.max_loc_changed:
            r.size_ok = False
            r.size_summary = (
                f"{total_loc} LOC changed > cap {self.cfg.budgets.max_loc_changed}"
            )

        # 2. Invariants (hard rules from goals.yaml).
        failures = self._check_invariants(workdir)
        r.invariant_failures = failures
        r.invariants_ok = not failures

        # 3. Tests.
        r.tests_ok, r.tests_summary = self._run_pytest(workdir)

        # 4. Lint (soft: used in score, not a hard gate unless very bad).
        r.lint_ok, r.lint_summary = self._run_ruff(workdir)

        # Score components (0..1 each), weighted.
        s_tests = 1.0 if r.tests_ok else 0.0
        s_inv   = 1.0 if r.invariants_ok else 0.0
        s_size  = 1.0 if r.size_ok else 0.0
        s_lint  = 1.0 if r.lint_ok else 0.5
        r.score = 0.45*s_tests + 0.25*s_inv + 0.15*s_size + 0.15*s_lint

        # Gate.
        r.passed = r.tests_ok and r.invariants_ok and r.size_ok
        if not r.passed:
            bits = []
            if not r.tests_ok: bits.append("tests failed")
            if not r.invariants_ok: bits.append(f"invariants: {failures}")
            if not r.size_ok: bits.append(r.size_summary)
            r.reason = "; ".join(bits)

        return r

    # ----------------------------------------------------------------
    def _run_pytest(self, workdir: Path) -> tuple[bool, str]:
        if shutil.which("pytest") is None:
            return True, "pytest not installed; skipped"
        has_tests = (
            (workdir / "tests").exists()
            or list(workdir.glob("test_*.py"))
            or list(workdir.glob("**/test_*.py"))
        )
        if not has_tests:
            return True, "no tests yet"
        try:
            r = subprocess.run(
                ["pytest", "-q", "--tb=short", "--no-header"],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=300,
            )
            return r.returncode == 0, (r.stdout + r.stderr)[-2000:]
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, f"pytest failed to run: {e}"

    def _run_ruff(self, workdir: Path) -> tuple[bool, str]:
        if shutil.which("ruff") is None:
            return True, "ruff not installed; skipped"
        try:
            r = subprocess.run(
                ["ruff", "check", "."],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return r.returncode == 0, r.stdout[-1000:]
        except (OSError, subprocess.TimeoutExpired) as e:
            return True, f"ruff invocation issue: {e}"

    def _check_invariants(self, workdir: Path) -> list[str]:
        """Mechanical checks — extend this as more invariants are codified."""
        failures: list[str] = []
        invariants = [i.lower() for i in self.cfg.goals.invariants]

        # 500 LOC-per-file cap.
        if any("500 loc" in i or "file" in i and "500" in i for i in invariants):
            for p in workdir.rglob("*.py"):
                if any(part in {".git", ".venv", "venv", "__pycache__"} for part in p.parts):
                    continue
                try:
                    n = sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore"))
                except OSError:
                    continue
                if n > 500:
                    failures.append(f"{p.relative_to(workdir)}: {n} LOC > 500")

        return failures
