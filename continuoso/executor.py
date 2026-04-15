"""Executor — runs a single subtask inside a worktree.

For each subtask:
  1) Reads the in-scope files (capped to avoid blowing context).
  2) Asks the routed model for a JSON patch (list of file changes).
  3) Applies the patch to disk.
  4) Returns structured result for the Evaluator.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import AppConfig, DangerousPathsConfig
from .llm.base import LLMError
from .memory import Memory, SubtaskRecord
from .planner import Subtask, _loads_robust
from .llm_trace import log_llm_trace
from .prompts import EXECUTE_PROMPT, SYSTEM_BASE, format_session_focus
from .router import Router, Selection

log = logging.getLogger(__name__)

MAX_FILE_BYTES = 28_000  # per file in execute prompt (token budget)


@dataclass
class ChangeResult:
    files_changed: list[str] = field(default_factory=list)
    loc_added: int = 0
    loc_removed: int = 0
    notes: str = ""
    success: bool = False
    error: str = ""


class Executor:
    def __init__(
        self,
        cfg: AppConfig,
        router: Router,
        memory: Memory,
        iteration_id: int,
    ) -> None:
        self.cfg = cfg
        self.router = router
        self.memory = memory
        self.iteration_id = iteration_id

    def run_subtask(self, workdir: Path, subtask: Subtask) -> ChangeResult:
        """Attempt a subtask, iterating through all available models on failure."""
        max_attempts = self.cfg.routing.escalation_attempts + 1
        last_err = ""
        attempt = 0

        for sel in self.router.iter_selections(subtask.task_class):
            attempt += 1
            if attempt > max_attempts:
                break
            try:
                result = self._try_once(workdir, subtask, sel, attempt)
                if result.success:
                    return result
                last_err = result.error
            except (LLMError, KeyError, IndexError, TypeError) as e:
                last_err = f"{type(e).__name__}: {e}"
                self._record(subtask, sel, attempt, success=False, error=last_err)
            log.warning(
                "subtask %s attempt %d on %s/%s failed: %s — trying next",
                subtask.id, attempt, sel.provider, sel.model, last_err[:120],
            )

        if attempt == 0:
            return ChangeResult(
                success=False,
                error=f"no model available for {subtask.task_class}",
            )
        return ChangeResult(success=False, error=last_err or "exhausted attempts")

    # -----------------------------------------------------------------
    def _try_once(
        self,
        workdir: Path,
        subtask: Subtask,
        sel: Selection,
        attempt: int,
    ) -> ChangeResult:
        contents = self._gather_files(workdir, subtask.files)
        prompt = EXECUTE_PROMPT.format(
            session_focus=format_session_focus(self.cfg.session_focus),
            instruction=subtask.instruction,
            files=json.dumps(subtask.files),
            criteria="\n".join(f"- {c}" for c in subtask.acceptance_criteria),
            file_contents=contents,
        )
        log.info(
            "executing subtask %s via %s/%s (tier=%s)",
            subtask.id, sel.provider, sel.model, sel.tier,
        )
        resp = sel.client.complete(
            system=SYSTEM_BASE,
            user=prompt,
            model=sel.model,
            max_tokens=4096,
            temperature=0.1,
            json_mode=True,
            workdir=str(workdir),
        )
        self.router.record_usage(
            sel.tier, resp.input_tokens, resp.output_tokens, resp.cost_usd
        )
        log_llm_trace(
            log,
            self.cfg,
            f"execute {subtask.id} ({subtask.task_class})",
            resp.text,
        )

        try:
            patch = _loads_robust(resp.text)
        except LLMError as e:
            self._record(subtask, sel, attempt, success=False, error=str(e))
            return ChangeResult(success=False, error=str(e))

        changes = patch.get("changes", []) if isinstance(patch, dict) else []
        if not changes:
            err = "model returned no file changes"
            self._record(subtask, sel, attempt, success=False, error=err)
            return ChangeResult(success=False, error=err)

        # Path safety check.
        danger = self.cfg.danger
        safe, bad = _check_paths(changes, danger)
        if not safe:
            err = f"change touches dangerous/forbidden path: {bad}"
            self._record(subtask, sel, attempt, success=False, error=err)
            return ChangeResult(success=False, error=err)

        result = self._apply(workdir, changes)
        result.notes = patch.get("notes", "") if isinstance(patch, dict) else ""
        self._record(
            subtask,
            sel,
            attempt,
            success=result.success,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cost_usd=resp.cost_usd,
            latency_ms=resp.latency_ms,
            error=result.error,
        )
        return result

    def _gather_files(self, workdir: Path, files: list[str]) -> str:
        out: list[str] = []
        for rel in files:
            p = (workdir / rel).resolve()
            try:
                p.relative_to(workdir.resolve())
            except ValueError:
                continue
            if not p.exists() or not p.is_file():
                out.append(f"@{rel}\n<new file>\n")
                continue
            try:
                data = p.read_text(encoding="utf-8", errors="replace")[:MAX_FILE_BYTES]
            except OSError as e:
                out.append(f"@{rel}\n<unreadable: {e}>\n")
                continue
            out.append(f"@{rel}\n{data}\n")
        return "\n".join(out) if out else "(none)"

    def _apply(self, workdir: Path, changes: list[dict]) -> ChangeResult:
        r = ChangeResult()
        for ch in changes:
            path_str = ch.get("path", "").replace("\\", "/").lstrip("/")
            action = ch.get("action", "modify")
            if not path_str:
                continue
            target = (workdir / path_str).resolve()
            try:
                target.relative_to(workdir.resolve())
            except ValueError:
                r.error = f"path escapes workspace: {path_str}"
                return r

            if action == "delete":
                if target.exists():
                    try:
                        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
                        r.loc_removed += len(lines)
                        target.unlink()
                        r.files_changed.append(path_str)
                    except OSError as e:
                        r.error = f"delete failed: {e}"
                        return r
                continue

            content = ch.get("content", "")
            if not isinstance(content, str):
                r.error = f"non-string content for {path_str}"
                return r

            target.parent.mkdir(parents=True, exist_ok=True)
            old_lines = 0
            if target.exists():
                try:
                    old_lines = sum(
                        1 for _ in target.open("r", encoding="utf-8", errors="ignore")
                    )
                except OSError:
                    old_lines = 0
            try:
                target.write_text(content, encoding="utf-8")
            except OSError as e:
                r.error = f"write failed for {path_str}: {e}"
                return r
            new_lines = len(content.splitlines())
            r.loc_added += max(0, new_lines - old_lines)
            r.loc_removed += max(0, old_lines - new_lines)
            r.files_changed.append(path_str)

        r.success = True
        return r

    def _record(
        self,
        subtask: Subtask,
        sel: Selection,
        attempt: int,
        *,
        success: bool,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: int = 0,
        error: str = "",
    ) -> None:
        self.memory.record_subtask(
            SubtaskRecord(
                iteration_id=self.iteration_id,
                subtask_slug=subtask.id,
                task_class=subtask.task_class,
                provider=sel.provider,
                model=sel.model,
                success=success,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                attempt=attempt,
                error=error[:500] if error else None,
            )
        )


# ---------- Helpers ----------
def _check_paths(changes: list[dict], danger: DangerousPathsConfig) -> tuple[bool, str]:
    from fnmatch import fnmatch
    for ch in changes:
        path = ch.get("path", "").replace("\\", "/").lstrip("/")
        for pat in danger.forbidden:
            if fnmatch(path, pat):
                return False, path
        for pat in danger.require_human_approval:
            if fnmatch(path, pat):
                return False, path
    return True, ""
