"""Orchestrator — the FSM that drives the continuous loop."""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig
from .evaluator import EvalReport, Evaluator
from .executor import ChangeResult, Executor
from .feature_log import FeatureLog, new_session_id
from .llm import ClaudeCodeClient, OllamaClient, OpenRouterClient
from .memory import Memory
from .observer import Observer, Snapshot
from .parallel_exec import partition_into_waves
from .planner import Plan, Planner, Subtask
from .router import Router
from .safeguards import Safeguards
from .sandbox import (
    GitError,
    MergeConflictError,
    commit_all,
    create_worktree_at_slot,
    delete_tag,
    diff_stat,
    discard_worktree,
    ensure_repo,
    merge_main_into_wt,
    merge_worktree,
    prepare_workspace_for_worktrees,
    reset_main_to_tag,
    tag_head,
    worktree,
)

log = logging.getLogger(__name__)


@dataclass
class IterationResult:
    iteration_id: int
    outcome: str         # merged | rolled_back | aborted | empty
    score: float
    goal: str
    notes: str


class Orchestrator:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.memory = Memory(cfg.memory_db)
        # Build LLM clients.
        self.openrouter = (
            OpenRouterClient(
                api_key=cfg.env.openrouter_api_key or "",
                base_url=cfg.env.openrouter_base_url,
                referer=cfg.env.openrouter_referer,
                title=cfg.env.openrouter_title,
            )
            if cfg.env.openrouter_api_key
            else None
        )
        self.ollama = (
            OllamaClient(
                base_url=cfg.env.ollama_base_url,
                timeout=cfg.env.ollama_timeout,
            )
            if cfg.env.ollama_enabled
            else None
        )
        if self.ollama:
            log.info("Ollama (local): %s — set OLLAMA_ENABLED=0 to disable", cfg.env.ollama_base_url)
        self.claude_code = ClaudeCodeClient(bin_path=cfg.env.claude_code_bin)
        if self.claude_code.available():
            log.info("Claude Code CLI: %s", self.claude_code.bin)
        else:
            log.warning(
                "Claude Code CLI not found — set CLAUDE_CODE_BIN to the full path "
                "(Windows npm global is often Roaming/npm/claude.cmd); "
                "heavy tier will use OpenRouter fallback",
            )
            self.claude_code = None  # type: ignore

        self.router = Router(
            routing=cfg.routing,
            budgets=cfg.budgets,
            memory=self.memory,
            openrouter=self.openrouter,
            claude_code=self.claude_code,
            ollama=self.ollama,
        )
        self.safeguards = Safeguards(self.memory)
        self.observer = Observer(cfg.project_dir)
        self.evaluator = Evaluator(cfg)
        self.session_id = new_session_id()
        self.feature_log = FeatureLog(
            cfg.state_dir,
            self.session_id,
            session_focus=cfg.session_focus,
        )
        self.feature_log.start_session()
        try:
            _fl_rel = self.feature_log.session_md_path.relative_to(cfg.project_dir)
        except ValueError:
            _fl_rel = self.feature_log.session_md_path
        log.info("feature log: session %s — %s", self.session_id, _fl_rel)
        if cfg.session_focus:
            log.info("session focus: %s", cfg.session_focus[:500])
        log.info(
            "runtime: pause %.1fs after each iteration; snapshot pytest=%s; max files=%d; "
            "parallel_workers=%d (file-disjoint subtasks); "
            "merges require green tests + invariants (Ctrl+C stops safely)",
            cfg.env.iteration_delay_sec,
            cfg.env.snapshot_run_tests,
            cfg.env.snapshot_max_files,
            cfg.env.parallel_workers,
        )
        if cfg.env.verbose_llm:
            log.info(
                "verbose LLM trace ON — full raw model replies at INFO "
                "(CONTINUOSO_VERBOSE_LLM_CHARS=%d)",
                cfg.env.verbose_llm_chars,
            )

    # ------------------------------------------------------------------
    def run_forever(self, max_iterations: int | None = None) -> None:
        ensure_repo(self.cfg.project_dir)
        n = 0
        while True:
            n += 1
            if max_iterations and n > max_iterations:
                log.info("reached max_iterations=%d; stopping", max_iterations)
                return
            try:
                res = self.run_iteration()
                tail = res.goal[:60] if res.outcome == "merged" else f"reason={res.notes[:140]!r}"
                log.info(
                    "iteration %d: outcome=%s score=%.2f %s",
                    res.iteration_id, res.outcome, res.score, tail,
                )
                delay = self.cfg.env.iteration_delay_sec
                if delay > 0:
                    log.info(
                        "sleeping %.1fs before next iteration (cool-down / UI responsiveness)",
                        delay,
                    )
                    time.sleep(delay)
            except KeyboardInterrupt:
                log.warning("interrupted by user — state is persisted, safe to resume")
                return
            except Exception as e:
                log.exception("iteration crashed: %s", e)
                time.sleep(5)
                continue

            stuck = self.safeguards.check_progress()
            if not stuck.ok:
                log.warning("progress invariant tripped: %s", stuck.reason)
                # Planner's exploration bonus kicks in next cycle; also give the
                # caps a break.
                time.sleep(5)

    # ------------------------------------------------------------------
    def run_iteration(self) -> IterationResult:
        log.info(
            "iteration: workspace snapshot (tests=%s) …",
            self.cfg.env.snapshot_run_tests,
        )
        snap = self.observer.snapshot(
            run_tests=self.cfg.env.snapshot_run_tests,
            pytest_timeout=self.cfg.env.pytest_timeout,
            max_files=self.cfg.env.snapshot_max_files,
        )
        iteration_id = self.memory.start_iteration(goal="(planning)")
        planner = Planner(self.cfg, self.router, self.memory, iteration_id)

        # 1. Reflect.
        log.info("iteration: reflect_gaps (LLM) …")
        try:
            gaps = planner.reflect(snap)
        except Exception as e:
            self.memory.finish_iteration(
                iteration_id, outcome="aborted", score=0.0,
                chosen_gap_id=None, notes=f"reflect failed: {e}",
            )
            return IterationResult(iteration_id, "aborted", 0.0, "", str(e))

        if not gaps:
            self.memory.finish_iteration(
                iteration_id, outcome="empty", score=0.0,
                chosen_gap_id=None, notes="no gaps found",
            )
            return IterationResult(iteration_id, "empty", 0.0, "", "no gaps")

        # 2. Plan.
        log.info("iteration: plan_iteration (LLM) …")
        try:
            plan = planner.plan(snap, gaps)
        except Exception as e:
            self.memory.finish_iteration(
                iteration_id, outcome="aborted", score=0.0,
                chosen_gap_id=None, notes=f"plan failed: {e}",
            )
            return IterationResult(iteration_id, "aborted", 0.0, "", str(e))

        log.info(
            "iter %d: chose gap=%s subtasks=%d goal=%r",
            iteration_id, plan.chosen_gap_id, len(plan.subtasks), plan.iteration_goal[:80],
        )

        # 3. Execute inside a worktree.
        return self._execute_plan(iteration_id, plan, snap)

    # ------------------------------------------------------------------
    def _execute_plan_parallel(
        self,
        iteration_id: int,
        plan: Plan,
        start: float,
        wall_cap: float,
        waves: list[list[Subtask]],
        pw: int,
    ) -> IterationResult:
        """Run file-disjoint subtasks in parallel worktrees; merge in order; conflicts roll back."""
        workspace = self.cfg.project_dir
        safety = f"continuoso-preiter-{iteration_id}"
        tag_head(workspace, safety)
        log.info(
            "parallel subtasks: workers=%d waves=%s",
            pw,
            [len(w) for w in waves],
        )

        slot = 0
        merge_idx = 0
        all_files: set[str] = set()
        loc_add = loc_rem = 0
        pending: set[Path] = set()

        def cleanup_wts() -> None:
            for wt in list(pending):
                if wt.exists():
                    try:
                        discard_worktree(workspace, wt)
                    except Exception:
                        log.debug("discard %s", wt, exc_info=True)
                pending.discard(wt)

        try:

            def run_one(pair: tuple[Path, Subtask]) -> tuple[Path, Subtask, ChangeResult]:
                wt, st = pair
                ex = Executor(self.cfg, self.router, self.memory, iteration_id)
                return wt, st, ex.run_subtask(wt, st)

            for wave in waves:
                if time.monotonic() - start > wall_cap:
                    raise RuntimeError("wall-clock cap hit")

                prepare_workspace_for_worktrees(workspace)
                batch: list[tuple[Path, Subtask]] = []
                for st in wave:
                    if self.safeguards.is_quarantined(st.id, "", st.files):
                        log.info("subtask %s quarantined; skipping", st.id)
                        continue
                    wt = create_worktree_at_slot(workspace, iteration_id, slot)
                    slot += 1
                    batch.append((wt, st))
                    pending.add(wt)

                if not batch:
                    continue

                max_t = min(pw, len(batch))
                with ThreadPoolExecutor(max_workers=max_t) as pool:
                    results = list(pool.map(run_one, batch))

                for wt, st, res in results:
                    if not res.success:
                        self.safeguards.observe_failure(
                            subtask_slug=st.id,
                            error=res.error,
                            files=st.files,
                        )
                        raise RuntimeError(
                            f"subtask {st.id} failed: {res.error[:300]}"
                        )

                ordered = sorted(results, key=lambda x: x[1].id)
                for wt, st, res in ordered:
                    if time.monotonic() - start > wall_cap:
                        raise RuntimeError("wall-clock cap hit")
                    all_files.update(res.files_changed)
                    loc_add += res.loc_added
                    loc_rem += res.loc_removed
                    if merge_idx > 0:
                        merge_main_into_wt(wt, workspace)
                    msg = (
                        f"feat({plan.chosen_gap_id}): {plan.iteration_goal[:50]} "
                        f"[{st.id}]"
                    )
                    merge_worktree(
                        workspace,
                        wt,
                        iteration_id,
                        f"{msg}\n\niter={iteration_id} parallel p{merge_idx}\n",
                        tag_suffix=f"-p{merge_idx}",
                    )
                    merge_idx += 1
                    pending.discard(wt)

            if not all_files:
                raise RuntimeError("no changes (all subtasks skipped or empty)")

            report = self.evaluator.evaluate(
                workspace,
                files_changed=sorted(all_files),
                loc_added=loc_add,
                loc_removed=loc_rem,
            )

            if not report.passed:
                log.warning(
                    "evaluator rejected parallel iter %d: %s",
                    iteration_id,
                    report.reason[:300],
                )
                reset_main_to_tag(workspace, safety)
                delete_tag(workspace, safety)
                cleanup_wts()
                self.memory.finish_iteration(
                    iteration_id,
                    outcome="rolled_back",
                    score=report.score,
                    chosen_gap_id=plan.chosen_gap_id,
                    notes=json.dumps(report.to_dict())[:2000],
                )
                self.safeguards.observe_failure(
                    subtask_slug=plan.chosen_gap_id or "unknown",
                    error=report.reason,
                    files=sorted(all_files),
                )
                return IterationResult(
                    iteration_id,
                    "rolled_back",
                    report.score,
                    plan.iteration_goal,
                    report.reason,
                )

            delete_tag(workspace, safety)
            self.memory.finish_iteration(
                iteration_id,
                outcome="merged",
                score=report.score,
                chosen_gap_id=plan.chosen_gap_id,
                notes=(
                    f"parallel files={len(all_files)} +{loc_add}/-{loc_rem}"
                ),
            )
            self.feature_log.append_merged(
                iteration_id=iteration_id,
                gap_id=plan.chosen_gap_id,
                goal=plan.iteration_goal,
                notes=f"parallel files={len(all_files)} +{loc_add}/-{loc_rem}",
            )
            return IterationResult(
                iteration_id,
                "merged",
                report.score,
                plan.iteration_goal,
                f"parallel files={len(all_files)} +{loc_add}/-{loc_rem}",
            )

        except (MergeConflictError, RuntimeError, GitError, OSError) as e:
            log.warning("parallel execution failed: %s", e)
            try:
                reset_main_to_tag(workspace, safety)
            except Exception:
                log.exception("parallel rollback")
            delete_tag(workspace, safety)
            cleanup_wts()
            self.memory.finish_iteration(
                iteration_id,
                outcome="aborted",
                score=0.0,
                chosen_gap_id=plan.chosen_gap_id,
                notes=f"parallel: {e}"[:2000],
            )
            return IterationResult(
                iteration_id,
                "aborted",
                0.0,
                plan.iteration_goal,
                str(e)[:500],
            )

    # ------------------------------------------------------------------
    def _execute_plan(
        self,
        iteration_id: int,
        plan: Plan,
        snap: Snapshot,
    ) -> IterationResult:
        start = time.monotonic()
        wall_cap = self.cfg.budgets.max_wall_seconds

        waves = partition_into_waves(plan.subtasks)
        pw = self.cfg.env.parallel_workers
        if (
            pw > 1
            and len(plan.subtasks) > 1
            and waves
            and max(len(w) for w in waves) > 1
        ):
            return self._execute_plan_parallel(
                iteration_id, plan, start, wall_cap, waves, pw
            )

        with worktree(self.cfg.project_dir, iteration_id) as wt:
            executor = Executor(self.cfg, self.router, self.memory, iteration_id)
            any_failed = False
            files_changed: set[str] = set()
            loc_add = loc_rem = 0

            for st in plan.subtasks:
                if time.monotonic() - start > wall_cap:
                    any_failed = True
                    log.warning("wall-clock cap hit; aborting iteration")
                    break

                # Fingerprint check — don't attempt quarantined tasks.
                if self.safeguards.is_quarantined(st.id, "", st.files):
                    log.info("subtask %s quarantined; skipping", st.id)
                    continue

                res = executor.run_subtask(wt, st)
                if not res.success:
                    any_failed = True
                    self.safeguards.observe_failure(
                        subtask_slug=st.id, error=res.error, files=st.files,
                    )
                    log.warning("subtask %s failed: %s", st.id, res.error[:200])
                    break

                files_changed.update(res.files_changed)
                loc_add += res.loc_added
                loc_rem += res.loc_removed

            if any_failed or not files_changed:
                discard_worktree(self.cfg.project_dir, wt)
                self.memory.finish_iteration(
                    iteration_id, outcome="aborted",
                    score=0.0, chosen_gap_id=plan.chosen_gap_id,
                    notes="subtask failure" if any_failed else "no changes",
                )
                return IterationResult(
                    iteration_id, "aborted", 0.0, plan.iteration_goal,
                    "subtask failed" if any_failed else "no changes",
                )

            # 4. Evaluate.
            report = self.evaluator.evaluate(
                wt,
                files_changed=sorted(files_changed),
                loc_added=loc_add,
                loc_removed=loc_rem,
            )

            if not report.passed:
                log.warning(
                    "evaluator rejected iter %d: %s",
                    iteration_id, report.reason[:300],
                )
                discard_worktree(self.cfg.project_dir, wt)
                self.memory.finish_iteration(
                    iteration_id, outcome="rolled_back",
                    score=report.score, chosen_gap_id=plan.chosen_gap_id,
                    notes=json.dumps(report.to_dict())[:2000],
                )
                # Record failure fingerprint so the same evaluator failure
                # eventually quarantines this gap.
                self.safeguards.observe_failure(
                    subtask_slug=plan.chosen_gap_id or "unknown",
                    error=report.reason,
                    files=sorted(files_changed),
                )
                return IterationResult(
                    iteration_id, "rolled_back", report.score,
                    plan.iteration_goal, report.reason,
                )

            # 5. Commit + merge.
            msg = f"feat({plan.chosen_gap_id}): {plan.iteration_goal[:60]}"
            sha = commit_all(wt, msg)
            if not sha:
                discard_worktree(self.cfg.project_dir, wt)
                self.memory.finish_iteration(
                    iteration_id, outcome="empty", score=0.0,
                    chosen_gap_id=plan.chosen_gap_id, notes="no diff after edits",
                )
                return IterationResult(
                    iteration_id, "empty", 0.0, plan.iteration_goal, "no diff",
                )

            stat = diff_stat(wt)
            merge_worktree(
                self.cfg.project_dir, wt, iteration_id,
                f"{msg}\n\niter={iteration_id}\n\n{stat}",
            )
            self.memory.finish_iteration(
                iteration_id, outcome="merged", score=report.score,
                chosen_gap_id=plan.chosen_gap_id,
                notes=f"files={len(files_changed)} +{loc_add}/-{loc_rem}",
            )
            self.feature_log.append_merged(
                iteration_id=iteration_id,
                gap_id=plan.chosen_gap_id,
                goal=plan.iteration_goal,
                notes=f"files={len(files_changed)} +{loc_add}/-{loc_rem}",
            )
            return IterationResult(
                iteration_id, "merged", report.score,
                plan.iteration_goal,
                f"files={len(files_changed)} +{loc_add}/-{loc_rem}",
            )
