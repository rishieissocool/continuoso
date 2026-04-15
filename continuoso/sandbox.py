"""Git worktree sandboxing.

Each iteration works inside its own worktree off the target repo. Successful
iterations are squash-merged back to main; failed iterations are discarded
by removing the worktree.

Parallel subtasks use multiple worktrees from the same snapshot; branches are
merged in order with ``git merge main`` into each worktree before squash-merge
to reduce conflicts when file sets are disjoint.
"""
from __future__ import annotations

import logging
import secrets
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


class GitError(RuntimeError):
    pass


class MergeConflictError(GitError):
    """Raised when merging main into a worktree or squash-merge hits conflicts."""


def _run(cmd: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess:
    log.debug("$ %s (cwd=%s)", " ".join(cmd), cwd)
    r = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=120
    )
    if check and r.returncode != 0:
        raise GitError(f"{' '.join(cmd)} -> {r.returncode}: {r.stderr.strip() or r.stdout}")
    return r


def ensure_repo(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    state = workspace / ".continuoso"
    state.mkdir(exist_ok=True)
    # Never let our private state be tracked by the target repo.
    gi = state / ".gitignore"
    if not gi.exists():
        gi.write_text("*\n", encoding="utf-8")

    if not (workspace / ".git").exists():
        _run(["git", "init", "-b", "main"], workspace)
        # Configure a local identity (only if missing) so commits work offline.
        name = _run(["git", "config", "user.name"], workspace, check=False).stdout.strip()
        if not name:
            _run(["git", "config", "user.name", "continuoso"], workspace)
            _run(["git", "config", "user.email", "bot@continuoso.local"], workspace)

    # Worktrees need HEAD to resolve; make an empty initial commit if repo has none.
    r = _run(["git", "rev-parse", "--verify", "HEAD"], workspace, check=False)
    if r.returncode != 0:
        _run(["git", "commit", "--allow-empty", "-m", "chore: continuoso init"], workspace)


def _snapshot_untracked(workspace: Path) -> None:
    """Commit any new/modified files so the worktree starts from the real state."""
    _run(["git", "add", "-A"], workspace)
    r = _run(["git", "diff", "--cached", "--quiet"], workspace, check=False)
    if r.returncode != 0:
        _run(["git", "commit", "-m", "chore: snapshot workspace state"], workspace)


def prepare_workspace_for_worktrees(workspace: Path) -> None:
    """Commit dirty tree once before creating one or more worktrees."""
    ensure_repo(workspace)
    _snapshot_untracked(workspace)


def create_worktree_at_slot(workspace: Path, iteration_id: int, slot: int) -> Path:
    """Create a new worktree branch from current main. Caller must have run prepare first."""
    wt_root = workspace / ".continuoso" / "worktrees"
    wt_root.mkdir(parents=True, exist_ok=True)
    branch = f"cont/iter-{iteration_id}-s{slot}-{secrets.token_hex(4)}"
    safe = branch.replace("/", "_")
    wt_path = wt_root / safe
    _run(["git", "worktree", "add", "-b", branch, str(wt_path), "main"], workspace)
    return wt_path


@contextmanager
def worktree(workspace: Path, iteration_id: int, slot: int = 0) -> Iterator[Path]:
    """Yield a Path to an ephemeral worktree. Caller decides merge/discard."""
    prepare_workspace_for_worktrees(workspace)
    wt_path = create_worktree_at_slot(workspace, iteration_id, slot)
    try:
        yield wt_path
    finally:
        pass


def merge_main_into_wt(wt: Path, workspace: Path) -> None:
    """Update worktree branch with latest main (call before squash-merge when main moved)."""
    r = _run(["git", "merge", "main"], wt, check=False)
    if r.returncode != 0:
        raise MergeConflictError(
            f"git merge main in worktree failed (conflict?): {r.stderr[:800] or r.stdout[:800]}"
        )


def commit_all(wt: Path, message: str) -> str | None:
    """Stage and commit everything in the worktree. Returns sha or None if empty."""
    _run(["git", "add", "-A"], wt)
    r = _run(["git", "diff", "--cached", "--quiet"], wt, check=False)
    if r.returncode == 0:
        return None  # nothing to commit
    _run(["git", "commit", "-m", message], wt)
    sha = _run(["git", "rev-parse", "HEAD"], wt).stdout.strip()
    return sha


def merge_worktree(
    workspace: Path,
    wt: Path,
    iteration_id: int,
    message: str,
    *,
    tag_suffix: str = "",
) -> None:
    """Squash-merge worktree branch into main, then remove the worktree."""
    branch = _branch_of(wt)
    _run(["git", "checkout", "main"], workspace)
    r = _run(["git", "merge", "--squash", branch], workspace, check=False)
    if r.returncode != 0:
        raise MergeConflictError(
            f"squash merge failed: {r.stderr[:800] or r.stdout[:800]}"
        )
    _run(["git", "commit", "-m", message], workspace)
    sha = _run(["git", "rev-parse", "HEAD"], workspace).stdout.strip()
    tag = f"iter-{iteration_id}{tag_suffix}"
    _run(["git", "tag", tag, sha], workspace)
    _remove_worktree(workspace, wt, branch)


def discard_worktree(workspace: Path, wt: Path) -> None:
    branch = _branch_of(wt)
    _remove_worktree(workspace, wt, branch)


def diff_stat(wt: Path) -> str:
    r = _run(["git", "diff", "main...HEAD", "--stat"], wt, check=False)
    return r.stdout.strip()


def tag_head(workspace: Path, name: str) -> None:
    """Annotate current main HEAD (for rollback)."""
    _run(["git", "tag", name, "HEAD"], workspace)


def reset_main_to_tag(workspace: Path, tag: str) -> None:
    """Hard reset main to the given tag (parallel rollback)."""
    _run(["git", "checkout", "main"], workspace)
    _run(["git", "reset", "--hard", tag], workspace)


def delete_tag(workspace: Path, tag: str) -> None:
    _run(["git", "tag", "-d", tag], workspace, check=False)


def rollback_last_iteration(workspace: Path, iteration_id: int) -> None:
    """Revert the merge commit tagged for the given iteration."""
    tag = f"iter-{iteration_id}"
    _run(["git", "revert", "--no-edit", tag], workspace)


# ---- internals ----
def _branch_of(wt: Path) -> str:
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], wt).stdout.strip()


def _remove_worktree(workspace: Path, wt: Path, branch: str) -> None:
    try:
        _run(["git", "worktree", "remove", "--force", str(wt)], workspace, check=False)
    finally:
        if wt.exists():
            shutil.rmtree(wt, ignore_errors=True)
        _run(["git", "branch", "-D", branch], workspace, check=False)
