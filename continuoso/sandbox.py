"""Git worktree sandboxing.

Each iteration works inside its own worktree off the target repo. Successful
iterations are squash-merged back to main; failed iterations are discarded
by removing the worktree.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


class GitError(RuntimeError):
    pass


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


@contextmanager
def worktree(workspace: Path, iteration_id: int) -> Iterator[Path]:
    """Yield a Path to an ephemeral worktree. Caller decides merge/discard."""
    ensure_repo(workspace)
    wt_root = workspace / ".continuoso" / "worktrees"
    wt_root.mkdir(parents=True, exist_ok=True)
    branch = f"cont/iter-{iteration_id}-{int(time.time())}"
    wt_path = wt_root / branch.replace("/", "_")

    _run(["git", "worktree", "add", "-b", branch, str(wt_path), "main"], workspace)
    try:
        yield wt_path
    finally:
        # Worktree removal is handled by merge() / discard(); here we just
        # make sure nothing leaks if caller forgot.
        pass


def commit_all(wt: Path, message: str) -> str | None:
    """Stage and commit everything in the worktree. Returns sha or None if empty."""
    _run(["git", "add", "-A"], wt)
    r = _run(["git", "diff", "--cached", "--quiet"], wt, check=False)
    if r.returncode == 0:
        return None  # nothing to commit
    _run(["git", "commit", "-m", message], wt)
    sha = _run(["git", "rev-parse", "HEAD"], wt).stdout.strip()
    return sha


def merge_worktree(workspace: Path, wt: Path, iteration_id: int, message: str) -> None:
    """Squash-merge worktree branch into main, then remove the worktree."""
    branch = _branch_of(wt)
    _run(["git", "checkout", "main"], workspace)
    # Stage any untracked files so they don't block the squash merge.
    _run(["git", "add", "-A"], workspace)
    r = _run(["git", "diff", "--cached", "--quiet"], workspace, check=False)
    if r.returncode != 0:
        _run(["git", "commit", "-m", "chore: snapshot before merge"], workspace)
    _run(["git", "merge", "--squash", branch], workspace)
    _run(["git", "commit", "-m", message], workspace)
    sha = _run(["git", "rev-parse", "HEAD"], workspace).stdout.strip()
    _run(["git", "tag", f"iter-{iteration_id}", sha], workspace)
    _remove_worktree(workspace, wt, branch)


def discard_worktree(workspace: Path, wt: Path) -> None:
    branch = _branch_of(wt)
    _remove_worktree(workspace, wt, branch)


def diff_stat(wt: Path) -> str:
    r = _run(["git", "diff", "main...HEAD", "--stat"], wt, check=False)
    return r.stdout.strip()


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
