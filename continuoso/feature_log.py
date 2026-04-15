"""Append-only feature log: per-session file + master list under `.continuoso/logs/`."""
from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

FEATURES_MD = "logs/features.md"
SESSIONS_DIR = "logs/sessions"


def new_session_id() -> str:
    """UTC timestamp + short hex — unique per `continuoso run` process."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{secrets.token_hex(3)}"


def _sanitize_line(s: str, max_len: int = 280) -> str:
    s = " ".join(s.split())
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


class FeatureLog:
    """Writes `logs/sessions/<session>.md` and appends to `logs/features.md`."""

    def __init__(
        self,
        state_dir: Path,
        session_id: str,
        *,
        session_focus: str | None = None,
    ) -> None:
        self.state_dir = state_dir
        self.session_id = session_id
        self.session_focus = (session_focus or "").strip() or None
        self._master_section_written = False

    @property
    def session_md_path(self) -> Path:
        return self.state_dir / SESSIONS_DIR / f"{self.session_id}.md"

    @property
    def master_md_path(self) -> Path:
        return self.state_dir / FEATURES_MD

    def start_session(self) -> None:
        """Create session file and ensure master log exists."""
        logs = self.state_dir / SESSIONS_DIR
        logs.mkdir(parents=True, exist_ok=True)
        self.master_md_path.parent.mkdir(parents=True, exist_ok=True)

        path = self.session_md_path
        if not path.exists():
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            focus_block = ""
            if self.session_focus:
                focus_block = f"**Session focus:** {self.session_focus}\n\n"
            path.write_text(
                f"# continuoso session `{self.session_id}`\n\n"
                f"**Started:** {now}\n\n"
                f"{focus_block}"
                f"## Features implemented\n\n",
                encoding="utf-8",
            )

        if not self.master_md_path.exists():
            self.master_md_path.write_text(
                "# continuoso — feature log\n\n"
                "Auto-generated list of merged iterations (one entry per successful merge).\n"
                "Per-session files: [`sessions/`](./sessions/).\n\n",
                encoding="utf-8",
            )

    def append_merged(
        self,
        *,
        iteration_id: int,
        gap_id: str | None,
        goal: str,
        notes: str,
    ) -> None:
        """Record a shipped iteration (after merge to main)."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        gid = gap_id or "—"
        g = _sanitize_line(goal)
        line = (
            f"- **{ts}** · iter {iteration_id} · gap `{gid}` — {g} — _{notes}_\n"
        )
        try:
            with self.session_md_path.open("a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            log.warning("feature log (session): %s", e)
            return

        try:
            with self.master_md_path.open("a", encoding="utf-8") as f:
                if not self._master_section_written:
                    f.write(f"\n## Session `{self.session_id}`\n\n")
                    self._master_section_written = True
                f.write(line)
        except OSError as e:
            log.warning("feature log (master): %s", e)
