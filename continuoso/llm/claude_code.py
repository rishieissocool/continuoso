"""Claude Code CLI wrapper.

Invokes the `claude` binary in non-interactive mode (`-p`) with JSON output,
`--dangerously-skip-permissions` (so file edits are not blocked on approval prompts),
run inside a specified working directory. Uses the user's existing Claude Code
auth, so no API key is needed here.

Resolution order (especially important on Windows):
  1. ``CLAUDE_CODE_BIN`` if it is an existing file path
  2. ``shutil.which`` on the hint and on ``claude.cmd`` / ``claude.exe``
  3. npm global shims under ``%APPDATA%\\npm`` and ``%LOCALAPPDATA%\\npm``
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

from .base import LLMClient, LLMError, LLMResponse

log = logging.getLogger(__name__)


def _try_parse_json_stdout(raw: str) -> dict | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else None
    except json.JSONDecodeError:
        return None


def _claude_envelope_is_failure(env: dict) -> str | None:
    """If this JSON envelope is an error/quota/limit, return a user-facing message."""
    result = str(env.get("result") or env.get("content") or env.get("text") or "").strip()
    if env.get("is_error"):
        return result or "Claude Code returned is_error (no message)"
    if env.get("subtype") == "error":
        return result or "Claude Code error"
    low = result.lower()
    if result and any(
        p in low
        for p in (
            "hit your limit",
            "you've hit your limit",
            "rate limit",
            "usage limit",
            "quota exceeded",
            "billing required",
        )
    ):
        return result
    return None


def resolve_claude_executable(hint: str = "claude") -> str | None:
    """Return a path to the Claude Code CLI, or None if not found."""
    hint = (hint or "claude").strip() or "claude"

    try:
        p = Path(hint)
        if p.is_file():
            return str(p.resolve())
        # e.g. "claude" without extension on Windows
        if os.name == "nt" and not p.suffix:
            for ext in (".exe", ".cmd", ".bat"):
                pe = p.with_suffix(ext)
                if pe.is_file():
                    return str(pe.resolve())
    except OSError:
        pass

    names: list[str] = [hint]
    if os.name == "nt":
        for extra in ("claude.cmd", "claude.exe"):
            if extra not in names:
                names.append(extra)

    for name in names:
        w = shutil.which(name)
        if w:
            return w

    # pipx / uv / standalone user installs (often missing from GUI/IDE PATH)
    try:
        local_bin = Path.home() / ".local" / "bin"
        if local_bin.is_dir():
            for name in ("claude.exe", "claude.cmd", "claude"):
                cand = local_bin / name
                if cand.is_file():
                    return str(cand)
    except OSError:
        pass

    if os.name == "nt":
        for base in (os.environ.get("APPDATA"), os.environ.get("LOCALAPPDATA")):
            if not base:
                continue
            for sub in ("npm/claude.cmd", "npm/claude.exe", "npm/claude"):
                cand = Path(base) / sub
                if cand.is_file():
                    return str(cand)

    return None


class ClaudeCodeClient(LLMClient):
    provider = "claude_code"

    def __init__(self, bin_path: str = "claude", timeout: int = 900) -> None:
        self._hint = bin_path
        self.timeout = timeout
        self.bin = resolve_claude_executable(bin_path) or ""

    def available(self) -> bool:
        return bool(self.bin)

    def _argv(self, prompt: str, model: str) -> list[str]:
        tail = [
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            model,
            "--dangerously-skip-permissions",
        ]
        if not self.bin:
            raise LLMError("Claude Code CLI not found on PATH")
        # npm's claude.cmd must run via cmd.exe; CreateProcess on .cmd is unreliable.
        if os.name == "nt" and self.bin.lower().endswith((".cmd", ".bat")):
            comspec = os.environ.get("COMSPEC", "cmd.exe")
            return [comspec, "/c", self.bin, *tail]
        return [self.bin, *tail]

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
        workdir: str | None = None,
    ) -> LLMResponse:
        if not self.available():
            raise LLMError("Claude Code CLI not found on PATH")

        prompt = f"{system}\n\n---\n\n{user}"
        cmd = self._argv(prompt, model)

        cwd = workdir
        if cwd:
            log.info(
                "claude_code: invoking CLI (cwd=%s, model=%s)",
                cwd,
                model,
            )

        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            raise LLMError("Claude Code CLI timed out")
        except FileNotFoundError as e:
            raise LLMError(f"Claude Code CLI not executable: {e}")

        latency_ms = int((time.monotonic() - start) * 1000)

        raw = proc.stdout.strip()
        combined_err = (proc.stderr or "").strip() or raw[:800]

        # Exit 1 often still prints a JSON envelope to stdout (quota, limit, is_error, …).
        env = _try_parse_json_stdout(raw)
        if isinstance(env, dict):
            fail_msg = _claude_envelope_is_failure(env)
            if fail_msg:
                log.warning(
                    "claude_code: CLI failure/quota — trying next model (%s)",
                    fail_msg[:200],
                )
                raise LLMError(f"Claude Code: {fail_msg}")

        if proc.returncode != 0:
            raise LLMError(
                f"Claude CLI exit {proc.returncode}: {combined_err[:600]}"
            )

        text = raw
        in_tok = out_tok = 0
        cost = 0.0
        if isinstance(env, dict):
            text = (
                env.get("result")
                or env.get("content")
                or env.get("text")
                or raw
            )
            usage = env.get("usage") or {}
            in_tok = int(usage.get("input_tokens", 0))
            out_tok = int(usage.get("output_tokens", 0))
            cost = float(env.get("total_cost_usd", 0.0))

        return LLMResponse(
            text=text,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
