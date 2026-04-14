"""Claude Code CLI wrapper.

Invokes the `claude` binary in non-interactive mode (`-p`) with JSON output,
run inside a specified working directory. Uses the user's existing Claude Code
auth, so no API key is needed here.

Resolution order (especially important on Windows):
  1. ``CLAUDE_CODE_BIN`` if it is an existing file path
  2. ``shutil.which`` on the hint and on ``claude.cmd`` / ``claude.exe``
  3. npm global shims under ``%APPDATA%\\npm`` and ``%LOCALAPPDATA%\\npm``
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from .base import LLMClient, LLMError, LLMResponse


def resolve_claude_executable(hint: str = "claude") -> str | None:
    """Return a path to the Claude Code CLI, or None if not found."""
    hint = (hint or "claude").strip() or "claude"

    try:
        p = Path(hint)
        if p.is_file():
            return str(p.resolve())
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

        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
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

        if proc.returncode != 0:
            raise LLMError(
                f"Claude CLI exit {proc.returncode}: {proc.stderr[:500] or proc.stdout[:500]}"
            )

        # --output-format json returns a JSON envelope with the assistant text
        # in a `result` or `content` field depending on version.
        raw = proc.stdout.strip()
        text = raw
        in_tok = out_tok = 0
        cost = 0.0
        try:
            env = json.loads(raw)
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
        except json.JSONDecodeError:
            # Older CLI versions stream plain text; use it directly.
            text = raw

        return LLMResponse(
            text=text,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
