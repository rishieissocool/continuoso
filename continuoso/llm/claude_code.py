"""Claude Code CLI wrapper.

Invokes the `claude` binary in non-interactive mode (`-p`) with JSON output,
run inside a specified working directory. Uses the user's existing Claude Code
auth, so no API key is needed here.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time

from .base import LLMClient, LLMError, LLMResponse


class ClaudeCodeClient(LLMClient):
    provider = "claude_code"

    def __init__(self, bin_path: str = "claude", timeout: int = 900) -> None:
        resolved = shutil.which(bin_path) or bin_path
        self.bin = resolved
        self.timeout = timeout

    def available(self) -> bool:
        return shutil.which(self.bin) is not None or (
            self.bin and not self.bin.endswith("claude")
        )

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
        cmd = [
            self.bin,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            model,
        ]

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
