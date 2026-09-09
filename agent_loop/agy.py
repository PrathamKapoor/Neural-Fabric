"""Antigravity CLI (`agy`) worker backend.

Fits the EXISTING worker interface: `AgyWorkerClient` exposes the same
`chat(messages, tools=None, force_json=False, on_retry=None)` contract as
`OpenAICompatibleClient`, so `RoleChain` (manager/worker separation,
infrastructure-only fallback, invocation records) works unchanged. The worker
chain becomes: AGY (local headless CLI) -> Nemotron Super (OpenRouter) ->
DeepSeek V4 Flash (NVIDIA).

Design notes (all load-bearing):
- No API wrapper around Gemini, no OAuth handling, no credentials anywhere:
  `agy` uses the user's own local Antigravity session. This module never reads,
  prints, or forwards any token.
- One `chat()` call == one headless `agy --print` run that executes the whole
  assigned task (AGY does its own internal tool loop). The returned assistant
  message carries NO `tool_calls`, so the orchestrator's worker step ends
  after exactly one turn and parses the report text as usual.
- Failure kinds are distinguished explicitly (missing/auth/permission/model/
  quota/timeout/exit-error/success) so provider failures take the fallback
  path while genuine task results flow to manager review. An AGY timeout is a
  task-budget event, never a research finding.
- `--dangerously-skip-permissions` is passed only when explicitly enabled
  in config (`AGY_DANGEROUSLY_SKIP_PERMISSIONS`, default on for this machine
  by direct user authorization); otherwise the scoped `--mode` (default
  `accept-edits`) governs repository work.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

try:
    from .clients import ApiError, TransientApiError
except ImportError:  # pragma: no cover - direct script use
    from clients import ApiError, TransientApiError


class AgyError(ApiError):
    """Base AGY failure with a machine-readable `kind`.

    Kinds: missing | auth | permission | model | quota | timeout |
    exit_error | usage. `retryable=True` marks quota/timeout (transient).
    """

    def __init__(self, kind: str, message: str, retryable: bool = False):
        super().__init__("[agy:%s] %s" % (kind, message))
        self.kind = kind
        self.retryable = retryable


_AUTH_HINTS = ("not logged in", "not authenticated", "authentication",
               "unauthorized", "401", "oauth", "login required",
               "please log in", "invalid grant", "expired token")
_PERMISSION_HINTS = ("permission denied", "not permitted", "approval required",
                     "needs approval", "denied by policy", "eperm", "eacces")
_QUOTA_HINTS = ("quota", "rate limit", "rate-limit", "429", "too many requests",
                "resource exhausted", "usage limit")
_MODEL_HINTS = ("model not found", "unknown model", "invalid model",
                "model unavailable")
_TIMEOUT_HINTS = ("timed out", "timeout", "deadline exceeded")

MAX_STDOUT_CHARS = 60000


def classify_stderr(text: str) -> str:
    """Map CLI stderr/exit context to a failure kind (no secrets involved)."""
    low = (text or "").lower()
    for hints, kind in (
        (_AUTH_HINTS, "auth"),
        (_PERMISSION_HINTS, "permission"),
        (_QUOTA_HINTS, "quota"),
        (_MODEL_HINTS, "model"),
        (_TIMEOUT_HINTS, "timeout"),
    ):
        if any(h in low for h in hints):
            return kind
    return "exit_error"


def extract_result_text(stdout: str) -> str:
    """Pull the agent's result text out of `--output-format json` output.

    Falls back to raw text when the CLI does not emit JSON (caller then lets
    the normal report parser judge it - never invent structure here).
    """
    text = (stdout or "").strip()
    if not text:
        return ""
    try:
        obj = json.loads(text)
    except ValueError:
        return text
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("result", "response", "output", "text", "content",
                    "message", "answer"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val
        # Empty result with headless soft-denials: surface WHAT was denied
        # so the manager can distinguish "task done" from "tools blocked".
        denied = obj.get("denied_actions")
        if isinstance(denied, list) and denied:
            names = ", ".join(
                str(d.get("display_name") or d.get("action", "?"))
                for d in denied if isinstance(d, dict))[:300]
            return ("[agy: requested tools were denied in headless mode: %s; "
                    "no task output was produced]" % (names or "?"))
        # Unknown envelope: preserve it verbatim (truncated) for the reviewer.
        return json.dumps(obj)[:MAX_STDOUT_CHARS]
    return text[:MAX_STDOUT_CHARS]


def find_agy(command: str = "agy") -> str | None:
    """Resolve the AGY executable (explicit path or PATH lookup)."""
    if os.path.isabs(command) or os.sep in command:
        return command if os.path.isfile(command) else None
    return shutil.which(command)


class AgyWorkerClient:
    """Headless `agy --print` executor behind the worker chat contract."""

    def __init__(self, repo_root: str = ".", command: str = "agy",
                 model: str = "", effort: str = "", mode: str = "accept-edits",
                 timeout_s: float = 360.0,
                 allow_dangerously_skip_permissions: bool = False):
        self.repo_root = os.path.abspath(repo_root)
        self.command = command
        self.model = model or "(agy default)"
        self._requested_model = model
        self.effort = effort
        self.mode = mode
        self.timeout_s = timeout_s
        self.allow_dangerous = allow_dangerously_skip_permissions
        self.provider = "agy"
        self.retries = 1  # no client-level retry; the chain owns fallback

    # -- introspection -------------------------------------------------
    def executable(self) -> str | None:
        return find_agy(self.command)

    def check_usability(self, probe_timeout_s: float = 45.0) -> dict:
        """Is AGY present AND authenticated? Never touches credentials.

        Runs `agy models` (read-only model list) as the auth probe: it
        requires a valid local session and changes nothing.
        """
        exe = self.executable()
        if not exe:
            return {"ok": False, "reason": "agy executable not found "
                    "(AGY_COMMAND=%r)" % self.command}
        try:
            proc = subprocess.run(
                [exe, "models"], cwd=self.repo_root, capture_output=True,
                text=True, timeout=probe_timeout_s)
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": "agy models probe timed out",
                    "kind": "timeout"}
        except OSError as e:
            return {"ok": False, "reason": "cannot execute agy: %s" % e,
                    "kind": "missing"}
        if proc.returncode != 0:
            blob = (proc.stdout or "") + (proc.stderr or "")
            return {"ok": False,
                    "reason": ("agy models probe failed: %s"
                               % blob.strip()[:300] or "exit %d"
                               % proc.returncode),
                    "kind": classify_stderr(blob)}
        return {"ok": True, "reason": "agy present and session valid",
                "models": [l.split()[0] for l in
                           (proc.stdout or "").splitlines() if l.strip()
                           ][:25]}

    # -- worker contract -----------------------------------------------
    def chat(self, messages: list, tools: list | None = None,
             force_json: bool = False, on_retry=None) -> dict:
        """Execute the assigned task via one headless AGY run.

        Returns a plain assistant message (never `tool_calls`): the worker
        step ends after this single turn and parses the content as the
        structured report. Raises AgyError (an ApiError) for provider-side
        failures so the chain falls back; returns raw text for everything
        else so the manager judges actual task results.
        """
        _ = (tools, force_json, on_retry)  # AGY owns its tool loop
        exe = self.executable()
        if not exe:
            raise AgyError("missing", "agy executable not found "
                           "(AGY_COMMAND=%r)" % self.command)
        prompt = _build_prompt(messages)
        args = [exe, "--print", prompt, "--output-format", "json",
                "--add-dir", self.repo_root, "--mode", self.mode]
        if self._requested_model:
            args += ["--model", self._requested_model]
        if self.effort:
            args += ["--effort", self.effort]
        if self.allow_dangerous:
            args += ["--dangerously-skip-permissions"]
        t0 = time.time()
        try:
            proc = subprocess.run(
                args, cwd=self.repo_root, capture_output=True, text=True,
                timeout=self.timeout_s)
        except subprocess.TimeoutExpired as e:
            partial = ""
            if isinstance(e.stdout, str):
                partial = e.stdout[-2000:]
            raise AgyError(
                "timeout", "agy exceeded %.0fs (task budget, not a research "
                "finding)%s" % (self.timeout_s,
                                 (": " + partial) if partial else ""),
                retryable=True)
        except OSError as e:
            raise AgyError("missing", "cannot execute agy: %s" % e)
        duration_s = time.time() - t0
        stdout = (proc.stdout or "")
        stderr = (proc.stderr or "")
        if proc.returncode != 0:
            kind = classify_stderr(stdout + "\n" + stderr)
            detail = (stderr.strip() or stdout.strip())[:800]
            raise AgyError(
                kind, "agy exited %d after %.1fs: %s"
                % (proc.returncode, duration_s, detail or "(no output)"),
                retryable=kind in ("quota", "timeout"))
        text = extract_result_text(stdout)
        if stderr.strip():
            text = (text + "\n\n[agy stderr]\n" + stderr.strip())[:MAX_STDOUT_CHARS]
        if not text.strip():
            raise AgyError("exit_error", "agy exited 0 with empty output "
                           "after %.1fs" % duration_s)
        return {"role": "assistant", "content": text, "_agy": True,
                "_agy_duration_s": round(duration_s, 1)}

    def check_connectivity(self) -> dict:
        res = self.check_usability()
        return {"ok": bool(res.get("ok")), "provider": self.provider,
                "model": self.model,
                "error": "" if res.get("ok") else res.get("reason", "")}


def _build_prompt(messages: list) -> str:
    """Flatten the loop's system+user messages into one AGY task prompt."""
    parts = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, list):
            content = "".join(b.get("text", "") for b in content
                              if isinstance(b, dict))
        if content:
            parts.append(str(content))
    return "\n\n".join(parts)
