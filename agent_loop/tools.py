"""Worker tool layer: real repository inspection / modification / execution.

Exposed to the worker LLM as OpenAI-compatible function tools, and executed
locally by ToolExecutor. Safety rules (git safety is load-bearing):

- All paths are confined to the repository root.
- `git push` (any form) is always blocked - the loop never pushes.
- Destructive git commands (reset --hard, clean -fdx, checkout -- .) blocked.
- `git commit` only when the current task explicitly instructs a commit.
- Shell commands run with a timeout; destructive rm patterns outside the repo
  or targeting .git are blocked.
"""
from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MAX_READ_CHARS = 60000
MAX_OUTPUT_CHARS = 20000

WORKER_TOOLS = [
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List/search repository files by glob pattern.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Glob like fabric/**/*.py (default **/*.py)"}},
         "required": []}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a repository file (truncated past 60k chars).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Repo-relative path"}},
         "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a repository file with full content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Repo-relative path"},
            "content": {"type": "string", "description": "Full new file content"}},
         "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace the first occurrence of old_string with new_string.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"}},
         "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Run a shell command in the repo root (timeout 120s default).",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout_s": {"type": "number"}},
         "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "git_status",
        "description": "Show git status (porcelain + branch).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "git_diff",
        "description": "Show git diff (stat + patch, truncated).",
        "parameters": {"type": "object", "properties": {
            "stat_only": {"type": "boolean"}},
         "required": []}}},
    {"type": "function", "function": {
        "name": "list_artifacts",
        "description": "List generated experiment artifacts (results/runs).",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}},
          "required": []}}},
    {"type": "function", "function": {
        "name": "queue_job",
        "description": ("Queue a DETACHED experiment job (survives this turn; "
                        "poll later with poll_job). argv list only, no shell. "
                        "Use for compute longer than a few minutes."),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "array", "items": {"type": "string"},
                        "description": "argv, e.g. [python, experiments/x.py]"},
            "experiment_id": {"type": "string"},
            "seed": {"type": "integer"},
            "timeout_s": {"type": "number",
                          "description": "experiment deadline seconds (hours allowed, max 86400)"},
            "mem_mb": {"type": "number"},
            "heartbeat": {"type": "string", "description": "repo-relative heartbeat JSONL path the script writes"},
            "result": {"type": "string", "description": "repo-relative result file marking completion"},
            "artifact": {"type": "string"},
            "checkpoint": {"type": "string"}},
          "required": ["command", "experiment_id"]}}},
    {"type": "function", "function": {
        "name": "poll_job",
        "description": "Check a detached job: status/elapsed/heartbeat/log tail. Never blocks.",
        "parameters": {"type": "object", "properties": {
            "job_id": {"type": "string"}},
          "required": ["job_id"]}}},
]

_DESTRUCTIVE_GIT = ("reset --hard", "clean -f", "clean -fd", "checkout -- .",
                    "checkout --", "branch -D", "push --force", "push -f",
                    "push --delete", "filter-branch")
_BLOCKED_RM = ("rm -rf /", "rm -rf ~", "rm -rf .git", "rmdir /s")


class ToolRefusal(RuntimeError):
    pass


class ToolExecutor:
    def __init__(self, repo_root: str | Path, allow_commit: bool = False,
                 command_timeout_s: float = 120.0):
        self.repo_root = Path(repo_root).resolve()
        self.allow_commit = allow_commit
        self.command_timeout_s = command_timeout_s
        self.invocation_log: list = []

    # -- helpers -------------------------------------------------------
    def _resolve(self, rel: str) -> Path:
        p = (self.repo_root / rel).resolve()
        try:
            p.relative_to(self.repo_root)
        except ValueError:
            raise ToolRefusal("path escapes repository root: %r" % rel)
        return p

    def _run(self, args: list, timeout: float | None = None) -> dict:
        try:
            proc = subprocess.run(
                args, cwd=str(self.repo_root), capture_output=True, text=True,
                timeout=timeout or self.command_timeout_s)
            out = (proc.stdout or "") + (proc.stderr or "")
            return {"ok": proc.returncode == 0, "returncode": proc.returncode,
                    "output": out[-MAX_OUTPUT_CHARS:]}
        except subprocess.TimeoutExpired:
            return {"ok": False, "returncode": -1, "output": "TIMEOUT after %ss" % timeout}
        except FileNotFoundError as e:
            return {"ok": False, "returncode": -1, "output": "command not found: %s" % e}

    def _git_guard(self, command: str):
        low = " ".join(command.split()).lower()
        if low.startswith("git push") or " git push " in (" " + low + " "):
            raise ToolRefusal("git push is disabled: the loop never pushes to remote.")
        for pat in _DESTRUCTIVE_GIT:
            if pat in low:
                raise ToolRefusal("destructive git operation blocked: %r" % pat)
        if low.startswith("git commit") and not self.allow_commit:
            raise ToolRefusal("git commit requires the manager task to explicitly say 'commit'.")

    # -- dispatch ------------------------------------------------------
    def execute(self, name: str, args: dict) -> dict:
        args = dict(args or {})
        t0 = time.time()
        try:
            if name == "list_files":
                result = self._list_files(args.get("pattern", "**/*.py"))
            elif name == "read_file":
                result = self._read(args["path"])
            elif name == "write_file":
                result = self._write(args["path"], args.get("content", ""))
            elif name == "edit_file":
                result = self._edit(args["path"], args.get("old_string", ""),
                                    args.get("new_string", ""))
            elif name == "run_command":
                result = self._shell(args["command"], args.get("timeout_s"))
            elif name == "git_status":
                result = {"ok": True, "output": self._git_status()}
            elif name == "git_diff":
                result = {"ok": True, "output": self._git_diff(bool(args.get("stat_only", False)))}
            elif name == "list_artifacts":
                result = self._artifacts(args.get("pattern", "*.json"))
            elif name == "queue_job":
                result = self._queue_job(args)
            elif name == "poll_job":
                result = self._poll_job(args.get("job_id", ""))
            else:
                result = {"ok": False, "output": "unknown tool: %s" % name}
        except ToolRefusal as e:
            result = {"ok": False, "output": "REFUSED: %s" % e}
        except KeyError as e:
            result = {"ok": False, "output": "missing argument: %s" % e}
        except Exception as e:  # noqa: BLE001 - tool failures are data
            result = {"ok": False, "output": "tool error: %s" % str(e)[:1000]}
        self.invocation_log.append({"tool": name, "args": _redact(args),
                                    "ok": result.get("ok"),
                                    "duration_s": round(time.time() - t0, 2)})
        return result

    # -- implementations -----------------------------------------------
    def _list_files(self, pattern: str) -> dict:
        hits = []
        for root, dirs, files in os.walk(self.repo_root):
            dirs[:] = [d for d in dirs if d not in (".git", ".venv", "__pycache__", ".agent_loop")]
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), self.repo_root).replace(os.sep, "/")
                if fnmatch.fnmatch(rel, pattern or "**/*.py"):
                    hits.append(rel)
        return {"ok": True, "output": "\n".join(sorted(hits)[:500]) or "(no matches)"}

    def _read(self, rel: str) -> dict:
        p = self._resolve(rel)
        if not p.is_file():
            return {"ok": False, "output": "not a file: %s" % rel}
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_READ_CHARS:
            text = text[:MAX_READ_CHARS] + "\n...[truncated]..."
        return {"ok": True, "output": text}

    def _write(self, rel: str, content: str) -> dict:
        p = self._resolve(rel)
        if ".git" in p.parts:
            raise ToolRefusal("refusing to write inside .git")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"ok": True, "output": "wrote %s (%d chars)" % (rel, len(content))}

    def _edit(self, rel: str, old: str, new: str) -> dict:
        if not old:
            return {"ok": False, "output": "old_string must be non-empty"}
        p = self._resolve(rel)
        text = p.read_text(encoding="utf-8", errors="replace")
        if old not in text:
            return {"ok": False, "output": "old_string not found in %s" % rel}
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        return {"ok": True, "output": "edited %s (1 occurrence replaced)" % rel}

    def _shell(self, command: str, timeout_s=None) -> dict:
        if not command or not command.strip():
            return {"ok": False, "output": "empty command"}
        low = command.lower()
        for pat in _BLOCKED_RM:
            if pat in low:
                raise ToolRefusal("destructive shell pattern blocked: %r" % pat)
        if low.strip().startswith("git"):
            self._git_guard(command)
        shell = os.name == "nt"
        try:
            proc = subprocess.run(
                command if shell else ["sh", "-c", command],
                cwd=str(self.repo_root), capture_output=True, text=True,
                timeout=timeout_s or self.command_timeout_s, shell=shell)
            out = (proc.stdout or "") + (proc.stderr or "")
            return {"ok": proc.returncode == 0, "returncode": proc.returncode,
                    "output": out[-MAX_OUTPUT_CHARS:] or "(no output)"}
        except subprocess.TimeoutExpired:
            return {"ok": False, "returncode": -1, "output": "TIMEOUT"}

    def _git_status(self) -> str:
        r = self._run(["git", "status", "--short", "--branch"])
        return r["output"] or "(clean)"

    def _git_diff(self, stat_only: bool) -> str:
        args = ["git", "diff", "--stat"] if stat_only else ["git", "diff", "--stat", "--patch", "--", "."]
        r = self._run(["git", "diff", "--stat"])
        out = r["output"]
        if not stat_only:
            r2 = self._run(["git", "diff", "--", "."])
            out = (out or "") + "\n" + (r2["output"] or "")
        return (out or "(no diff)")[-MAX_OUTPUT_CHARS:]

    def _artifacts(self, pattern: str) -> dict:
        hits = []
        for sub in ("experiments/results", "experiments/runs"):
            d = self.repo_root / sub
            if d.is_dir():
                for f in sorted(d.glob(pattern or "*.json")):
                    try:
                        hits.append("%s (%d bytes)" % (f.relative_to(self.repo_root), f.stat().st_size))
                    except OSError:
                        pass
        return {"ok": True, "output": "\n".join(hits) or "(no artifacts)"}

    # -- detached jobs (manager/compute decoupling) --------------------
    def _job_manager(self):
        from lab.jobs import JobManager
        return JobManager(self.repo_root)

    def _queue_job(self, args: dict) -> dict:
        cmd = args.get("command")
        if not isinstance(cmd, list) or not cmd or not all(isinstance(a, str) for a in cmd):
            return {"ok": False, "output": "command must be a non-empty argv list"}
        exe = Path(cmd[0])
        if exe.is_absolute():
            if (exe.name.lower() not in ("python.exe", "pythonw.exe", "python")
                    and exe.resolve() != Path(sys.executable).resolve()):
                return {"ok": False, "output": "absolute executables limited to python"}
        else:
            # repo-relative executable/script: must stay jailed
            try:
                (self.repo_root / cmd[0]).resolve().relative_to(self.repo_root)
            except ValueError:
                return {"ok": False, "output": "command escapes repository root"}
        try:
            timeout_s = float(args.get("timeout_s", 3600.0))
        except (TypeError, ValueError):
            return {"ok": False, "output": "timeout_s must be numeric"}
        timeout_s = min(max(timeout_s, 1.0), 86400.0)
        try:
            rec = self._job_manager().submit(
                cmd, str(args.get("experiment_id", "adhoc")),
                seed=int(args.get("seed", 0)), timeout_s=timeout_s,
                mem_mb=args.get("mem_mb"),
                heartbeat_rel=str(args.get("heartbeat", "") or ""),
                result_rel=str(args.get("result", "") or ""),
                artifact_rel=str(args.get("artifact", "") or ""),
                checkpoint_rel=str(args.get("checkpoint", "") or ""))
        except ValueError as e:
            return {"ok": False, "output": "refused: %s" % e}
        self._job_manager().start_due()
        rec = self._job_manager()._read(rec["job_id"]) or rec
        return {"ok": True, "output": "queued job %s status=%s (poll with poll_job)" % (
            rec["job_id"], rec.get("status"))}

    def _poll_job(self, job_id: str) -> dict:
        if not job_id:
            return {"ok": False, "output": "job_id required"}
        jm = self._job_manager()
        rec = jm.poll(str(job_id))
        if rec is None:
            return {"ok": False, "output": "unknown job: %s" % job_id}
        tail = ""
        try:
            with open(rec.get("log_path", ""), "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 2000))
                tail = f.read().decode("utf-8", "replace")
        except OSError:
            pass
        started = rec.get("start_time") or int(time.time())
        summary = {
            "job_id": rec["job_id"], "status": rec.get("status"),
            "elapsed_s": int(time.time()) - started,
            "pid": rec.get("pid"), "heartbeat": rec.get("heartbeat", {}),
            "progress": rec.get("progress", {}),
            "exit_code": rec.get("exit_code"),
            "failure_reason": rec.get("failure_reason", ""),
            "result": rec.get("result_path", ""),
            "log_tail": tail[-1500:],
        }
        return {"ok": True, "output": json.dumps(summary, indent=2)[:6000]}


def _redact(args: dict) -> dict:
    red = dict(args)
    if "content" in red and isinstance(red["content"], str) and len(red["content"]) > 200:
        red["content"] = red["content"][:200] + "...[truncated %d chars]" % len(red["content"])
    return red
