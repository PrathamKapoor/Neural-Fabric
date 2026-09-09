"""Persistent detached experiment jobs (stdlib only, Windows-verified).

Separation of timeouts (architectural invariant):
- LLM timeout      (REQUEST_TIMEOUT_S, ~minutes): a single model call.
- WORKER timeout   (WORKER_TASK_TIMEOUT_SECONDS): one coherent work unit.
- EXPERIMENT timeout (job deadline, hours allowed): detached compute that
  MUST survive LLM/worker termination. Never conflate the three.

Lifecycle: QUEUED -> STARTING -> RUNNING -> CHECKPOINTING ->
COMPLETED | FAILED | TIMED_OUT | CANCELLED. Crash recovery: RUNNING with a
dead PID and no result file -> ORPHANED (requeueable); with a result file
-> COMPLETED (recovered). UNKNOWN only when the OS refuses to answer.

Windows notes: detached spawn uses DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP
with redirected stdio; liveness via OpenProcess; tree kill via taskkill /T.
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

STATUSES = ("QUEUED", "STARTING", "RUNNING", "CHECKPOINTING", "COMPLETED",
            "FAILED", "TIMED_OUT", "CANCELLED", "ORPHANED", "RECOVERING",
            "UNKNOWN")
TERMINAL = ("COMPLETED", "FAILED", "TIMED_OUT", "CANCELLED")

_DEFAULT_MEM_MB = 512
_MIN_DISK_BYTES = 1 << 30  # 1 GiB floor before starting new compute


# ---------------------------------------------------------------- resources
def cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def memory() -> dict:
    """Free/total RAM in MB (Windows API, stdlib ctypes)."""
    try:
        class _Stat(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = _Stat()
        st.dwLength = ctypes.sizeof(_Stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            mb = 1 << 20
            return {"total_mb": st.ullTotalPhys // mb,
                    "free_mb": st.ullAvailPhys // mb}
    except Exception:
        pass
    return {"total_mb": 0, "free_mb": 0}


def disk_free(repo_root: str | Path) -> int:
    try:
        return shutil.disk_usage(str(repo_root)).free
    except OSError:
        return 0


def gpu_info() -> dict:
    """nvidia-smi probe; absent GPU is a fact, not an error."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return {"present": False, "gpus": []}
    if proc.returncode != 0:
        return {"present": False, "gpus": []}
    gpus = [l.strip() for l in (proc.stdout or "").splitlines() if l.strip()]
    return {"present": bool(gpus), "gpus": gpus}


def resources(repo_root: str | Path) -> dict:
    mem = memory()
    return {"cpu": cpu_count(), "ram_mb": mem,
            "disk_free_bytes": disk_free(repo_root),
            "gpu": gpu_info()}


# ------------------------------------------------------------------ process
def _pid_alive(pid: int) -> bool | None:
    """True while RUNNING, False when terminated, None when unqueryable.

    OpenProcess succeeding does NOT mean alive on Windows (it also opens
    terminated-but-unreaped processes); only GetExitCodeProcess == STILL_ACTIVE
    counts as running.
    """
    if not pid:
        return False
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        code = ctypes.c_ulong(0)
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        if not ok:
            return None
        return bool(code.value == STILL_ACTIVE)
    except Exception:
        return None


STILL_ACTIVE = 259

# Module-level handle registry: Popen objects must be wait()ed to release
# OS handles; the creating JobManager may be long gone by completion time.
_LIVE: dict = {}


def _track(proc) -> None:
    try:
        _LIVE[int(proc.pid)] = proc
    except Exception:
        pass


def _reap(pid: int):
    """wait() a tracked child (releases its handle); None if untracked."""
    proc = _LIVE.pop(int(pid), None) if pid else None
    if proc is None:
        return None
    try:
        return proc.wait(timeout=10)
    except Exception:
        return None


def _pid_exit_code(pid: int) -> int | None:
    """Best-effort exit code (None while running or unreadable)."""
    if not pid:
        return None
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return None
        code = ctypes.c_ulong(0)
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        if not ok or code.value == STILL_ACTIVE:
            return None
        return int(code.value)
    except Exception:
        return None


def _tree_kill(pid: int) -> bool:
    try:
        proc = subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                              capture_output=True, timeout=60)
        return proc.returncode == 0
    except Exception:
        return False


# ------------------------------------------------------------------ records
def _now() -> int:
    return int(time.time())


class JobManager:
    """Filesystem-backed scheduler + supervisor for detached experiments."""

    def __init__(self, repo_root: str | Path, jobs_dir: str | Path | None = None,
                 max_concurrent: int | None = None,
                 default_mem_mb: int = _DEFAULT_MEM_MB):
        self.repo_root = Path(repo_root).resolve()
        self.dir = Path(jobs_dir) if jobs_dir else self.repo_root / ".agent_loop" / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent or max(1, min(cpu_count(), 4))
        self.default_mem_mb = default_mem_mb

    # -- paths ---------------------------------------------------------
    def _path(self, job_id: str) -> Path:
        safe = "".join(c for c in job_id if c.isalnum() or c in ("-", "_"))[:64]
        return self.dir / ("%s.json" % (safe or "job"))

    def _read(self, job_id: str) -> dict | None:
        try:
            return json.loads(self._path(job_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write(self, rec: dict):
        p = self._path(rec["job_id"])
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=2)[:200000], encoding="utf-8")
        tmp.replace(p)

    def _log(self, rec: dict, event: str):
        rec.setdefault("history", []).append({"t": _now(), "event": event[:300]})
        rec["history"] = rec["history"][-100:]

    # -- submit / schedule ----------------------------------------------
    def _jail(self, path: str) -> Path:
        p = (self.repo_root / path).resolve()
        try:
            p.relative_to(self.repo_root)
        except ValueError:
            raise ValueError("path escapes repository: %r" % path)
        return p

    def submit(self, command: list, experiment_id: str, seed: int = 0,
               timeout_s: float = 3600.0, mem_mb: int | None = None,
               heartbeat_rel: str = "", result_rel: str = "",
               artifact_rel: str = "", checkpoint_rel: str = "",
               config: dict | None = None, group: str = "") -> dict:
        """Queue a job. command MUST be an argv list (no shell injection)."""
        if not isinstance(command, (list, tuple)) or not command:
            raise ValueError("command must be a non-empty argv list")
        if any(not isinstance(a, str) for a in command):
            raise ValueError("argv entries must be strings")
        for arg in command[1:]:
            if ("/" in arg or "\\" in arg or arg.endswith(
                    (".py", ".json", ".txt", ".md", ".log", ".tmp",
                     ".ckpt", ".pt", ".pth"))):
                cand = Path(arg)
                if not cand.is_absolute():
                    try:
                        (self.repo_root / arg).resolve().relative_to(self.repo_root)
                    except ValueError:
                        raise ValueError("argv path escapes repository: %r" % arg)
        job_id = uuid.uuid4().hex[:12]
        hb = str(self._jail(heartbeat_rel)) if heartbeat_rel else ""
        rec = {
            "job_id": job_id, "experiment_id": experiment_id,
            "project": self.repo_root.name,
            "git_commit": _git_commit(self.repo_root),
            "created_at": time.time(),
            "configuration": dict(config or {}), "seed": seed,
            "command": list(command),
            "working_directory": str(self.repo_root),
            "status": "QUEUED", "start_time": 0,
            "expected_duration": float(timeout_s),
            "deadline": 0, "pid": 0,
            "checkpoint_path": str(self._jail(checkpoint_rel)) if checkpoint_rel else "",
            "artifact_path": str(self._jail(artifact_rel)) if artifact_rel else "",
            "log_path": str(self.dir / ("%s.log" % job_id)),
            "result_path": str(self._jail(result_rel)) if result_rel else "",
            "heartbeat_path": hb,
            "heartbeat": {}, "progress": {}, "exit_code": None,
            "failure_reason": "", "queue_reason": "awaiting scheduler",
            "mem_mb": int(mem_mb or self.default_mem_mb),
            "group": group, "history": [{"t": _now(), "event": "QUEUED"}],
        }
        self._write(rec)
        return rec

    def running_jobs(self) -> list:
        return [r for r in self.list_jobs() if r.get("status") == "RUNNING"]

    def list_jobs(self) -> list:
        out = []
        for f in sorted(self.dir.glob("*.json")):
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return out

    def start_due(self) -> list:
        """Start queued jobs while resources allow. Returns started records."""
        started = []
        mem = self._memory_robust()
        queued = sorted(
            (r for r in self.list_jobs() if r.get("status") == "QUEUED"),
            key=lambda r: (r.get("created_at", 0), r["job_id"]))
        for rec in queued:
            if len(self.running_jobs()) >= self.max_concurrent:
                rec["queue_reason"] = "max_concurrent=%d reached" % self.max_concurrent
                self._write(rec)
                continue
            if mem.get("probed", True) and mem.get("free_mb", 0) < rec.get("mem_mb", 0):
                rec["queue_reason"] = "insufficient RAM for mem_mb=%d" % rec.get("mem_mb", 0)
                self._write(rec)
                continue
            if disk_free(self.repo_root) < _MIN_DISK_BYTES:
                rec["queue_reason"] = "disk floor reached"
                self._write(rec)
                continue
            if mem.get("probed") is False:
                rec["queue_reason"] = "RAM probe failed; proceeding (concurrency capped)"
            started.append(self._start(rec))
        return started

    @staticmethod
    def _memory_robust(retries: int = 3) -> dict:
        """RAM probe with retries; fail-open (marked) if the OS won't answer.

        A zero reading positively indicates probe failure (no machine has 0
        RAM); blocking all compute on it would deadlock the scheduler, so the
        concurrency cap remains the backstop instead.
        """
        for _ in range(max(1, retries)):
            mem = memory()
            if mem.get("total_mb", 0) > 0:
                mem["probed"] = True
                return mem
            time.sleep(0.5)
        mem["probed"] = False
        return mem

    def _start(self, rec: dict) -> dict:
        rec["status"] = "STARTING"
        self._log(rec, "STARTING")
        self._write(rec)
        DETACHED = getattr(subprocess, "DETACHED_PROCESS", 0)
        NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            logf = open(rec["log_path"], "ab", buffering=0)
        except OSError as e:
            return self._fail(rec, "cannot open log: %s" % e)
        try:
            proc = subprocess.Popen(
                rec["command"], cwd=str(self.repo_root),
                stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
                creationflags=DETACHED | NEW_GROUP, close_fds=True)
            rec["pid"] = proc.pid
            _track(proc)
        except (OSError, ValueError) as e:
            try:
                logf.close()
            except OSError:
                pass
            return self._fail(rec, "spawn failed: %s" % e)
        try:
            logf.close()  # child keeps its own copy; parent must not leak
        except OSError:
            pass
        rec["status"] = "RUNNING"
        rec["start_time"] = _now()
        rec["deadline"] = rec["start_time"] + float(rec.get("expected_duration", 3600))
        self._log(rec, "RUNNING pid=%s" % rec["pid"])
        self._write(rec)
        return rec

    # -- poll / supervise -----------------------------------------------
    def _read_heartbeat(self, rec: dict):
        hp = rec.get("heartbeat_path", "")
        if not hp:
            return
        try:
            text = Path(hp).read_text(encoding="utf-8")[-4000:]
            data = json.loads(text.strip().splitlines()[-1])
            if isinstance(data, dict):
                rec["heartbeat"] = {k: str(v)[:200] for k, v in list(data.items())[:12]}
                rec["progress"] = {k: rec["heartbeat"].get(k, "")
                                    for k in ("step", "epoch", "best", "checkpoint") if k in rec["heartbeat"]}
        except (OSError, ValueError, IndexError):
            pass

    def poll(self, job_id: str) -> dict | None:
        """Refresh one job: liveness, heartbeat, deadline, result pickup."""
        rec = self._read(job_id)
        if rec is None or rec.get("status") in TERMINAL:
            return rec
        status = rec.get("status", "QUEUED")
        if status in ("QUEUED", "STARTING"):
            return rec
        self._read_heartbeat(rec)
        alive = _pid_alive(rec.get("pid", 0))
        if alive is None:
            rec["status"] = "UNKNOWN"
            self._log(rec, "UNKNOWN: OS refused PID query")
            self._write(rec)
            return rec
        if not alive:
            code = _pid_exit_code(rec.get("pid", 0))
            _reap(rec.get("pid", 0))
            if self._result_present(rec):
                rec["status"] = "COMPLETED"
                rec["exit_code"] = code
                self._log(rec, "COMPLETED (result present, pid gone)")
            else:
                rec["status"] = "ORPHANED"
                rec["exit_code"] = code
                rec["failure_reason"] = "process gone with no result artifact"
                self._log(rec, "ORPHANED")
            self._write(rec)
            return rec
        if _now() > float(rec.get("deadline", 0) or 0):
            _tree_kill(rec.get("pid", 0))
            _reap(rec.get("pid", 0))
            rec["status"] = "TIMED_OUT"
            rec["failure_reason"] = "deadline exceeded"
            self._log(rec, "TIMED_OUT (killed)")
            self._write(rec)
            return rec
        self._write(rec)
        return rec

    def poll_all(self) -> list:
        return [self.poll(r["job_id"]) for r in self.list_jobs()
                if r.get("status") not in TERMINAL]

    def _result_present(self, rec: dict) -> bool:
        rp = rec.get("result_path", "")
        try:
            return bool(rp) and Path(rp).is_file() and Path(rp).stat().st_size > 0
        except OSError:
            return False

    # -- control ----------------------------------------------------------
    def _fail(self, rec: dict, reason: str) -> dict:
        rec["status"] = "FAILED"
        rec["failure_reason"] = reason[:500]
        self._log(rec, "FAILED: %s" % reason[:200])
        self._write(rec)
        return rec

    def cancel(self, job_id: str) -> dict | None:
        rec = self._read(job_id)
        if rec is None or rec.get("status") in TERMINAL:
            return rec
        if rec.get("status") == "QUEUED":
            rec["status"] = "CANCELLED"
            self._log(rec, "CANCELLED while queued")
            self._write(rec)
            return rec
        _tree_kill(rec.get("pid", 0))
        _reap(rec.get("pid", 0))
        rec["status"] = "CANCELLED"
        self._log(rec, "CANCELLED (killed)")
        self._write(rec)
        return rec

    def requeue(self, job_id: str) -> dict | None:
        """Return an ORPHANED/FAILED job to QUEUED (operator decision)."""
        rec = self._read(job_id)
        if rec is None or rec.get("status") not in ("ORPHANED", "FAILED", "TIMED_OUT"):
            return rec
        rec.update(status="QUEUED", pid=0, exit_code=None, failure_reason="",
                   queue_reason="requeued by operator")
        self._log(rec, "QUEUED (requeue)")
        self._write(rec)
        return rec

    def recover(self) -> list:
        """Startup reconciliation after crash/restart. Returns changed records."""
        changed = []
        for rec in self.list_jobs():
            if rec.get("status") in ("RUNNING", "STARTING", "CHECKPOINTING",
                                     "RECOVERING", "UNKNOWN"):
                before = rec.get("status")
                rec["status"] = "RECOVERING"
                self._write(rec)
                out = self.poll(rec["job_id"])
                if out is not None and out.get("status") != before:
                    changed.append(out)
        return changed

    # -- helpers for experiments -------------------------------------------
    @staticmethod
    def heartbeat(path: str | Path, **data):
        """Atomic heartbeat write for experiment scripts (JSON line)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"t": _now(), **{k: str(v)[:300] for k, v in data.items()}})
        tmp = p.with_suffix(".tmp")
        # heartbeats append (history) unless file is huge; rotate past 1MB
        try:
            if p.exists() and p.stat().st_size > (1 << 20):
                p.write_text(line + "\n", encoding="utf-8")
                return str(p)
            with open(p, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            try:
                tmp.write_text(line + "\n", encoding="utf-8")
                tmp.replace(p)
            except OSError:
                pass
        return str(p)

    @staticmethod
    def checkpoint_save(path: str | Path, state: dict) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(state)[:2000000], encoding="utf-8")
        tmp.replace(p)
        return str(p)

    @staticmethod
    def checkpoint_load(path: str | Path) -> dict | None:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None


def _git_commit(repo_root: Path) -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_root),
                              capture_output=True, text=True, timeout=20)
        out = (proc.stdout or "").strip()
        return out if out else "unknown"
    except Exception:
        return "unknown"
