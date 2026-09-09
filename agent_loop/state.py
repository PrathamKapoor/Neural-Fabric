"""Persistent loop state: survives API failures, crashes, and restarts.

Layout inside <repo>/.agent_loop/:
  objective.md            the original user objective (never rewritten by loop)
  state.json              live counters + current status (rewritten each step)
  manager_decision.json   latest manager decision (pointer copy)
  worker_report.json      latest worker report (pointer copy)
  history/iter_NNNN_*.json  immutable per-iteration records (never overwritten)
  logs/                   free-form run logs
  STOP                    presence of this file requests a graceful stop
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class LoopState:
    def __init__(self, state_dir: str | Path):
        self.dir = Path(state_dir)
        self.history_dir = self.dir / "history"
        self.logs_dir = self.dir / "logs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._state = self._load() or {
            "iteration": 0,
            "status": "initialized",
            "consecutive_failures": 0,
            "api_failures": 0,
            "started_at": int(time.time()),
        }

    # -- objective -----------------------------------------------------
    def set_objective(self, objective: str):
        (self.dir / "objective.md").write_text(objective.strip() + "\n", encoding="utf-8")

    def get_objective(self) -> str:
        p = self.dir / "objective.md"
        return p.read_text(encoding="utf-8").strip() if p.exists() else ""

    # -- counters ------------------------------------------------------
    @property
    def iteration(self) -> int:
        return int(self._state.get("iteration", 0))

    @property
    def consecutive_failures(self) -> int:
        return int(self._state.get("consecutive_failures", 0))

    @property
    def api_failures(self) -> int:
        return int(self._state.get("api_failures", 0))

    def bump_iteration(self) -> int:
        self._state["iteration"] = self.iteration + 1
        self._save()
        return self.iteration

    def note_success(self):
        self._state["consecutive_failures"] = 0
        self._save()

    def note_failure(self):
        self._state["consecutive_failures"] = self.consecutive_failures + 1
        self._save()

    def note_api_failure(self):
        self._state["api_failures"] = self.api_failures + 1
        self._save()

    def set_status(self, status: str):
        self._state["status"] = status
        self._save()

    # -- autonomous session (24h mode) -----------------------------------
    def ensure_session(self, objective: str, session_hours: float) -> dict:
        """Create the session once; resume NEVER resets the clock.

        Returns {"session_id", "started_at", "deadline_at", "remaining_s"}.
        deadline_at == 0 means no deadline (session_hours <= 0).
        """
        import uuid as _uuid
        now = int(time.time())
        if not self._state.get("session_id"):
            self._state["session_id"] = _uuid.uuid4().hex[:12]
            self._state["session_started_at"] = now
            self._state["deadline_at"] = (
                now + int(session_hours * 3600) if session_hours > 0 else 0)
            self._state.setdefault("research", {"completed_tasks": [],
                                                "discoveries": [],
                                                "unresolved": []})
            self._state.setdefault("totals", {"manager_s": 0.0, "worker_s": 0.0,
                                              "tool_calls": 0, "model_turns": 0})
            self._save()
        return self.session_info()

    def session_info(self) -> dict:
        return {"session_id": self._state.get("session_id", ""),
                "started_at": self._state.get("session_started_at", 0),
                "deadline_at": self._state.get("deadline_at", 0),
                "remaining_s": self.session_remaining()}

    def session_remaining(self) -> float | None:
        deadline = self._state.get("deadline_at", 0)
        if not deadline:
            return None
        return float(deadline - time.time())

    def add_totals(self, manager_s: float = 0.0, worker_s: float = 0.0,
                   tool_calls: int = 0, model_turns: int = 0):
        t = self._state.setdefault("totals", {"manager_s": 0.0, "worker_s": 0.0,
                                              "tool_calls": 0, "model_turns": 0})
        t["manager_s"] = round(t.get("manager_s", 0.0) + manager_s, 1)
        t["worker_s"] = round(t.get("worker_s", 0.0) + worker_s, 1)
        t["tool_calls"] = t.get("tool_calls", 0) + tool_calls
        t["model_turns"] = t.get("model_turns", 0) + model_turns
        self._save()

    def update_research(self, decision: dict, report: dict):
        """Fold one iteration into the durable research state (capped)."""
        r = self._state.setdefault("research", {"completed_tasks": [],
                                                "discoveries": [],
                                                "unresolved": []})
        task = str(decision.get("next_task", ""))[:300]
        if task:
            entry = "%s -> worker:%s" % (task, report.get("status", "?"))
            if entry not in r["completed_tasks"]:
                r["completed_tasks"].append(entry)
        for key, src in (("discoveries", [report.get("recommendation", "")]),
                         ("unresolved", decision.get("gaps", [])
                          + report.get("problems", [])[:2])):
            for item in src:
                item = str(item)[:300]
                if item and item not in r[key]:
                    r[key].append(item)
        for key in r:
            r[key] = r[key][-30:]
        self._save()

    def research_summary(self) -> str:
        r = self._state.get("research", {})
        lines = []
        done = r.get("completed_tasks", [])[-8:]
        if done:
            lines.append("COMPLETED TASKS:")
            lines += ["- " + d[:160] for d in done]
        disc = r.get("discoveries", [])[-5:]
        if disc:
            lines.append("DISCOVERIES/NOTES:")
            lines += ["- " + d[:160] for d in disc]
        unres = r.get("unresolved", [])[-5:]
        if unres:
            lines.append("UNRESOLVED:")
            lines += ["- " + d[:160] for d in unres]
        pending = self._state.get("pending_replan")
        if pending and pending.get("reasons"):
            lines.append("REPLAN REQUIRED (stagnation/drift, do not retry):")
            lines += ["- " + str(x)[:160] for x in pending["reasons"][:4]]
        return "\n".join(lines) or "(no research history yet)"

    # -- task state machine (v2: PLANNED -> ... -> ACCEPTED/REPAIR/REPLAN) --
    def set_task_state(self, iteration: int, task_state: str, note: str = ""):
        """Record an explicit research-state transition (persisted)."""
        tasks = self._state.setdefault("tasks", {})
        key = "iter_%04d" % iteration
        entry = tasks.setdefault(key, {"history": []})
        entry["state"] = task_state
        entry["updated_at"] = int(time.time())
        if note:
            entry["history"].append({"t": int(time.time()),
                                    "state": task_state,
                                    "note": str(note)[:300]})
            entry["history"] = entry["history"][-20:]
        # keep only recent tasks bounded
        keys = sorted(tasks)
        for old in keys[:-30]:
            tasks.pop(old, None)
        self._save()

    def get_task_state(self, iteration: int) -> str:
        return self._state.get("tasks", {}).get(
            "iter_%04d" % iteration, {}).get("state", "")

    def signal_replan(self, reasons: list):
        """Force REPLAN: the next manager decision must replan, not retry."""
        self._state["pending_replan"] = {
            "reasons": [str(r)[:300] for r in reasons][:6],
            "at": int(time.time())}
        self._save()

    def take_replan_signal(self):
        """Consume-and-clear the pending REPLAN signal (manager turn)."""
        sig = self._state.pop("pending_replan", None)
        if sig is not None:
            self._save()
        return sig

    def save_final_report(self, text: str) -> str:
        path = self.dir / "final_report.md"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def save_progress(self, iteration: int, snapshot: dict) -> str:
        """Periodic progress snapshot (overwrites progress.json).

        progress.json is the crash-safe operator view: elapsed/remaining,
        current + last tasks, latest evidence/tests/failures, and the next
        direction. An immutable copy per iteration also goes to history/.
        """
        path = self.dir / "progress.json"
        path.write_text(json.dumps(snapshot, indent=2)[:100000],
                        encoding="utf-8")
        hist = self.history_dir / ("iter_%04d_progress.json" % iteration)
        if not hist.exists():
            hist.write_text(json.dumps(snapshot, indent=2)[:100000],
                            encoding="utf-8")
        return str(path)

    # -- records -------------------------------------------------------
    def save_manager_decision(self, iteration: int, decision: dict):
        (self.dir / "manager_decision.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8")
        self._append_history(iteration, "manager_decision", decision)

    def save_worker_report(self, iteration: int, report: dict):
        (self.dir / "worker_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        self._append_history(iteration, "worker_report", report)

    def save_evidence(self, iteration: int, evidence: dict):
        self._append_history(iteration, "evidence", evidence)

    def save_verification(self, iteration: int, record: dict):
        """Immutable per-iteration tester verdict + gate findings."""
        self._append_history(iteration, "verification", record)

    def save_analysis(self, iteration: int, record: dict):
        """Immutable per-iteration analyst/critic record."""
        self._append_history(iteration, "analysis", record)

    def save_invocations(self, iteration: int, phase: str, invocations: list):
        """Persist per-attempt model invocation records (incl. fallbacks).

        Full records go to immutable history files; a capped tail is kept in
        state.json so fallback history survives restarts and stays queryable.
        """
        self._append_history(iteration, "invocations_%s" % phase,
                             {"iteration": iteration, "phase": phase,
                              "invocations": invocations})
        tail = self._state.get("invocations_tail", [])
        tail.extend(invocations)
        self._state["invocations_tail"] = tail[-50:]
        self._state["fallback_count"] = sum(
            1 for i in tail if isinstance(i, dict) and i.get("fallback_from"))
        self._save()

    def recent_invocations(self, limit: int = 10) -> list:
        tail = self._state.get("invocations_tail", [])
        return [i for i in tail if isinstance(i, dict)][-limit:]

    def history(self) -> list:
        """Concise per-iteration summaries for the manager's context window."""
        items = []
        for f in sorted(self.history_dir.glob("iter_*_summary.json")):
            try:
                items.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return items

    def save_iteration_summary(self, iteration: int, summary: dict):
        path = self.history_dir / ("iter_%04d_summary.json" % iteration)
        if path.exists():
            # Never overwrite history: suffix a duplicate marker.
            path = self.history_dir / ("iter_%04d_summary_dup%d.json"
                                       % (iteration, int(time.time())))
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _append_history(self, iteration: int, kind: str, payload: dict):
        path = self.history_dir / ("iter_%04d_%s.json" % (iteration, kind))
        if path.exists():
            path = self.history_dir / ("iter_%04d_%s_%d.json"
                                       % (iteration, kind, int(time.time())))
        path.write_text(json.dumps(payload, indent=2)[:200000], encoding="utf-8")

    # -- stop file -----------------------------------------------------
    def stop_requested(self) -> bool:
        return (self.dir / "STOP").exists()

    def request_stop(self, reason: str = ""):
        (self.dir / "STOP").write_text(
            "stop requested: %s\n" % reason, encoding="utf-8")

    # -- persistence ---------------------------------------------------
    def _load(self) -> dict | None:
        p = self.dir / "state.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            backup = self.dir / "state.json.corrupt"
            try:
                p.rename(backup)
            except OSError:
                pass
            return None

    def _save(self):
        self._state["updated_at"] = int(time.time())
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        tmp.replace(self.dir / "state.json")
