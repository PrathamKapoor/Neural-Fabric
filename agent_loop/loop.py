"""Orchestration: USER OBJECTIVE -> MANAGER -> WORKER -> EVIDENCE -> MANAGER ...

Each iteration:
  1. Manager chain emits ONE structured decision (task + expected + validation).
     Primary: Nemotron 3 Ultra free (OpenRouter, constrained free endpoint -
     one coherent work unit per call, gentle retries); infra failure only ->
     Kimi K3 (NVIDIA) fallback. The manager never touches files and never
     receives tools.
  2. Worker chain executes it as ONE COHERENT work unit. Primary: headless
     AGY CLI (single run per task, AGY's own tool loop); infra failure only
     -> Nemotron Super (OpenRouter, real repository tools) -> DeepSeek V4
     Flash (NVIDIA). Bounded by turn / tool-call / wall-clock budgets.
  3. Evidence collector snapshots objective repo facts.
  4. History (decisions, reports, evidence, invocation/fallback records,
     per-phase durations) is persisted; termination conditions are checked.
  5. Next iteration's manager call receives worker report + evidence and
     judges EXPECTED vs ACTUAL (assessment + next decision in one call).

24-hour mode: an absolute session deadline bounds the whole run. Every
iteration checks remaining time against the minimum quantum for another work
unit; on expiry the loop collects final evidence, asks the manager for a
closing summary when time allows, writes `.agent_loop/final_report.md`, and
terminates. `--resume` continues the SAME session: the deadline is never
reset. Only `--new-session` starts a fresh clock (old state is archived).

A bad TASK result is never an infrastructure failure: no silent fallback.
The result flows to evidence -> manager review, which decides.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from .clients import OpenAICompatibleClient
from .config import LoopConfig
from .evidence import (collect_evidence, porcelain_lines,
                        summarize_evidence)
from .prompts import build_manager_messages, build_worker_messages
from .roles import ChainExhausted, RoleChain
from .schemas import (manager_fallback, parse_manager_output,
                      parse_worker_output, worker_fallback)
from .state import LoopState
from .tester import ApiTester, OpenCodeTester, build_tester_messages, combine_verdict, parse_verdict
from .tools import WORKER_TOOLS, ToolExecutor
from .verify import (StagnationTracker, changed_delta, check_claim_strength,
                     check_claims_vs_artifacts, check_scope)

FINAL_SUMMARY_MIN_REMAINING_S = 900.0  # else write state-derived report directly
SESSION_END_BUFFER_S = 90.0  # stop starting new turns below this remaining time


def _content_text(message: dict) -> str:
    c = message.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):  # content-block style providers
        return "".join(b.get("text", "") for b in c if isinstance(b, dict))
    return ""  # None content (e.g. tool-call-only turn) -> no text


def _tool_calls(message: dict) -> list:
    calls = message.get("tool_calls") or []
    norm = []
    for tc in calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
        try:
            args = json.loads(fn.get("arguments", "{}") or "{}")
        except ValueError:
            args = {"_raw": fn.get("arguments", "")}
        norm.append({"id": tc.get("id", ""), "name": fn.get("name", ""),
                     "arguments": args if isinstance(args, dict) else {"_raw": args}})
    return norm


def ensure_utf8_stdio() -> None:
    """Force UTF-8 on stdio so non-ASCII model text can never crash logging.

    On Windows the console/redirection encoding is often cp1252: a single
    Unicode character (e.g. U+2265 from a manager decision) then kills the
    whole run inside a UI print. This must run once at process startup.
    """
    import sys as _sys
    for _stream in ("stdout", "stderr"):
        try:
            getattr(_sys, _stream).reconfigure(encoding="utf-8",
                                               errors="replace")
        except Exception:
            pass


class AgentLoop:
    def __init__(self, config: LoopConfig, manager: RoleChain, worker: RoleChain,
                 state: LoopState | None = None, ui=None):
        if manager.role != "manager":
            raise ValueError("manager chain must have role='manager'")
        if worker.role != "worker":
            raise ValueError("worker chain must have role='worker'")
        # Structural isolation: no worker client may appear in the manager
        # chain and no manager client may appear in the worker chain (all
        # fallback levels checked - AGY can never manage, Kimi never executes).
        worker_clients = [worker.primary] + list(worker.fallbacks)
        manager_clients = [manager.primary] + list(manager.fallbacks)
        for wc in worker_clients:
            if wc in manager_clients:
                raise ValueError("chain cross-contamination: worker client in manager chain")
        for mc in manager_clients:
            if mc is worker.primary:
                raise ValueError("chain cross-contamination: manager client in worker chain")
        self.cfg = config
        self.manager = manager
        self.worker = worker
        self.repo_root = str(Path(config.repo_root).resolve())
        self.state = state or LoopState(Path(self.repo_root) / config.state_dir)
        self.ui = ui or _NullUI()
        self.terminated_reason = ""
        self.live_counts = {"manager_calls": 0, "worker_calls": 0,
                            "tool_calls": 0, "fallbacks": 0}

    # -- public --------------------------------------------------------
    def run(self, objective: str) -> dict:
        """Run the loop until COMPLETE / BLOCKED / deadline / limit / stop."""
        if objective and not self.state.get_objective():
            self.state.set_objective(objective)
        objective = self.state.get_objective() or objective
        if not objective.strip():
            raise ValueError("an objective is required (arg, --objective-file, or prompt)")

        session = self.state.ensure_session(objective, self.cfg.session_hours)
        self.ui.banner(objective, self.cfg, session)
        prev_decision, prev_report, prev_evidence = None, None, None

        while True:
            if self.state.stop_requested():
                return self._finish("stopped", "STOP file present - human override.")
            remaining = self.state.session_remaining()
            if remaining is not None and remaining <= 0:
                return self._finalize("Session deadline reached.")
            if self.state.iteration >= self.cfg.max_iterations:
                return self._finish("max_iterations",
                                    "Reached MAX_ITERATIONS=%d." % self.cfg.max_iterations)
            if self.state.consecutive_failures >= self.cfg.max_consecutive_failures:
                return self._finish("failure_threshold",
                                    "Worker failed %d times in a row." % self.state.consecutive_failures)
            if self.state.api_failures >= self.cfg.max_api_failures:
                return self._finish("api_failure_threshold",
                                    "Too many API failures (%d)." % self.state.api_failures)
            if not self._quantum_ok(remaining):
                return self._finalize(
                    "Insufficient remaining time (%.1fh) for another work unit." % (remaining / 3600))

            iter_t0 = time.time()
            iteration = self.state.bump_iteration()
            self.ui.iteration_header(iteration, remaining)

            # 1. MANAGER -> structured decision -------------------------
            try:
                decision, manager_s = self._manager_step(
                    iteration, objective, prev_decision, prev_report, prev_evidence)
            except ChainExhausted as e:
                self.state.note_api_failure()
                self.ui.warn("manager chain exhausted: %s" % e)
                prev_decision, prev_report, prev_evidence = None, None, None
                continue

            self.state.save_manager_decision(iteration, decision)
            self.ui.manager(decision)
            self.state.set_task_state(iteration, "PLANNED",
                                      str(decision.get("next_task", ""))[:200])
            self.state.take_replan_signal()  # seen by this decision; clear it
            if decision["status"] == "complete":
                self.state.update_research(decision, {"status": "done",
                                                      "recommendation": "",
                                                      "problems": []})
                return self._finish("complete", decision.get("reason") or decision.get("assessment", ""))
            if decision["status"] == "blocked":
                return self._finish("blocked", decision.get("reason") or decision.get("assessment", ""))

            # 2. WORKER executes ONE coherent work unit via tools --------
            self.state.set_task_state(iteration, "BUILDING", "worker started")
            pre_changed = porcelain_lines(self.repo_root)
            try:
                report, worker_s, turns, tool_calls = self._worker_step(
                    iteration, objective, decision)
            except ChainExhausted as e:
                self.state.note_api_failure()
                self.ui.warn("worker chain exhausted: %s" % e)
                report = worker_fallback("worker chain exhausted (all providers failed "
                                         "infrastructurally): %s" % e,
                                         decision.get("next_task", ""))
                worker_s, turns, tool_calls = 0.0, 0, 0
            self.ui.worker(report)

            # 3. Objective evidence --------------------------------------
            evidence = collect_evidence(self.repo_root)
            self.state.save_evidence(iteration, evidence)
            self.ui.evidence(summarize_evidence(evidence))

            # 3b. Independent verification (builder -> tester -> verdict) -
            if report.get("tests") or report.get("artifacts"):
                self.state.set_task_state(iteration, "EXPERIMENTING",
                                          "tests/artifacts reported")
            verification = self._verify_step(iteration, decision, report,
                                             evidence, pre_changed)
            report["verification"] = verification
            analysis = self._analyze_step(iteration, decision, report,
                                          evidence, verification)
            report["analysis"] = analysis
            if analysis.get("critic_flags"):
                self.state.update_research(
                    {"next_task": ""},
                    {"status": "done", "recommendation": "",
                     "problems": ["critic: " + f for f in
                                  analysis["critic_flags"][:3]]})
            self.state.save_worker_report(iteration, report)

            # 4. Book-keeping: failures, research, totals, history --------
            if (report.get("status") in ("failed", "blocked")
                    or verification.get("verdict") == "FAIL"):
                self.state.note_failure()
            else:
                self.state.note_success()
            self.state.update_research(decision, report)
            self.state.add_totals(manager_s=manager_s, worker_s=worker_s,
                                  tool_calls=tool_calls, model_turns=turns + 1)
            iter_s = time.time() - iter_t0
            self.state.save_iteration_summary(iteration, {
                "iteration": iteration,
                "manager_status": decision.get("status"),
                "manager_model": decision.get("_model", ""),
                "task": decision.get("next_task", "")[:500],
                "worker_status": report.get("status"),
                "worker_model": report.get("_model", ""),
                "verification": verification.get("verdict", "-"),
                "task_state": verification.get("task_state", "-"),
                "critic": ",".join(analysis.get("critic_flags", []))[:200],
                "gate_findings": list(verification.get("gate_findings", []))[:5],
                "changes": report.get("changes", [])[:10],
                "tests": report.get("tests", [])[:10],
                "problems": report.get("problems", [])[:10],
                "evidence_summary": summarize_evidence(evidence),
                "durations": {"iteration_s": round(iter_s, 1),
                              "manager_s": round(manager_s, 1),
                              "worker_s": round(worker_s, 1),
                              "turns": turns, "tool_calls": tool_calls},
            })
            self.ui.iteration_done(iter_s, manager_s, worker_s, turns, tool_calls)
            self._write_progress(iteration, decision, report, evidence,
                                 iter_s, manager_s, worker_s, turns, tool_calls)
            prev_decision, prev_report, prev_evidence = decision, report, evidence
            self.ui.next_action(decision, report)

    # -- session helpers --------------------------------------------------
    def _write_progress(self, iteration, decision, report, evidence,
                        iter_s, manager_s, worker_s, turns, tool_calls):
        """Crash-safe operator snapshot (deterministic local state, no model)."""
        info = self.state.session_info()
        started = info.get("started_at") or int(time.time())
        remaining = info.get("remaining_s")
        completed = (self.state._state.get("research", {})
                     .get("completed_tasks", []))
        snapshot = {
            "session_id": info.get("session_id", ""),
            "iteration": iteration,
            "timestamp": int(time.time()),
            "elapsed_s": round(time.time() - started, 1),
            "remaining_s": round(remaining, 1) if remaining is not None else None,
            "current_task": str(decision.get("next_task", ""))[:500],
            "manager_status": decision.get("status"),
            "worker_status": report.get("status"),
            "last_completed_task": completed[-1] if completed else "",
            "latest_evidence": summarize_evidence(evidence),
            "tests": list(report.get("tests", []))[:10],
            "problems": list(report.get("problems", []))[:10],
            "consecutive_failures": self.state.consecutive_failures,
            "api_failures": self.state.api_failures,
            "verification": report.get("verification", {}).get("verdict", "-"),
            "task_state": report.get("verification", {}).get("task_state", "-"),
            "next_direction": str(decision.get("reason", "")
                                  or decision.get("status", ""))[:300],
            "durations": {"iteration_s": round(iter_s, 1),
                          "manager_s": round(manager_s, 1),
                          "worker_s": round(worker_s, 1),
                          "turns": turns, "tool_calls": tool_calls},
            "totals": dict(self.state._state.get("totals", {})),
        }
        path = self.state.save_progress(iteration, snapshot)
        self.ui.progress(path, snapshot)

    def _quantum_ok(self, remaining: float | None) -> bool:
        """Is there enough time left to safely start another work unit?

        Gate is deliberately small (end-buffer + ONE model attempt + margin):
        the worker step already clamps itself to (remaining - buffer), and the
        deadline is re-checked every iteration, so a large gate would only
        needlessly block short validation sessions (e.g. --hours 0.1).
        """
        if remaining is None:
            return True
        quantum = SESSION_END_BUFFER_S + self.cfg.request_timeout_s + 60.0
        return remaining >= quantum

    def _finalize(self, reason: str) -> dict:
        """Deadline path: optional manager closing summary + final report."""
        remaining = self.state.session_remaining()
        closing = ""
        if remaining is not None and remaining > FINAL_SUMMARY_MIN_REMAINING_S:
            try:
                closing = self._manager_final_summary()
            except ChainExhausted as e:
                self.ui.warn("final summary unavailable: %s" % e)
        path = self.state.save_final_report(self._render_final_report(reason, closing))
        self.ui.warn("final session report written to %s" % path)
        return self._finish("deadline", reason)

    def _manager_final_summary(self) -> str:
        objective = self.state.get_objective()
        messages = [
            {"role": "system", "content":
             "You are the MANAGER of the Neural Fabric autonomous research loop. "
             "The session time budget is exhausted. Write a concise plain-text "
             "closing summary: what was accomplished, key evidence, what failed, "
             "unresolved questions, and the recommended next direction. "
             "No JSON, no tools, plain text only."},
            {"role": "user", "content":
             "OBJECTIVE:\n%s\n\nRESEARCH STATE:\n%s\n\nRECENT HISTORY:\n%s"
             % (objective, self.state.research_summary(),
                json.dumps(self.state.history()[-5:], indent=2)[:4000])},
        ]
        raw, invocations = self.manager.chat(
            messages, on_call_start=self._call_starter("manager"))
        self._record_invocations(self.state.iteration, "manager_final", invocations)
        return _content_text(raw)[:3000]

    def _render_final_report(self, reason: str, closing: str) -> str:
        info = self.state.session_info()
        totals = self.state._state.get("totals", {})
        invos = self.state.recent_invocations(200)
        models = sorted({(i.get("role"), i.get("provider"), i.get("model"))
                         for i in invos})
        fallbacks = [i for i in invos if i.get("fallback_from")]
        hist = self.state.history()
        fmt_ts = lambda ts: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "-"
        lines = [
            "# Autonomous session final report", "",
            "- session: %s" % info.get("session_id"),
            "- started: %s" % fmt_ts(info.get("started_at")),
            "- ended: %s (%s)" % (fmt_ts(int(time.time())), reason),
            "- iterations: %d" % self.state.iteration,
            "- objective: %s" % self.state.get_objective()[:500], "",
            "## Totals",
            "- manager time: %.1fs | worker time: %.1fs" % (
                totals.get("manager_s", 0), totals.get("worker_s", 0)),
            "- model turns: %d | tool calls: %d" % (
                totals.get("model_turns", 0), totals.get("tool_calls", 0)),
            "- models used: %s" % (
                ", ".join("%s/%s/%s" % m for m in models) or "(none)"),
            "- fallbacks: %d" % len(fallbacks), "",
            "## Research state", self.state.research_summary(), "",
            "## Iteration outcomes",
        ]
        for h in hist[-25:]:
            d = h.get("durations", {}) if isinstance(h, dict) else {}
            lines.append("- iter %s: manager=%s worker=%s turns=%s tools=%s iter=%.1fs | %s" % (
                h.get("iteration"), h.get("manager_status"), h.get("worker_status"),
                d.get("turns", "?"), d.get("tool_calls", "?"),
                d.get("iteration_s", 0), str(h.get("task", ""))[:120]))
        if closing:
            lines += ["", "## Manager closing summary", "", closing]
        lines += ["", "## Evidence trail",
                  "Per-iteration records in `.agent_loop/history/` (decisions, "
                  "reports, evidence snapshots, invocation/duration logs)."]
        return "\n".join(lines) + "\n"

    # -- steps -----------------------------------------------------------
    def _state_summary(self) -> str:
        hist = self.state.history()
        info = self.state.session_info()
        lines = ["iteration=%d failures=%d" % (self.state.iteration,
                                               self.state.consecutive_failures)]
        if info.get("deadline_at"):
            lines.append("session %s remaining=%.1fh (deadline absolute, survives resume)" % (
                info.get("session_id"), max(0.0, (info.get("remaining_s") or 0)) / 3600))
        lines.append("RESEARCH STATE:\n" + self.state.research_summary())
        for h in hist[-5:]:
            lines.append("iter %s: manager=%s worker=%s | %s" % (
                h.get("iteration"), h.get("manager_status"), h.get("worker_status"),
                str(h.get("task", ""))[:120]))
        fb = self.state.recent_invocations(8)
        if fb:
            lines.append("MODEL INVOCATIONS (role/provider/model/attempt/status/duration):")
            for i in fb:
                lines.append("- %s/%s/%s attempt=%s %s %.1fs%s" % (
                    i.get("role"), i.get("provider"), i.get("model"),
                    i.get("attempt"), i.get("status"), i.get("duration_s", 0),
                    (" fallback_from=" + i["fallback_from"]) if i.get("fallback_from") else ""))
        return "\n".join(lines)

    def _project_state(self) -> str:
        try:
            proc = subprocess.run(["git", "status", "--short", "--branch"],
                                  cwd=self.repo_root, capture_output=True,
                                  text=True, timeout=20)
            return "git status:\n" + ((proc.stdout or "").strip() or "(clean)")
        except Exception as e:  # noqa: BLE001
            return "(git unavailable: %s)" % e

    def _call_starter(self, role: str):
        def _cb(provider, model, attempt, is_fallback):
            self.ui.call_start(role, provider, model, attempt, is_fallback,
                               self.cfg.request_timeout_s)
        return _cb

    def _retry_notifier(self, role: str):
        def _cb(provider, model, attempt, wait_s, error):
            self.ui.retry_wait(role, provider, model, attempt, wait_s, error)
        return _cb

    def _record_invocations(self, iteration: int, phase: str, invocations: list):
        dicts = [i.as_dict() for i in invocations]
        self.state.save_invocations(iteration, phase, dicts)
        for i in dicts:
            if i["status"] == "success":
                if i["role"] == "manager":
                    self.live_counts["manager_calls"] += 1
                else:
                    self.live_counts["worker_calls"] += 1
            if i.get("fallback_from"):
                self.live_counts["fallbacks"] += 1
        self.ui.invocations(dicts)

    def _manager_step(self, iteration, objective, prev_decision, prev_report,
                      prev_evidence) -> tuple:
        messages = build_manager_messages(
            objective, self._state_summary(),
            prev_decision, prev_report, prev_evidence,
            caps=(self.cfg.manager_decision_chars,
                  self.cfg.manager_report_chars,
                  self.cfg.manager_evidence_chars))
        t0 = time.time()
        raw, invocations = self.manager.chat(
            messages, force_json=True,
            on_call_start=self._call_starter("manager"),
            on_retry=self._retry_notifier("manager"))
        manager_s = time.time() - t0
        self._record_invocations(iteration, "manager", invocations)
        text = _content_text(raw)
        used_model = next((i.model for i in invocations
                           if i.status == "success"), self.manager.primary.model)
        try:
            decision = parse_manager_output(text)
        except ValueError as e:
            self.ui.warn("malformed manager output (%s); using corrective fallback" % e)
            decision = manager_fallback(str(e)[:300])
        decision["_model"] = used_model
        return decision, manager_s

    def _worker_step(self, iteration: int, objective: str, decision: dict) -> tuple:
        """Returns (report, worker_s, turns, tool_calls). Budgets -> partial."""
        task = decision.get("next_task", "")
        messages = build_worker_messages(
            objective, self._project_state(), task,
            decision.get("expected_result", ""), decision.get("validation", []),
            decision.get("constraints", []),
            scope={"hypothesis": decision.get("hypothesis", ""),
                   "expected_files": decision.get("expected_files", []),
                   "allowed_dirs": decision.get("allowed_dirs", []),
                   "required_evidence": decision.get("required_evidence", []),
                   "success_criteria": decision.get("success_criteria", "")})
        allow_commit = "commit" in task.lower() and not self.cfg.dry_run
        tools = ToolExecutor(self.repo_root, allow_commit=allow_commit)
        worker_msgs = list(messages)
        final_text = ""
        used_models: list = []
        turns, tool_calls = 0, 0
        t_start = time.time()
        task_deadline = t_start + self.cfg.worker_task_timeout_s
        sess_remaining = self.state.session_remaining()
        if sess_remaining is not None:
            task_deadline = min(task_deadline,
                                t_start + max(0.0, sess_remaining - SESSION_END_BUFFER_S))

        def _partial(reason: str) -> tuple:
            rep = worker_fallback(
                "%s (turns=%d tool_calls=%d elapsed=%.0fs)" % (
                    reason, turns, tool_calls, time.time() - t_start), task)
            rep["status"] = "partial"
            rep["recommendation"] = ("Execution budget reached; evidence so far is "
                                     "preserved. Manager should narrow/resume the task.")
            trail = ["%s ok=%s %.1fs" % (t["tool"], t["ok"], t.get("duration_s", 0))
                     for t in tools.invocation_log]
            if trail:
                rep["commands"] = (rep.get("commands", []) + trail)[:30]
            rep["_model"] = used_models[-1] if used_models else ""
            return rep, time.time() - t_start, turns, tool_calls

        for _ in range(self.cfg.max_tool_steps + 1):
            if tool_calls >= self.cfg.worker_max_tool_calls:
                return _partial("worker tool-call budget exhausted "
                                "(WORKER_MAX_TOOL_CALLS=%d)" % self.cfg.worker_max_tool_calls)
            if time.time() >= task_deadline:
                return _partial("worker task time budget exhausted "
                                "(WORKER_TASK_TIMEOUT_SECONDS=%.0f)" % self.cfg.worker_task_timeout_s)
            turns += 1
            raw, invocations = self.worker.chat(
                worker_msgs, tools=WORKER_TOOLS,
                on_call_start=self._call_starter("worker"),
                on_retry=self._retry_notifier("worker"))
            self._record_invocations(iteration, "worker", invocations)
            used_models += [i.model for i in invocations if i.status == "success"]
            calls = _tool_calls(raw)
            text = _content_text(raw)
            via_agy = bool(raw.get("_agy"))  # AGY executes with its own tools
            if text:
                final_text = text
            worker_msgs.append({"role": "assistant",
                                "content": text,
                                **({"tool_calls": raw["tool_calls"]} if calls else {})})
            if not calls:
                break  # worker is answering (should be the JSON report)
            for tc in calls:
                if tool_calls >= self.cfg.worker_max_tool_calls:
                    break
                if time.time() >= task_deadline:
                    break
                res = tools.execute(tc["name"], tc["arguments"])
                tool_calls += 1
                self.live_counts["tool_calls"] += 1
                worker_msgs.append({
                    "role": "tool", "tool_call_id": tc["id"],
                    "name": tc["name"],
                    "content": json.dumps({"ok": res.get("ok"),
                                           "output": str(res.get("output", ""))[-8000:]})})
            # outer loop re-checks budgets at the top -> partial on breach
        else:
            return _partial("exceeded MAX_TOOL_STEPS=%d without a final report"
                            % self.cfg.max_tool_steps)
        worker_s = time.time() - t_start
        if not final_text.strip():
            rep = worker_fallback("worker returned no report text", task)
            rep["_model"] = used_models[-1] if used_models else ""
            return rep, worker_s, turns, tool_calls
        try:
            report = parse_worker_output(final_text)
        except ValueError as e:
            report = worker_fallback(str(e)[:300], task)
        if via_agy:
            # AGY ran the task with its own tool loop: record provenance and
            # the real headless-run trail. The zero-tool guard below applies
            # to API function-tool workers only - AGY legitimately reports
            # tool_calls == 0 at the loop level.
            report["_executor"] = "agy"
            dur = raw.get("_agy_duration_s", 0.0)
            report["commands"] = (["agy headless run ok=True %.1fs"
                                   % float(dur or 0.0)]
                                  + list(report.get("commands", [])))[:30]
        if (not via_agy and tool_calls == 0
                and report.get("status") == "done"
                and (report.get("changes") or report.get("artifacts"))):
            # Deterministic anti-fabrication guard: a "done" report claiming
            # file changes/artifacts without a single tool call cannot be
            # trusted. Downgrade so the manager reviews it as a failure.
            report["status"] = "failed"
            report["problems"] = (["claimed changes/artifacts with zero tool "
                                   "calls - no evidence of execution"]
                                  + list(report.get("problems", [])))[:10]
            report["recommendation"] = ("retry: execute the task with "
                                        "repository tools; do not describe "
                                        "actions you did not perform.")
        # Attach the real tool/command trail so the manager sees actions, not claims.
        trail = ["%s ok=%s %.1fs" % (t["tool"], t["ok"], t.get("duration_s", 0))
                 for t in tools.invocation_log]
        if trail:
            report["commands"] = (report.get("commands", []) + trail)[:30]
        report["_model"] = used_models[-1] if used_models else ""
        return report, worker_s, turns, tool_calls

    def _build_tester(self):
        """Return (primary_tester, fallback_tester|None) per TESTER_BACKEND."""
        backend = (self.cfg.tester_backend or "off").strip().lower()
        if backend == "off":
            return None, None
        opencode_t = OpenCodeTester(
            repo_root=self.repo_root, timeout_s=self.cfg.tester_timeout_s,
            model=self.cfg.tester_model)
        api_t = None
        if self.cfg.manager_fallback_api_key and self.cfg.manager_fallback_base_url:
            api_t = ApiTester(
                self.cfg.manager_fallback_api_key,
                self.cfg.manager_fallback_base_url,
                self.cfg.manager_fallback_model,
                timeout_s=self.cfg.request_timeout_s,
                retries=max(1, self.cfg.request_retries - 1))
        if backend == "opencode":
            return opencode_t, api_t
        return api_t, opencode_t

    def _verify_step(self, iteration: int, decision: dict, report: dict,
                     evidence: dict, pre_changed: list | None = None) -> dict:
        """Independent verification: gates -> LLM tester -> verdict.

        Gates judge the per-iteration DELTA (post-worker changes minus the
        pre-worker image), so pre-existing untracked scaffolding never counts
        as scope drift. Returns a JSON-serializable verification record
        attached to the worker report for manager review. Never raises:
        tester failures degrade to INCONCLUSIVE, never to crashes.
        """
        t0 = time.time()
        self.state.set_task_state(iteration, "VERIFYING", "tester started")
        changed = changed_delta(pre_changed or [],
                                list(evidence.get("changed_files", []) or []))
        status = str(report.get("status", ""))
        record: dict = {"iteration": iteration,
                        "worker_status": status,
                        "gate_findings": [], "llm": None,
                        "verdict": "INCONCLUSIVE", "task_state": "VERIFYING",
                        "changed_delta": len(changed),
                        "changed_paths": [str(c)[:200] for c in changed[:50]]}

        tracker = StagnationTracker(
            self.state, repeat=self.cfg.stagnation_repeat,
            window=self.cfg.stagnation_window)
        ok, reasons = tracker.observe(
            str(decision.get("next_task", "")),
            changed, summarize_evidence(evidence), status)
        record["stagnation"] = {"ok": ok, "reasons": reasons}
        replanned = False
        if not ok:
            replanned = True
            self.state.signal_replan(reasons)
            self.state.set_task_state(
                iteration, "REPLAN", "; ".join(reasons)[:200])
            record["task_state"] = "REPLAN"

        if status in ("failed", "blocked"):
            # Infrastructure/budget failure: nothing completed to verify.
            record.update(notes="worker reported %s; no verification target"
                                % status)
            if not replanned:
                record["task_state"] = "REPAIR"
                self.state.set_task_state(iteration, "REPAIR", record["notes"])
            self.state.save_verification(iteration, record)
            self.ui.verify(record)
            return record

        gate_findings = (
            check_claims_vs_artifacts(report, self.repo_root, changed)
            + check_scope(decision, changed)
            + check_claim_strength(report))
        record["gate_findings"] = gate_findings

        llm = None
        skipped_llm = False
        if status == "done":
            hard = [f for f in gate_findings
                    if f.startswith("claimed artifact missing")
                    or f.startswith("worker claims file changes")]
            if not hard:
                prompt = build_tester_messages(
                    str(decision.get("next_task", "")),
                    str(decision.get("expected_result", "")),
                    report, str(evidence.get("git_diff", "")))
                llm = self._run_tester_backends(prompt)
            else:
                skipped_llm = True
        else:  # partial: deterministic gates only, no LLM burn
            skipped_llm = True

        combined = combine_verdict(gate_findings, llm)
        if skipped_llm and combined["verdict"] == "INCONCLUSIVE":
            combined["notes"] = ("LLM tester skipped (%s); gates decide"
                                 % ("partial report" if status == "partial"
                                    else "hard gate failure"))
        record.update(verdict=combined["verdict"], llm=llm,
                      notes=combined.get("notes", ""),
                      skipped_llm=skipped_llm,
                      duration_s=round(time.time() - t0, 1))
        final_evidence_state = "EVIDENCE_CHECK"
        self.state.set_task_state(iteration, final_evidence_state,
                                  "verdict=%s" % combined["verdict"])
        final_state = ({"PASS": "ACCEPTED", "FAIL": "REPAIR"}.get(
            combined["verdict"], "EVIDENCE_CHECK"))
        if record["task_state"] != "REPLAN":
            record["task_state"] = final_state
            self.state.set_task_state(iteration, final_state,
                                      combined["verdict"])
        self.state.save_verification(iteration, record)
        self.ui.verify(record)
        return record

    def _run_tester_backends(self, prompt: str):
        """Run primary tester backend, failing over once. Returns llm dict."""
        primary, fallback = self._build_tester()
        last_error = ""
        for tester in (primary, fallback):
            if tester is None:
                continue
            try:
                text = tester.verify(prompt)
            except Exception as e:  # noqa: BLE001 - degraded, not fatal
                last_error = "%s: %s" % (
                    getattr(tester, "backend", "?"), str(e)[:200])
                self.ui.warn("tester backend failed: %s" % last_error)
                continue
            try:
                out = parse_verdict(text)
            except ValueError as e:
                self.ui.warn("tester verdict unparseable: %s" % e)
                continue
            out["backend"] = getattr(tester, "backend", "?")
            return out
        self.ui.warn("no tester verdict available%s"
                     % (" (%s)" % last_error if last_error else ""))
        return None

    def _analyze_step(self, iteration: int, decision: dict, report: dict,
                      evidence: dict, verification: dict) -> dict:
        """Analyst (deterministic) + critic (deterministic) stage.

        Classifies the report's claims, runs critic alternatives on accepted
        work, and self-audits changed docs against this iteration's evidence.
        Never raises; never spends model quota (LLM critic only when
        TESTER_CRITIC is enabled, reusing the tester backends once).
        """
        import re as _re
        from lab import analyze as _analyze
        from lab import critic as _critic
        record: dict = {"iteration": iteration, "claim_kinds": {},
                        "critic_flags": [], "self_audit": [],
                        "llm_critic": None}
        blob_parts = (list(report.get("evidence", []) or [])
                      + [report.get("recommendation", "")])
        for part in blob_parts[:6]:
            kind = _analyze.claim_kind(part)
            record["claim_kinds"][kind] = record["claim_kinds"].get(kind, 0) + 1
        if verification.get("verdict") == "PASS":
            text = json.dumps(report)[:6000]
            summary = {
                "n_seeds": max([int(x) for x in
                                _re.findall(r"(\d+)\s+seeds?", text.lower())]
                               or [1]),
                "budgets_matched": ("flops" in text.lower()
                                    and "params" in text.lower()),
                "splits_shared": True,
                "variance_reported": ("std" in text.lower()
                                      or "\u00b1" in text),
                "mechanism": ("argmax" if "argmax" in text.lower() else ""),
            }
            for alt in _critic.generate_alternatives(summary):
                record["critic_flags"].append(alt["alternative"])
            crit = _critic.critique_conclusion(
                report.get("recommendation", ""), {
                    "n_seeds": summary["n_seeds"],
                    "baseline_compared": ("baseline" in text.lower()),
                    "replicated": False,
                    "negative_checked": ("null" in text.lower()
                                         or "no " in text.lower())})
            record["critic_verdict"] = crit["verdict"]
            record["critic_reasons"] = crit["reasons"]
        md_changed = [f for f in (verification.get("changed_paths", [])
                                  or []) if f.endswith(".md")]
        if md_changed:
            record["self_audit"] = _analyze.reconcile_iteration(
                md_changed, {"artifacts": report.get("artifacts", []),
                             "evidence_summary": ""}, self.repo_root)
        if self.cfg.tester_critic and verification.get("verdict") == "PASS":
            prompt = ("Adversarially review this accepted builder report. "
                      "Attempt to falsify its conclusions; return exactly one "
                      "verdict JSON {verdict: PASS|FAIL|INCONCLUSIVE, "
                      "evidence: [], notes: ''}.\nREPORT:\n"
                      + json.dumps(report)[:6000])
            llm = self._run_tester_backends(prompt)
            record["llm_critic"] = llm
            if isinstance(llm, dict) and llm.get("verdict") == "FAIL":
                record["critic_flags"].append("llm-critic FAIL")
        self.state.save_analysis(iteration, record)
        self.ui.analyze(record)
        return record

    def _finish(self, status: str, reason: str) -> dict:
        self.state.set_status(status)
        self.terminated_reason = reason
        summary = {"status": status, "reason": reason,
                   "iterations": self.state.iteration,
                   "live_counts": dict(self.live_counts),
                   "totals": dict(self.state._state.get("totals", {})),
                   "session": self.state.session_info()}
        self.ui.finish(summary)
        return summary


class _NullUI:
    def banner(self, *a, **k): pass
    def iteration_header(self, *a, **k): pass
    def call_start(self, *a, **k): pass
    def retry_wait(self, *a, **k): pass
    def progress(self, *a, **k): pass
    def verify(self, *a, **k): pass
    def analyze(self, *a, **k): pass
    def manager(self, *a, **k): pass
    def worker(self, *a, **k): pass
    def evidence(self, *a, **k): pass
    def next_action(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def finish(self, *a, **k): pass
    def invocations(self, *a, **k): pass
    def iteration_done(self, *a, **k): pass


class TerminalUI:
    """Readable one-screen-per-iteration log (no dashboard, just text)."""

    def banner(self, objective, cfg, session=None):
        print("=" * 70)
        print("NEURAL FABRIC AUTONOMOUS LOOP")
        for role, c in (("manager", cfg.redacted()["manager"]),
                        ("manager_fallback", cfg.redacted()["manager_fallback"]),
                        ("worker", cfg.redacted()["worker"]),
                        ("worker_backup", cfg.redacted()["worker_backup"])):
            print("  %s: %s [%s] %s" % (role, c.get("model"), c.get("provider"),
                                        c.get("base_url")))
        print("  budgets: turns<=%d tools<=%d task<=%.0fs session=%.1fh "
              "req_timeout=%.0fs req_retries=%d" % (
            cfg.max_tool_steps, cfg.worker_max_tool_calls,
            cfg.worker_task_timeout_s, cfg.session_hours,
            cfg.request_timeout_s, cfg.request_retries))
        if session and session.get("deadline_at"):
            print("  session %s deadline %s (%.1fh remaining)" % (
                session.get("session_id"), time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(session["deadline_at"])),
                max(0.0, (session.get("remaining_s") or 0)) / 3600))
        elif session:
            print("  session %s (no deadline)" % session.get("session_id"))
        print("objective: %s" % objective[:300])
        print("=" * 70, flush=True)

    def iteration_header(self, n, remaining=None):
        extra = (" (session %.1fh left)" % (remaining / 3600)
                 if remaining is not None else "")
        print("\n--- Iteration %d%s ---" % (n, extra), flush=True)

    def call_start(self, role, provider, model, attempt, is_fallback, timeout_s):
        print("  -> %s call: %s/%s attempt=%d timeout=%.0fs%s" % (
            role, provider, model, attempt, timeout_s,
            " [FALLBACK]" if is_fallback else ""), flush=True)

    def invocations(self, dicts):
        for i in dicts:
            print("  [%s] %s/%s attempt=%s %.1fs -> %s%s" % (
                i["role"], i["provider"], i["model"], i["attempt"],
                i.get("duration_s", 0), i["status"],
                (" (fallback from %s: %s)" % (i["fallback_from"], i["reason"][:160])
                 if i.get("fallback_from") else "")), flush=True)

    def manager(self, d):
        print("MANAGER status=%s model=%s\n  task: %s" % (
            d.get("status"), d.get("_model", "?"),
            str(d.get("next_task", ""))[:400]), flush=True)

    def worker(self, r):
        print("WORKER status=%s model=%s task=%.120s\n  files changed: %d tests: %d" % (
            r.get("status"), r.get("_model", "?"), r.get("task", ""),
            len(r.get("changes", [])), len(r.get("tests", []))), flush=True)

    def verify(self, record):
        print("VERIFY verdict=%s state=%s gates=%d llm=%s" % (
            record.get("verdict"), record.get("task_state"),
            len(record.get("gate_findings", [])),
            (record.get("llm") or {}).get("verdict", "-")
            if isinstance(record.get("llm"), dict) else "-"), flush=True)
        for f in list(record.get("gate_findings", []))[:4]:
            print("  gate: %s" % str(f)[:200], flush=True)

    def analyze(self, record):
        print("ANALYZE kinds=%s critic=%s self_audit=%d" % (
            record.get("claim_kinds"), record.get("critic_verdict", "-"),
            len(record.get("self_audit", []))), flush=True)
        for f in list(record.get("critic_flags", []))[:4]:
            print("  critic: %s" % str(f)[:200], flush=True)
        for f in list(record.get("self_audit", []))[:4]:
            print("  audit: %s" % str(f)[:200], flush=True)

    def evidence(self, summary):
        print("EVIDENCE %s" % summary, flush=True)

    def iteration_done(self, iter_s, manager_s, worker_s, turns, tool_calls):
        print("TIMING iter=%.1fs manager=%.1fs worker=%.1fs turns=%d tools=%d" % (
            iter_s, manager_s, worker_s, turns, tool_calls), flush=True)

    def next_action(self, decision, report):
        gaps = decision.get("gaps", []) or report.get("problems", [])
        print("MANAGER REVIEW expected=%.160s" % decision.get("expected_result", ""), flush=True)
        if gaps:
            print("  gaps: %s" % "; ".join(gaps[:3])[:300], flush=True)
        print("NEXT: %s" % decision.get("status"), flush=True)

    def warn(self, msg):
        print("WARN: %s" % msg[:400], flush=True)

    def retry_wait(self, role, provider, model, attempt, wait_s, error):
        print("  [%s] %s/%s attempt %d slow/failed (%s); retrying in %.0fs ..."
              % (role, provider, model, attempt,
                 str(error)[:120], wait_s), flush=True)

    def progress(self, path, snapshot):
        print("PROGRESS iter=%s elapsed=%.1fh remaining=%s task=%.100s" % (
            snapshot.get("iteration"), snapshot.get("elapsed_s", 0) / 3600,
            ("%.1fh" % (snapshot["remaining_s"] / 3600))
            if snapshot.get("remaining_s") is not None else "-",
            snapshot.get("current_task", "")), flush=True)

    def finish(self, summary):
        print("\nLOOP %s after %d iteration(s): %s" % (
            summary["status"], summary["iterations"],
            str(summary["reason"])[:300]), flush=True)
        print("live counts: %s" % summary.get("live_counts"), flush=True)
        print("totals: %s" % summary.get("totals"), flush=True)


def build_chains(cfg: LoopConfig):
    """Construct the isolated per-role fallback chains. Chains never cross.

    Manager: Kimi K3 (primary provider) -> Kimi K3 (NVIDIA fallback).
    Worker:  AGY headless CLI -> Nemotron Super (OpenRouter) -> Flash (NVIDIA).
    With AGY_ENABLED=false the worker chain is Nemotron -> Flash (legacy).
    """
    from .agy import AgyWorkerClient

    kw = {"timeout_s": cfg.request_timeout_s, "retries": cfg.request_retries}
    manager_primary = OpenAICompatibleClient(
        cfg.manager_api_key, cfg.manager_base_url, cfg.manager_model, **kw)
    manager_chain = RoleChain("manager", manager_primary)
    if cfg.manager_fallback_enabled:
        # No default URL exists for Kimi: fail loudly here, not at 2am mid-loop.
        if not cfg.manager_fallback_base_url:
            raise ValueError(
                "MANAGER_FALLBACK_BASE_URL is required but missing for %r; "
                "set it or use MANAGER_FALLBACK_ENABLED=false."
                % cfg.manager_fallback_model)
        manager_chain = RoleChain(
            "manager", manager_primary,
            OpenAICompatibleClient(cfg.manager_fallback_api_key,
                                   cfg.manager_fallback_base_url,
                                   cfg.manager_fallback_model, **kw))
    nemotron = OpenAICompatibleClient(
        cfg.worker_api_key, cfg.worker_base_url, cfg.worker_model, **kw)
    flash = OpenAICompatibleClient(
        cfg.worker_backup_api_key, cfg.worker_backup_base_url,
        cfg.worker_backup_model, **kw)
    if cfg.agy_enabled:
        agy_primary = AgyWorkerClient(
            repo_root=cfg.repo_root, command=cfg.agy_command,
            model=cfg.agy_model, effort=cfg.agy_effort, mode=cfg.agy_mode,
            timeout_s=cfg.agy_timeout_s,
            allow_dangerously_skip_permissions=cfg.agy_dangerous)
        worker_chain = RoleChain("worker", agy_primary, [nemotron, flash])
    else:
        worker_chain = RoleChain("worker", nemotron, flash)
    return manager_chain, worker_chain
