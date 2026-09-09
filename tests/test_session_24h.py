"""Tests for timing instrumentation, worker budgets, and 24h session mode.

All hermetic (MockClients, temp dirs). No live calls.

Run:  python -m unittest discover -s tests -v   (from repo root)
"""
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_loop.clients import MockClient, TransientApiError
from agent_loop.config import LoopConfig
from agent_loop.loop import AgentLoop
from agent_loop.roles import RoleChain
from agent_loop.state import LoopState

VALID_MANAGER = json.dumps({
    "status": "continue", "assessment": "a", "completed": [], "gaps": [],
    "problems": [], "next_task": "Do the thing via git_status.",
    "expected_result": "e", "validation": ["v"], "constraints": [],
    "reason": "r"})
VALID_WORKER = json.dumps({
    "status": "done", "task": "t", "changes": [], "tests": [], "commands": [],
    "evidence": [], "artifacts": [], "problems": [], "recommendation": "c"})


def mgr_msg(status="continue", task="Do the thing via git_status."):
    d = json.loads(VALID_MANAGER)
    d["status"] = status
    d["next_task"] = task
    return {"role": "assistant", "content": json.dumps(d)}


def tool_msg(name="git_status", args=None):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": name,
                                         "arguments": json.dumps(args or {})}}]}


def _cfg(tmpdir, **kw):
    d = dict(repo_root=tmpdir, state_dir=os.path.join(tmpdir, ".agent_loop"),
             max_iterations=5, max_tool_steps=6, max_consecutive_failures=2,
             max_api_failures=3, session_hours=24)
    d.update(kw)
    return LoopConfig(manager_api_key="m", worker_api_key="w",
                      worker_backup_api_key="b",
                      manager_fallback_enabled=False, **d)


def _repo(testcase):
    tmp = tempfile.TemporaryDirectory()
    testcase.addCleanup(tmp.cleanup)
    with open(os.path.join(tmp.name, "x.py"), "w") as f:
        f.write("y = 1\n")
    return tmp


class TestTiming(unittest.TestCase):
    def test_invocation_durations_recorded(self):
        c = RoleChain("manager", MockClient([mgr_msg("complete", "")], model="m"))
        started = []

        def hook(p, m, a, f):
            started.append((m, a, f))

        msg, invos = c.chat([{"role": "user", "content": "hi"}],
                            on_call_start=hook)
        self.assertEqual(started, [("m", 1, False)])
        self.assertIn("duration_s", invos[0].as_dict())
        self.assertGreaterEqual(invos[0].duration_s, 0.0)

    def test_fallback_hook_flags(self):
        seen = []
        c = RoleChain("manager",
                      MockClient([TransientApiError("x")], model="mp"),
                      MockClient([mgr_msg("complete", "")], model="mf"))
        c.chat([], on_call_start=lambda p, m, a, f: seen.append((m, f)))
        self.assertEqual(seen, [("mp", False), ("mf", True)])

    def test_iteration_durations_in_history(self):
        tmp = _repo(self)
        mgr = RoleChain("manager", MockClient(
            [mgr_msg("continue"), mgr_msg("complete", "")], model="mp"))
        wrk = RoleChain("worker", MockClient(
            [{"role": "assistant", "content": VALID_WORKER}], model="wp"))
        loop = AgentLoop(_cfg(tmp.name), mgr, wrk)
        out = loop.run("O")
        self.assertEqual(out["status"], "complete")
        summaries = loop.state.history()
        self.assertIn("durations", summaries[0])
        d = summaries[0]["durations"]
        for k in ("iteration_s", "manager_s", "worker_s", "turns", "tool_calls"):
            self.assertIn(k, d)
        totals = out["totals"]
        self.assertIn("manager_s", totals)

    def test_tool_durations_logged(self):
        from agent_loop.tools import ToolExecutor
        tmp = _repo(self)
        ex = ToolExecutor(tmp.name)
        ex.execute("git_status", {})
        self.assertIn("duration_s", ex.invocation_log[0])


class TestWorkerBudgets(unittest.TestCase):
    def _always_tools(self):
        return MockClient([tool_msg()], model="wp")  # infinite tool loop

    def test_tool_call_budget_yields_partial(self):
        tmp = _repo(self)
        mgr = RoleChain("manager", MockClient([mgr_msg("continue")], model="mp"))
        wrk = RoleChain("worker", self._always_tools())
        loop = AgentLoop(_cfg(tmp.name, max_iterations=1,
                              worker_max_tool_calls=2), mgr, wrk)
        out = loop.run("O")
        with open(os.path.join(tmp.name, ".agent_loop",
                               "worker_report.json")) as f:
            rep = json.load(f)
        self.assertEqual(rep["status"], "partial")
        self.assertTrue(any("budget" in p for p in rep["problems"]))
        self.assertIn(out["status"], ("max_iterations", "failure_threshold",
                                      "complete", "blocked"))

    def test_turn_budget_yields_partial(self):
        tmp = _repo(self)
        mgr = RoleChain("manager", MockClient([mgr_msg("continue")], model="mp"))
        wrk = RoleChain("worker", self._always_tools())
        loop = AgentLoop(_cfg(tmp.name, max_iterations=1, max_tool_steps=1),
                         mgr, wrk)
        loop.run("O")
        with open(os.path.join(tmp.name, ".agent_loop",
                               "worker_report.json")) as f:
            rep = json.load(f)
        self.assertEqual(rep["status"], "partial")
        self.assertTrue(any("MAX_TOOL_STEPS" in p for p in rep["problems"]))

    def test_task_timeout_yields_partial(self):
        tmp = _repo(self)
        mgr = RoleChain("manager", MockClient([mgr_msg("continue")], model="mp"))
        wrk = RoleChain("worker", self._always_tools())
        loop = AgentLoop(_cfg(tmp.name, max_iterations=1,
                              worker_task_timeout_s=0), mgr, wrk)
        loop.run("O")
        with open(os.path.join(tmp.name, ".agent_loop",
                               "worker_report.json")) as f:
            rep = json.load(f)
        self.assertEqual(rep["status"], "partial")


class TestSession(unittest.TestCase):
    def test_deadline_set_once_and_preserved_on_resume(self):
        tmp = _repo(self)
        s = LoopState(os.path.join(tmp.name, ".agent_loop"))
        first = s.ensure_session("O", 24)
        time.sleep(0.02)
        second = s.ensure_session("O", 0.0001)  # must NOT reset the clock
        self.assertEqual(first["session_id"], second["session_id"])
        self.assertEqual(first["deadline_at"], second["deadline_at"])
        # fresh instance = restart: same deadline
        s2 = LoopState(os.path.join(tmp.name, ".agent_loop"))
        third = s2.ensure_session("O", 24)
        self.assertEqual(third["deadline_at"], first["deadline_at"])
        self.assertGreater(third["remaining_s"], 20 * 3600)

    def test_expired_deadline_finalizes_with_report(self):
        tmp = _repo(self)
        mgr = RoleChain("manager", MockClient([mgr_msg("continue")], model="mp"))
        wrk = RoleChain("worker", MockClient(
            [{"role": "assistant", "content": VALID_WORKER}], model="wp"))
        loop = AgentLoop(_cfg(tmp.name, session_hours=0.00001), mgr, wrk)
        out = loop.run("O")
        self.assertEqual(out["status"], "deadline")
        final = os.path.join(tmp.name, ".agent_loop", "final_report.md")
        self.assertTrue(os.path.exists(final))
        with open(final) as f:
            body = f.read()
        self.assertIn("final report", body.lower())
        self.assertIn("O", body)

    def test_research_state_accumulates(self):
        tmp = _repo(self)
        mgr = RoleChain("manager", MockClient(
            [mgr_msg("continue"), mgr_msg("complete", "")], model="mp"))
        wrk = RoleChain("worker", MockClient(
            [{"role": "assistant", "content": VALID_WORKER}], model="wp"))
        loop = AgentLoop(_cfg(tmp.name), mgr, wrk)
        loop.run("O")
        s2 = LoopState(os.path.join(tmp.name, ".agent_loop"))
        summary = s2.research_summary()
        self.assertIn("COMPLETED TASKS", summary)

    def test_no_deadline_when_disabled(self):
        tmp = _repo(self)
        s = LoopState(os.path.join(tmp.name, ".agent_loop"))
        info = s.ensure_session("O", 0)
        self.assertEqual(info["deadline_at"], 0)
        self.assertIsNone(info["remaining_s"])


class TestRequestTimeoutEnv(unittest.TestCase):
    def _isolated_env(self, testcase, **kw):
        tmp = tempfile.TemporaryDirectory()
        testcase.addCleanup(tmp.cleanup)
        old = dict(os.environ)
        for k in ("REQUEST_TIMEOUT_S", "REQUEST_RETRIES", "MANAGER_API_KEY",
                  "WORKER_API_KEY", "WORKER_BACKUP_API_KEY",
                  "MANAGER_FALLBACK_ENABLED", "OPENAI_API_KEY",
                  "OPENAI_BASE_URL"):
            os.environ.pop(k, None)
        os.environ.update(kw)
        return tmp, old

    def test_defaults(self):
        tmp, old = self._isolated_env(self)
        try:
            cfg = LoopConfig.from_env(repo_root=tmp.name)
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(cfg.request_timeout_s, 120.0)
        self.assertEqual(cfg.request_retries, 3)

    def test_env_override(self):
        tmp, old = self._isolated_env(self, REQUEST_TIMEOUT_S="30",
                                      REQUEST_RETRIES="1",
                                      MANAGER_API_KEY="m",
                                      WORKER_API_KEY="w",
                                      WORKER_BACKUP_API_KEY="b",
                                      MANAGER_FALLBACK_ENABLED="false")
        try:
            cfg = LoopConfig.from_env(repo_root=tmp.name)
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(cfg.request_timeout_s, 30.0)
        self.assertEqual(cfg.request_retries, 1)
        self.assertEqual(cfg.validate(), [])

    def test_invalid_values_rejected(self):
        tmp, old = self._isolated_env(self, REQUEST_TIMEOUT_S="0",
                                      REQUEST_RETRIES="0")
        try:
            cfg = LoopConfig.from_env(repo_root=tmp.name)
            errs = cfg.validate()
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertTrue(any("REQUEST_TIMEOUT_S" in e for e in errs))
        self.assertTrue(any("REQUEST_RETRIES" in e for e in errs))


class TestQuantumGate(unittest.TestCase):
    def _loop(self, tmpdir, **kw):
        mgr = RoleChain("manager", MockClient([mgr_msg("complete", "")],
                                              model="mp"))
        wrk = RoleChain("worker", MockClient(
            [{"role": "assistant", "content": VALID_WORKER}], model="wp"))
        return AgentLoop(_cfg(tmpdir, **kw), mgr, wrk)

    def test_short_session_with_failfast_timeouts_proceeds(self):
        # Regression: the old gate (task_timeout + timeout*retries + 120)
        # blocked ANY session under ~0.9h, making --hours 0.1 unusable.
        tmp = _repo(self)
        loop = self._loop(tmp.name, request_timeout_s=30, request_retries=1,
                          worker_task_timeout_s=180)
        self.assertTrue(loop._quantum_ok(360.0))  # 0.1h session

    def test_tiny_remaining_still_finalizes(self):
        tmp = _repo(self)
        loop = self._loop(tmp.name, request_timeout_s=30, request_retries=1)
        self.assertFalse(loop._quantum_ok(60.0))

    def test_no_deadline_always_ok(self):
        tmp = _repo(self)
        loop = self._loop(tmp.name)
        self.assertTrue(loop._quantum_ok(None))


class TestProgressSnapshot(unittest.TestCase):
    def test_snapshot_written_per_iteration(self):
        tmp = _repo(self)
        mgr = RoleChain("manager", MockClient(
            [mgr_msg("continue"), mgr_msg("complete", "")], model="mp"))
        wrk = RoleChain("worker", MockClient(
            [{"role": "assistant", "content": VALID_WORKER}], model="wp"))
        loop = AgentLoop(_cfg(tmp.name), mgr, wrk)
        loop.run("O")
        prog_path = os.path.join(tmp.name, ".agent_loop", "progress.json")
        self.assertTrue(os.path.exists(prog_path))
        with open(prog_path) as f:
            snap = json.load(f)
        for k in ("session_id", "iteration", "elapsed_s", "remaining_s",
                  "current_task", "manager_status", "worker_status",
                  "latest_evidence", "tests", "problems",
                  "consecutive_failures", "api_failures",
                  "next_direction", "durations", "totals"):
            self.assertIn(k, snap)
        self.assertEqual(snap["iteration"], 1)
        self.assertIsNotNone(snap["remaining_s"])  # 24h deadline active
        hist_copy = os.path.join(tmp.name, ".agent_loop", "history",
                                 "iter_0001_progress.json")
        self.assertTrue(os.path.exists(hist_copy))
        # survives restart: fresh state instance still reads it
        with open(prog_path) as f:
            self.assertEqual(json.load(f)["session_id"], snap["session_id"])


class TestZeroToolGuard(unittest.TestCase):
    def test_done_with_changes_but_no_tools_downgraded(self):
        tmp = _repo(self)
        fabricated = json.dumps({
            "status": "done", "task": "t", "changes": ["f.txt"],
            "tests": [], "commands": ["echo hi"], "evidence": ["trust me"],
            "artifacts": ["f.txt"], "problems": [],
            "recommendation": "all good"})
        mgr = RoleChain("manager", MockClient(
            [mgr_msg("continue"), mgr_msg("complete", "")], model="mp"))
        wrk = RoleChain("worker", MockClient(
            [{"role": "assistant", "content": fabricated}], model="wp"))
        loop = AgentLoop(_cfg(tmp.name), mgr, wrk)
        loop.run("O")
        with open(os.path.join(tmp.name, ".agent_loop",
                               "worker_report.json")) as f:
            rep = json.load(f)
        self.assertEqual(rep["status"], "failed")
        self.assertTrue(any("zero tool calls" in p for p in rep["problems"]))

    def test_done_without_changes_and_no_tools_passes_through(self):
        tmp = _repo(self)
        honest = json.dumps({
            "status": "done", "task": "t", "changes": [],
            "tests": ["manual inspection"], "commands": [],
            "evidence": ["nothing to change"], "artifacts": [],
            "problems": [], "recommendation": "c"})
        mgr = RoleChain("manager", MockClient(
            [mgr_msg("continue"), mgr_msg("complete", "")], model="mp"))
        wrk = RoleChain("worker", MockClient(
            [{"role": "assistant", "content": honest}], model="wp"))
        loop = AgentLoop(_cfg(tmp.name), mgr, wrk)
        loop.run("O")
        with open(os.path.join(tmp.name, ".agent_loop",
                               "worker_report.json")) as f:
            rep = json.load(f)
        self.assertEqual(rep["status"], "done")


class TestRetryVisibility(unittest.TestCase):
    def test_utf8_stdio_never_crashes_on_unicode(self):
        import sys
        from agent_loop.loop import TerminalUI, ensure_utf8_stdio
        ensure_utf8_stdio()
        self.assertEqual(sys.stdout.encoding, "utf-8")
        ui = TerminalUI()
        # The exact crash: cp1252 stdout + U+2265 in manager text.
        ui.next_action({"status": "continue",
                        "expected_result": "loss \u2265 0.3 threshold",
                        "gaps": []},
                       {"status": "done", "problems": []})
        ui.manager({"status": "continue", "_model": "m",
                    "next_task": "verify \u0394 routing"})
        ui.worker({"status": "done", "_model": "w", "task": "t",
                   "changes": ["f \u2192 g"], "tests": []})
    def test_on_retry_fires_before_backoff(self):
        from agent_loop.clients import (OpenAICompatibleClient,
                                        TransientApiError)
        seen = []

        class Flaky(OpenAICompatibleClient):
            def __init__(self):
                super().__init__("k", "https://x.example/v1", "m", retries=2)
                self.n = 0

            def _post(self, payload):
                self.n += 1
                if self.n == 1:
                    raise TransientApiError("slow provider")
                return {"choices": [{"message": {"role": "assistant",
                                                 "content": "{}"}}]}

        c = Flaky()
        out = c.chat([], on_retry=lambda a, w, e: seen.append((a, w)))
        self.assertEqual(out["content"], "{}")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], 1)  # attempt number
        self.assertGreater(seen[0][1], 0)  # wait seconds

    def test_role_chain_forwards_retry_hook(self):
        from agent_loop.clients import TransientApiError
        seen = []
        c = RoleChain("manager",
                      MockClient([TransientApiError("x")], model="mp"),
                      MockClient([mgr_msg("complete", "")], model="mf"))
        # MockClient ignores on_retry; chain must still fall back correctly
        msg, invos = c.chat([],
                            on_retry=lambda *a: seen.append(a))
        self.assertIn("complete", msg["content"])
        self.assertEqual(invos[1].fallback_from, "mp")


if __name__ == "__main__":
    unittest.main()
