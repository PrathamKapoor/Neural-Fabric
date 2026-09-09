"""Tests for the AGY headless worker backend (mocked; no real CLI calls).

All AGY subprocess interaction is faked by swapping the `subprocess`
attribute inside `agent_loop.agy` (no global patching, no binaries).

Run:  python -m unittest discover -s tests -v   (from repo root)
"""
import json
import os
import subprocess as real_subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_loop.agy as agy_mod
from agent_loop.agy import (AgyError, AgyWorkerClient, classify_stderr,
                            extract_result_text, find_agy)
from agent_loop.clients import MockClient, TransientApiError
from agent_loop.config import LoopConfig
from agent_loop.loop import AgentLoop, build_chains
from agent_loop.roles import ChainExhausted, RoleChain
from agent_loop.state import LoopState

VALID_REPORT = json.dumps({
    "status": "done", "task": "t", "changes": ["f.txt"], "tests": ["ok"],
    "commands": ["read f.txt"], "evidence": ["saw f.txt"],
    "artifacts": [], "problems": [], "recommendation": "continue"})


class FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeSubprocess:
    TimeoutExpired = real_subprocess.TimeoutExpired

    def __init__(self, func):
        self.func = func
        self.calls = []

    def run(self, args, **kw):
        self.calls.append((args, kw))
        return self.func(args, kw)


def _patch_run(testcase, func):
    fake = FakeSubprocess(func)
    p = patch.object(agy_mod, "subprocess", fake)
    p.start()
    testcase.addCleanup(p.stop)
    return fake


def _repo(testcase):
    tmp = tempfile.TemporaryDirectory()
    testcase.addCleanup(tmp.cleanup)
    with open(os.path.join(tmp.name, "x.py"), "w") as f:
        f.write("y = 1\n")
    return tmp


def _client(tmpdir, **kw):
    d = dict(repo_root=tmpdir, command="agy")
    d.update(kw)
    return AgyWorkerClient(**d)


class TestDetection(unittest.TestCase):
    def test_find_agy_resolves_path(self):
        self.assertTrue(find_agy() is None or isinstance(find_agy(), str))

    def test_missing_binary_reported(self):
        c = AgyWorkerClient(repo_root=".", command="definitely-not-agy-9z9z")
        self.assertIsNone(c.executable())
        info = c.check_usability()
        self.assertFalse(info["ok"])
        self.assertIn("not found", info["reason"])

    def test_missing_binary_raises_classified_error(self):
        c = AgyWorkerClient(repo_root=".", command="definitely-not-agy-9z9z")
        with self.assertRaises(AgyError) as ctx:
            c.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.kind, "missing")


class TestFailureKinds(unittest.TestCase):
    def _failing_client(self, testcase, stdout="", stderr="", returncode=1):
        tmp = _repo(testcase)
        c = _client(tmp.name)
        with patch.object(AgyWorkerClient, "executable",
                          lambda self: "agy"):
            _patch_run(testcase, lambda a, k: FakeCompleted(
                returncode, stdout, stderr))
            with self.assertRaises(AgyError) as ctx:
                c.chat([{"role": "user", "content": "do it"}])
        return ctx.exception

    def test_auth_failure(self):
        e = self._failing_client(self, stderr="Error: not logged in, run login")
        self.assertEqual(e.kind, "auth")
        self.assertFalse(e.retryable)

    def test_permission_denied(self):
        e = self._failing_client(self, stderr="permission denied by policy")
        self.assertEqual(e.kind, "permission")

    def test_quota_is_retryable(self):
        e = self._failing_client(self, stderr="quota exceeded, 429")
        self.assertEqual(e.kind, "quota")
        self.assertTrue(e.retryable)

    def test_model_unavailable(self):
        e = self._failing_client(self, stderr="unknown model 'zzz'")
        self.assertEqual(e.kind, "model")

    def test_timeout(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        c = _client(tmp.name, timeout_s=5)

        def _slow(args, kw):
            raise real_subprocess.TimeoutExpired(args, 5)

        with patch.object(AgyWorkerClient, "executable", lambda self: "agy"):
            _patch_run(self, _slow)
            with self.assertRaises(AgyError) as ctx:
                c.chat([{"role": "user", "content": "do it"}])
        self.assertEqual(ctx.exception.kind, "timeout")
        self.assertTrue(ctx.exception.retryable)
        self.assertIn("not a research", str(ctx.exception))

    def test_nonzero_exit_generic(self):
        e = self._failing_client(self, stderr="something broke")
        self.assertEqual(e.kind, "exit_error")

    def test_empty_output_on_success_is_error(self):
        e = self._failing_client(self, stdout="   ", returncode=0)
        self.assertEqual(e.kind, "exit_error")


class TestOutputParsing(unittest.TestCase):
    def test_json_envelope_result_key(self):
        out = extract_result_text(json.dumps({"result": VALID_REPORT,
                                              "stats": {}}))
        self.assertEqual(out, VALID_REPORT)

    def test_plain_text_passthrough(self):
        self.assertEqual(extract_result_text("just prose"), "just prose")

    def test_unknown_envelope_preserved(self):
        out = extract_result_text(json.dumps({"weird": [1, 2]}))
        self.assertIn("weird", out)

    def test_denied_actions_surfaced_when_response_empty(self):
        out = extract_result_text(json.dumps({
            "status": "SUCCESS", "response": "",
            "denied_actions": [{"action": "command",
                                "display_name": "RunCommand"}]}))
        self.assertIn("denied", out)
        self.assertIn("RunCommand", out)

    def test_success_returns_report_text(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        c = _client(tmp.name)
        with patch.object(AgyWorkerClient, "executable", lambda self: "agy"):
            _patch_run(self, lambda a, k: FakeCompleted(
                0, json.dumps({"result": VALID_REPORT}), ""))
            msg = c.chat([{"role": "system", "content": "sys"},
                          {"role": "user", "content": "task"}])
        self.assertEqual(msg["role"], "assistant")
        self.assertIn('"status": "done"', msg["content"])
        self.assertTrue(msg["_agy"])
        self.assertNotIn("tool_calls", msg)

    def test_malformed_output_returns_text_for_manager_review(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        c = _client(tmp.name)
        with patch.object(AgyWorkerClient, "executable", lambda self: "agy"):
            _patch_run(self, lambda a, k: FakeCompleted(
                0, "All done, trust me.", ""))
            msg = c.chat([{"role": "user", "content": "task"}])
        self.assertEqual(msg["content"], "All done, trust me.")

    def test_prompt_contains_task_and_repo_scoped_flags(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        c = _client(tmp.name)
        with patch.object(AgyWorkerClient, "executable", lambda self: "agy"):
            fake = _patch_run(self, lambda a, k: FakeCompleted(
                0, json.dumps({"result": "{}"}), ""))
            c.chat([{"role": "system", "content": "SYS"},
                    {"role": "user", "content": "TASK-9"}])
        args = fake.calls[0][0]
        self.assertIn("--print", args)
        self.assertIn("--output-format", args)
        self.assertIn("json", args)
        self.assertIn("--add-dir", args)
        self.assertIn(tmp.name, args)
        self.assertIn("--mode", args)
        self.assertNotIn("--dangerously-skip-permissions", args)
        prompt = args[args.index("--print") + 1]
        self.assertIn("SYS", prompt)
        self.assertIn("TASK-9", prompt)

    def test_no_model_flag_when_default(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        c = _client(tmp.name)  # model="" -> AGY default
        with patch.object(AgyWorkerClient, "executable", lambda self: "agy"):
            fake = _patch_run(self, lambda a, k: FakeCompleted(
                0, json.dumps({"result": "{}"}), ""))
            c.chat([{"role": "user", "content": "t"}])
        self.assertNotIn("--model", fake.calls[0][0])

    def test_classify_helper(self):
        self.assertEqual(classify_stderr("401 Unauthorized"), "auth")
        self.assertEqual(classify_stderr("needs approval"), "permission")
        self.assertEqual(classify_stderr("429 slow down"), "quota")
        self.assertEqual(classify_stderr("model not found"), "model")
        self.assertEqual(classify_stderr("boom"), "exit_error")


class TestChainFallback(unittest.TestCase):
    def test_fallback_agy_to_nemotron(self):
        agy_primary = AgyWorkerClient(
            repo_root=".", command="definitely-not-agy-9z9z")
        nemotron = MockClient(
            [{"role": "assistant", "content": VALID_REPORT}],
            model="nvidia/nemotron-3-super-120b-a12b:free",
            provider="openrouter")
        chain = RoleChain("worker", agy_primary, [nemotron])
        msg, invos = chain.chat([{"role": "user", "content": "t"}],
                                tools=[{"type": "function"}])
        self.assertEqual([i.status for i in invos],
                         ["infrastructure_failure", "success"])
        self.assertEqual(invos[1].fallback_from, "(agy default)")
        self.assertIn("done", msg["content"])

    def test_two_level_fallback_order(self):
        seen = []
        agy_primary = AgyWorkerClient(
            repo_root=".", command="definitely-not-agy-9z9z")
        bad = MockClient([TransientApiError("overloaded")],
                         model="nvidia/nemotron-3-super-120b-a12b:free")
        good = MockClient([{"role": "assistant", "content": VALID_REPORT}],
                          model="deepseek-ai/deepseek-v4-flash-0731")
        chain = RoleChain("worker", agy_primary, [bad, good])
        msg, invos = chain.chat(
            [], on_call_start=lambda p, m, a, f: seen.append((m, f)))
        self.assertEqual([m for m, _ in seen],
                         ["(agy default)",
                          "nvidia/nemotron-3-super-120b-a12b:free",
                          "deepseek-ai/deepseek-v4-flash-0731"])
        self.assertEqual([f for _, f in seen], [False, True, True])
        self.assertIn("done", msg["content"])

    def test_all_workers_unavailable_exhausts(self):
        agy_primary = AgyWorkerClient(
            repo_root=".", command="definitely-not-agy-9z9z")
        bad = MockClient([TransientApiError("down")], model="nemotron")
        bad2 = MockClient([TransientApiError("down")], model="flash")
        chain = RoleChain("worker", agy_primary, [bad, bad2])
        with self.assertRaises(ChainExhausted) as ctx:
            chain.chat([])
        self.assertEqual(len(ctx.exception.invocations), 3)

    def test_build_chains_wires_agy_primary(self):
        cfg = LoopConfig(manager_api_key="m", worker_api_key="w",
                         worker_backup_api_key="b",
                         manager_fallback_enabled=False)
        mgr, wrk = build_chains(cfg)
        self.assertEqual(wrk.primary.provider, "agy")
        self.assertEqual(wrk.primary.model, "gemini-3.8-flash-high")
        self.assertEqual([f.model for f in wrk.fallbacks],
                         ["nvidia/nemotron-3-super-120b-a12b:free",
                          "deepseek-ai/deepseek-v4-flash-0731"])
        self.assertEqual(mgr.primary.model,
                         "nvidia/nemotron-3-ultra-550b-a55b:free")

    def test_dangerous_flag_on_by_authorized_default(self):
        cfg = LoopConfig(manager_api_key="m", worker_api_key="w",
                         worker_backup_api_key="b",
                         manager_fallback_enabled=False)
        self.assertTrue(cfg.agy_dangerous)
        _, wrk = build_chains(cfg)
        self.assertTrue(wrk.primary.allow_dangerous)

    def test_dangerous_flag_reaches_agy_argv(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        c = AgyWorkerClient(repo_root=tmp.name, command="agy",
                            allow_dangerously_skip_permissions=True)
        with patch.object(AgyWorkerClient, "executable", lambda self: "agy"):
            fake = _patch_run(self, lambda a, k: FakeCompleted(
                0, json.dumps({"result": "{}"}), ""))
            c.chat([{"role": "user", "content": "t"}])
        self.assertIn("--dangerously-skip-permissions", fake.calls[0][0])

    def test_dangerous_flag_opt_out(self):
        cfg = LoopConfig(manager_api_key="m", worker_api_key="w",
                         worker_backup_api_key="b",
                         manager_fallback_enabled=False,
                         agy_dangerous=False)
        _, wrk = build_chains(cfg)
        self.assertFalse(wrk.primary.allow_dangerous)

    def test_agy_disabled_preserves_legacy_chain(self):
        cfg = LoopConfig(manager_api_key="m", worker_api_key="w",
                         worker_backup_api_key="b",
                         manager_fallback_enabled=False,
                         agy_enabled=False)
        _, wrk = build_chains(cfg)
        self.assertEqual(wrk.primary.model,
                         "nvidia/nemotron-3-super-120b-a12b:free")
        self.assertEqual(wrk.fallback.model,
                         "deepseek-ai/deepseek-v4-flash-0731")


def _mgr_msg(status="continue", task="Do it with tools."):
    d = {"status": status, "assessment": "a", "completed": [], "gaps": [],
         "problems": [], "next_task": task, "expected_result": "e",
         "validation": ["v"], "constraints": [], "reason": "r"}
    return {"role": "assistant", "content": json.dumps(d)}


def _cfg(tmpdir, **kw):
    d = dict(repo_root=tmpdir, state_dir=os.path.join(tmpdir, ".agent_loop"),
             max_iterations=4, max_tool_steps=4, max_consecutive_failures=2,
             max_api_failures=3, session_hours=1)
    d.update(kw)
    return LoopConfig(manager_api_key="m", worker_api_key="w",
                      worker_backup_api_key="b",
                      manager_fallback_enabled=False, **d)


class TestLoopWithAgy(unittest.TestCase):
    def _agy_success(self, testcase, tmpdir, report=VALID_REPORT,
                     side_effect=None):
        def _fake(args, kw):
            if side_effect is not None:
                side_effect(args, kw)
            return FakeCompleted(0, json.dumps({"result": report}), "")

        _patch_run(testcase, _fake)
        return AgyWorkerClient(repo_root=tmpdir, command="agy")

    def test_agy_success_end_to_end_with_evidence(self):
        import subprocess as sp
        tmp = _repo(self)
        sp.run(["git", "init"], cwd=tmp.name, capture_output=True)
        sp.run(["git", "config", "user.email", "t@t.t"], cwd=tmp.name,
               capture_output=True)
        sp.run(["git", "config", "user.name", "t"], cwd=tmp.name,
               capture_output=True)

        def _write(args, kw):
            with open(os.path.join(tmp.name, "agy_made.txt"), "w") as f:
                f.write("real\n")

        agy_primary = self._agy_success(self, tmp.name,
                                        side_effect=_write)
        mgr = RoleChain("manager", MockClient(
            [_mgr_msg("continue"), _mgr_msg("complete", "")], model="kimi"))
        wrk = RoleChain("worker", agy_primary)
        loop = AgentLoop(_cfg(tmp.name), mgr, wrk)
        out = loop.run("Objective O")
        self.assertEqual(out["status"], "complete")
        with open(os.path.join(tmp.name, ".agent_loop",
                               "worker_report.json")) as f:
            rep = json.load(f)
        self.assertEqual(rep["status"], "done")
        self.assertEqual(rep["_executor"], "agy")
        ev_files = []
        for f in os.listdir(os.path.join(tmp.name, ".agent_loop",
                                         "history")):
            if f.endswith("_evidence.json"):
                with open(os.path.join(tmp.name, ".agent_loop",
                                       "history", f)) as fh:
                    ev_files.append(json.load(fh))
        self.assertTrue(any("agy_made.txt" in " ".join(
            e.get("changed_files", [])) for e in ev_files))

    def test_all_workers_down_flows_to_blocked(self):
        tmp = _repo(self)
        agy_primary = AgyWorkerClient(repo_root=tmp.name,
                                      command="definitely-not-agy-9z9z")
        bad = MockClient([TransientApiError("down")], model="nemotron")
        bad2 = MockClient([TransientApiError("down")], model="flash")
        mgr = RoleChain("manager", MockClient(
            [_mgr_msg("continue"), _mgr_msg("blocked", "stuck")],
            model="kimi"))
        loop = AgentLoop(_cfg(tmp.name), mgr,
                         RoleChain("worker", agy_primary, [bad, bad2]))
        out = loop.run("Objective O")
        self.assertEqual(out["status"], "blocked")
        self.assertGreaterEqual(loop.state.api_failures, 1)

    def test_resume_preserves_session_and_deadline(self):
        tmp = _repo(self)
        agy_primary = self._agy_success(self, tmp.name)
        mgr = RoleChain("manager", MockClient([_mgr_msg("complete", "")],
                                              model="kimi"))
        loop = AgentLoop(_cfg(tmp.name), mgr,
                         RoleChain("worker", agy_primary))
        loop.run("First")
        first_deadline = loop.state.session_info()["deadline_at"]
        loop2 = AgentLoop(
            _cfg(tmp.name),
            RoleChain("manager", MockClient([_mgr_msg("complete", "")],
                                            model="kimi")),
            RoleChain("worker", self._agy_success(self, tmp.name)),
            state=LoopState(os.path.join(tmp.name, ".agent_loop")))
        self.assertEqual(loop2.state.session_info()["deadline_at"],
                         first_deadline)
        self.assertEqual(loop2.state.get_objective(), "First")


class TestNoLeak(unittest.TestCase):
    def test_agy_argv_and_prompts_contain_no_secrets(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        secret = "sk-fake-secret-ABCDEF123456"
        old = os.environ.get("MANAGER_API_KEY")
        os.environ["MANAGER_API_KEY"] = secret
        try:
            c = AgyWorkerClient(repo_root=tmp.name, command="agy")
            with patch.object(AgyWorkerClient, "executable",
                              lambda self: "agy"):
                fake = _patch_run(self, lambda a, k: FakeCompleted(
                    0, json.dumps({"result": "{}"}), ""))
                c.chat([{"role": "user",
                         "content": "task mentioning sk-fake? no"}])
            blob = json.dumps(fake.calls[0][0])
            self.assertNotIn(secret, blob)
            red = LoopConfig(manager_api_key=secret).redacted()
            self.assertNotIn(secret, json.dumps(red))
        finally:
            if old is None:
                del os.environ["MANAGER_API_KEY"]
            else:
                os.environ["MANAGER_API_KEY"] = old

    def test_check_usability_reports_no_secrets(self):
        c = AgyWorkerClient(repo_root=".", command="agy")
        with patch.object(AgyWorkerClient, "executable",
                          lambda self: "agy"):
            _patch_run(self, lambda a, k: FakeCompleted(
                0, "model-a\nmodel-b\n", ""))
            info = c.check_usability()
        self.assertTrue(info["ok"])
        self.assertNotIn("sk-", json.dumps(info))


if __name__ == "__main__":
    unittest.main()
