"""Unit tests for the autonomous manager/worker loop (no live LLM calls).

Run:  python -m unittest discover -s tests -v   (from repo root)
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_loop import clients, evidence, schemas, state as state_mod, tools
from agent_loop.config import LoopConfig
from agent_loop.loop import AgentLoop
from agent_loop.roles import RoleChain
from agent_loop.prompts import (PROJECT_CONSTRAINTS, build_manager_messages,
                                build_worker_messages)

VALID_MANAGER = json.dumps({
    "status": "continue", "assessment": "good progress",
    "completed": ["x"], "gaps": ["y"], "problems": [],
    "next_task": "Run the benchmark via run_command.",
    "expected_result": "A benchmark JSON artifact.",
    "validation": ["artifact exists"], "constraints": [], "reason": "next step"})

VALID_WORKER = json.dumps({
    "status": "done", "task": "t", "changes": ["f.py"], "tests": ["ok"],
    "commands": ["echo hi"], "evidence": ["log"], "artifacts": ["a.json"],
    "problems": [], "recommendation": "continue"})


def mgr_msg(status="continue", task="Do the thing."):
    d = json.loads(VALID_MANAGER)
    d["status"] = status
    d["next_task"] = task
    return {"role": "assistant", "content": json.dumps(d)}


def worker_tool_msg(name="git_status", args=None, call_id="c1"):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name,
                                         "arguments": json.dumps(args or {})}}]}


class TestSchemas(unittest.TestCase):
    def test_manager_valid(self):
        d = schemas.parse_manager_output(VALID_MANAGER)
        self.assertEqual(d["status"], "continue")
        self.assertIn("benchmark", d["next_task"])

    def test_manager_fenced(self):
        d = schemas.parse_manager_output("```json\n%s\n```" % VALID_MANAGER)
        self.assertEqual(d["status"], "continue")

    def test_manager_prose_around_json(self):
        d = schemas.parse_manager_output("Here you go:\n%s\nDone." % VALID_MANAGER)
        self.assertEqual(d["status"], "continue")

    def test_manager_bad_status_rejected(self):
        with self.assertRaises(ValueError):
            schemas.parse_manager_output('{"status": "finished", "next_task": "x"}')

    def test_manager_continue_requires_task(self):
        with self.assertRaises(ValueError):
            schemas.parse_manager_output('{"status": "continue", "next_task": ""}')

    def test_manager_complete_without_task_ok(self):
        d = schemas.parse_manager_output('{"status": "complete", "assessment": "done"}')
        self.assertEqual(d["status"], "complete")

    def test_manager_garbage_raises(self):
        with self.assertRaises(ValueError):
            schemas.parse_manager_output("Done. Tests pass. Implemented successfully.")

    def test_manager_fallback_is_retry(self):
        d = schemas.manager_fallback("boom")
        self.assertEqual(d["status"], "retry")
        self.assertTrue(d["next_task"])

    def test_worker_valid(self):
        r = schemas.parse_worker_output(VALID_WORKER)
        self.assertEqual(r["status"], "done")

    def test_worker_all_statuses(self):
        for s in ("done", "partial", "blocked", "failed"):
            r = schemas.parse_worker_output('{"status": "%s"}' % s)
            self.assertEqual(r["status"], s)

    def test_worker_garbage_raises_and_fallback(self):
        with self.assertRaises(ValueError):
            schemas.parse_worker_output("All tests pass, trust me.")
        fb = schemas.worker_fallback("bad json")
        self.assertEqual(fb["status"], "failed")


class TestPrompts(unittest.TestCase):
    def test_manager_input_contains_all_required_parts(self):
        msgs = build_manager_messages("OBJ", "STATE", {"status": "continue"},
                                      {"status": "done"}, {"git_status": "clean"})
        blob = msgs[0]["content"] + msgs[1]["content"]
        for needle in ("OBJ", "STATE", "REPOSITORY EVIDENCE",
                       "EXPECTED vs ACTUAL", "Scientific honesty",
                       "heterogeneous", "routing collapse", "multi-seed"):
            self.assertIn(needle, blob)

    def test_worker_input_contains_task_contract(self):
        msgs = build_worker_messages("OBJ", "PROJ", "TASK-X", "EXP-Y",
                                     ["v1"], ["c1"])
        blob = msgs[0]["content"] + msgs[1]["content"]
        for needle in ("OBJ", "TASK-X", "EXP-Y", "v1", "c1", "Never fabricate"):
            self.assertIn(needle, blob)

    def test_constraints_mention_honesty(self):
        self.assertIn("never fabricate", PROJECT_CONSTRAINTS.lower())


class TestConfig(unittest.TestCase):
    def _env(self, **kw):
        # Isolated repo root: never reads the developer's real .env.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self._repo_root = tmp.name
        old = dict(os.environ)
        for k in ("MANAGER_API_KEY", "WORKER_API_KEY", "WORKER_BACKUP_API_KEY",
                  "MANAGER_FALLBACK_API_KEY", "MANAGER_FALLBACK_BASE_URL",
                  "MANAGER_FALLBACK_ENABLED", "OPENAI_API_KEY", "OPENAI_BASE_URL",
                  "MANAGER_MODEL", "WORKER_MODEL"):
            os.environ.pop(k, None)
        os.environ.update(kw)
        return old

    def _load(self):
        return LoopConfig.from_env(repo_root=self._repo_root)

    def test_from_env_and_redaction(self):
        old = self._env(MANAGER_API_KEY="sk-manager-1234567890",
                        WORKER_API_KEY="sk-worker-1234567890",
                        WORKER_BACKUP_API_KEY="sk-backup-1234567890",
                        MANAGER_FALLBACK_API_KEY="sk-kimi-1234567890",
                        MANAGER_FALLBACK_BASE_URL="https://kimi.example.com/v1",
                        MANAGER_MODEL="deepseek-ai/deepseek-v4-pro-0813",
                        WORKER_MODEL="thinkingmachines/inkling-small:free",
                        MAX_ITERATIONS="7")
        try:
            cfg = self._load()
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(cfg.max_iterations, 7)
        self.assertEqual(cfg.manager_model, "deepseek-ai/deepseek-v4-pro-0813")
        self.assertEqual(cfg.worker_model, "thinkingmachines/inkling-small:free")
        self.assertEqual(cfg.validate(), [])
        red = cfg.redacted()
        blob = json.dumps(red)
        for secret in ("sk-manager-1234567890", "sk-worker-1234567890",
                       "sk-backup-1234567890", "sk-kimi-1234567890"):
            self.assertNotIn(secret, blob)

    def test_missing_keys_reported(self):
        old = self._env()
        try:
            cfg = self._load()
            errs = cfg.validate()
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertTrue(any("MANAGER_API_KEY" in e for e in errs))
        self.assertTrue(any("WORKER_API_KEY" in e for e in errs))

    def test_shared_key_fills_all_roles(self):
        old = self._env(OPENAI_API_KEY="shared-key")
        try:
            cfg = self._load()
            errs = cfg.validate()
        finally:
            os.environ.clear()
            os.environ.update(old)
        # shared key covers all four key slots; Kimi URL is user-specified
        self.assertEqual(errs, [])
        self.assertEqual(cfg.manager_fallback_base_url,
                         "https://integrate.api.nvidia.com/v1")

    def test_explicit_beats_shared(self):
        # precedence: explicit per-role URL > shared OPENAI_BASE_URL > default
        old = self._env(OPENAI_API_KEY="shared-key",
                        OPENAI_BASE_URL="https://shared.example.com/v1",
                        MANAGER_API_KEY="explicit-manager",
                        MANAGER_BASE_URL="https://manager.example.com/v1",
                        MANAGER_FALLBACK_ENABLED="false",
                        WORKER_BACKUP_API_KEY="explicit-backup")
        try:
            cfg = self._load()
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(cfg.manager_api_key, "explicit-manager")
        self.assertEqual(cfg.worker_api_key, "shared-key")
        self.assertEqual(cfg.worker_backup_api_key, "explicit-backup")
        self.assertEqual(cfg.manager_base_url, "https://manager.example.com/v1")
        self.assertEqual(cfg.worker_base_url, "https://shared.example.com/v1")


class TestTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        with open(os.path.join(self.root, "a.py"), "w") as f:
            f.write("x = 1\n")
        self.ex = tools.ToolExecutor(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_list_read_write_edit_roundtrip(self):
        r = self.ex.execute("list_files", {"pattern": "*.py"})
        self.assertTrue(r["ok"] and "a.py" in r["output"])
        r = self.ex.execute("read_file", {"path": "a.py"})
        self.assertTrue(r["ok"] and "x = 1" in r["output"])
        r = self.ex.execute("write_file", {"path": "sub/b.txt", "content": "hello"})
        self.assertTrue(r["ok"])
        r = self.ex.execute("edit_file", {"path": "a.py", "old_string": "x = 1",
                                          "new_string": "x = 2"})
        self.assertTrue(r["ok"])
        with open(os.path.join(self.root, "a.py")) as f:
            self.assertIn("x = 2", f.read())

    def test_edit_missing_string_fails_cleanly(self):
        r = self.ex.execute("edit_file", {"path": "a.py", "old_string": "zzz",
                                          "new_string": "y"})
        self.assertFalse(r["ok"])

    def test_path_escape_refused(self):
        r = self.ex.execute("read_file", {"path": "../outside.py"})
        self.assertFalse(r["ok"])

    def test_run_command_echo(self):
        r = self.ex.execute("run_command", {"command": "echo hello-loop"})
        self.assertTrue(r["ok"] and "hello-loop" in r["output"])

    def test_run_command_failure_is_data(self):
        r = self.ex.execute("run_command", {"command": "exit 3"})
        self.assertFalse(r["ok"])

    def test_git_push_blocked(self):
        r = self.ex.execute("run_command", {"command": "git push origin main"})
        self.assertFalse(r["ok"])
        self.assertIn("REFUSED", r["output"])

    def test_git_reset_hard_blocked(self):
        r = self.ex.execute("run_command", {"command": "git reset --hard HEAD"})
        self.assertFalse(r["ok"])

    def test_git_commit_blocked_without_instruction(self):
        r = self.ex.execute("run_command", {"command": "git commit -m hi"})
        self.assertFalse(r["ok"])
        ex2 = tools.ToolExecutor(self.root, allow_commit=True)
        # still fails (no repo / nothing to commit) but NOT refused
        r2 = ex2.execute("run_command", {"command": "git commit -m hi"})
        self.assertNotIn("REFUSED", r2["output"])

    def test_destructive_rm_blocked(self):
        r = self.ex.execute("run_command", {"command": "rm -rf / tmp"})
        self.assertFalse(r["ok"])

    def test_invocation_log_recorded(self):
        self.ex.execute("read_file", {"path": "a.py"})
        self.assertEqual(self.ex.invocation_log[-1]["tool"], "read_file")


class TestEvidenceAndGit(unittest.TestCase):
    def test_git_evidence_in_real_repo(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ev = evidence.collect_evidence(repo)
        for k in ("git_status", "git_diff_stat", "changed_files",
                  "artifacts", "recent_commits"):
            self.assertIn(k, ev)
        self.assertIsInstance(ev["changed_files"], list)

    def test_evidence_in_temp_git_repo(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        subprocess.run(["git", "init"], cwd=tmp.name, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=tmp.name,
                       capture_output=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=tmp.name,
                       capture_output=True)
        with open(os.path.join(tmp.name, "f.txt"), "w") as f:
            f.write("hi\n")
        subprocess.run(["git", "add", "."], cwd=tmp.name, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp.name,
                       capture_output=True)
        with open(os.path.join(tmp.name, "f.txt"), "a") as f:
            f.write("more\n")
        ev = evidence.collect_evidence(tmp.name)
        self.assertIn("f.txt", ev["git_status"])
        self.assertIn("more", ev["git_diff"])


class TestState(unittest.TestCase):
    def test_persistence_and_resume(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        s1 = state_mod.LoopState(os.path.join(tmp.name, ".agent_loop"))
        s1.set_objective("Test objective")
        s1.bump_iteration()
        s1.save_manager_decision(1, {"status": "continue"})
        s1.save_worker_report(1, {"status": "done"})
        s1.save_iteration_summary(1, {"iteration": 1})
        # new instance = simulated restart
        s2 = state_mod.LoopState(os.path.join(tmp.name, ".agent_loop"))
        self.assertEqual(s2.get_objective(), "Test objective")
        self.assertEqual(s2.iteration, 1)
        self.assertEqual(len(s2.history()), 1)

    def test_history_never_overwritten(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        s = state_mod.LoopState(os.path.join(tmp.name, ".agent_loop"))
        s.save_iteration_summary(1, {"a": 1})
        s.save_iteration_summary(1, {"a": 2})
        files = list((s.history_dir).glob("iter_0001_summary*.json"))
        self.assertEqual(len(files), 2)

    def test_stop_file(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        s = state_mod.LoopState(os.path.join(tmp.name, ".agent_loop"))
        self.assertFalse(s.stop_requested())
        s.request_stop("human")
        self.assertTrue(s.stop_requested())


def _cfg(tmpdir, **kw):
    d = dict(repo_root=tmpdir, state_dir=os.path.join(tmpdir, ".agent_loop"),
             max_iterations=5, max_tool_steps=4, max_consecutive_failures=2,
             max_api_failures=3)
    d.update(kw)
    return LoopConfig(manager_api_key="m", worker_api_key="w", **d)


class TestLoop(unittest.TestCase):
    def _loop(self, cfg, mgr, wrk):
        return AgentLoop(cfg, RoleChain("manager", mgr), RoleChain("worker", wrk))

    def _repo(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with open(os.path.join(tmp.name, "x.py"), "w") as f:
            f.write("y = 1\n")
        return tmp

    def test_complete_terminates(self):
        tmp = self._repo()
        mgr = clients.MockClient([mgr_msg("complete", "")])
        wrk = clients.MockClient([{"role": "assistant", "content": VALID_WORKER}])
        loop = self._loop(_cfg(tmp.name), mgr, wrk)
        out = loop.run("Do X")
        self.assertEqual(out["status"], "complete")
        self.assertEqual(wrk.calls, [])  # no worker needed when complete

    def test_manager_worker_evidence_flow_with_mock_tools(self):
        tmp = self._repo()
        mgr = clients.MockClient([
            mgr_msg("continue", "Run git_status via the git_status tool."),
            mgr_msg("complete", ""),
        ])
        wrk = clients.MockClient([
            worker_tool_msg("git_status"),
            {"role": "assistant", "content": VALID_WORKER},
        ])
        loop = self._loop(_cfg(tmp.name), mgr, wrk)
        out = loop.run("Do Y")
        self.assertEqual(out["status"], "complete")
        self.assertEqual(out["iterations"], 2)
        # worker really invoked the tool layer (trail attached to report)
        with open(os.path.join(tmp.name, ".agent_loop", "worker_report.json")) as f:
            last_report = json.load(f)
        self.assertTrue(any("git_status" in c for c in last_report["commands"]))
        # history persisted
        hist = list(loop.state.history_dir.glob("iter_*"))
        self.assertTrue(len(hist) >= 4)

    def test_malformed_manager_output_triggers_retry_path(self):
        tmp = self._repo()
        mgr = clients.MockClient([
            {"role": "assistant", "content": "Done. Tests pass. Trust me."},
            mgr_msg("complete", ""),
        ])
        wrk = clients.MockClient([{"role": "assistant", "content": VALID_WORKER}])
        loop = self._loop(_cfg(tmp.name), mgr, wrk)
        out = loop.run("Do Z")
        self.assertEqual(out["status"], "complete")
        first_path = os.path.join(
            tmp.name, ".agent_loop", "history", "iter_0001_manager_decision.json")
        with open(first_path) as f:
            first = json.load(f)
        self.assertEqual(first["status"], "retry")  # automatic corrective fallback

    def test_malformed_worker_output_counts_as_failure(self):
        tmp = self._repo()
        mgr = clients.MockClient([mgr_msg("continue", "task"), mgr_msg("blocked", "stuck")])
        wrk = clients.MockClient([{"role": "assistant", "content": "all good, done!"},
                                  {"role": "assistant", "content": "all good, done!"}])
        loop = self._loop(_cfg(tmp.name), mgr, wrk)
        out = loop.run("Do W")
        self.assertEqual(out["status"], "blocked")

    def test_max_iterations_terminates(self):
        tmp = self._repo()
        mgr = clients.MockClient([mgr_msg("continue", "keep going")])
        wrk = clients.MockClient([{"role": "assistant", "content": VALID_WORKER}])
        loop = self._loop(_cfg(tmp.name, max_iterations=2), mgr, wrk)
        out = loop.run("Never ends")
        self.assertEqual(out["status"], "max_iterations")

    def test_repeated_worker_failure_threshold(self):
        tmp = self._repo()
        bad = VALID_WORKER.replace('"done"', '"failed"')
        mgr = clients.MockClient([mgr_msg("continue", "t")])
        wrk = clients.MockClient([{"role": "assistant", "content": bad}])
        loop = self._loop(_cfg(tmp.name, max_consecutive_failures=2), mgr, wrk)
        out = loop.run("Fragile")
        self.assertEqual(out["status"], "failure_threshold")

    def test_api_failure_retry_then_threshold(self):
        tmp = self._repo()
        mgr = clients.MockClient([clients.TransientApiError("down"),
                                  clients.TransientApiError("down"),
                                  clients.TransientApiError("down")])
        wrk = clients.MockClient([{"role": "assistant", "content": VALID_WORKER}])
        loop = self._loop(_cfg(tmp.name, max_api_failures=2), mgr, wrk)
        out = loop.run("API down")
        self.assertEqual(out["status"], "api_failure_threshold")

    def test_resume_after_restart(self):
        tmp = self._repo()
        mgr = clients.MockClient([mgr_msg("complete", "")])
        wrk = clients.MockClient([])
        loop = self._loop(_cfg(tmp.name), mgr, wrk)
        loop.run("First")
        # simulated restart: fresh loop, same state dir
        mgr2 = clients.MockClient([mgr_msg("complete", "")])
        loop2 = self._loop(_cfg(tmp.name), mgr2, wrk)
        self.assertEqual(loop2.state.get_objective(), "First")
        out = loop2.run("")
        self.assertEqual(out["status"], "complete")


if __name__ == "__main__":
    unittest.main()
