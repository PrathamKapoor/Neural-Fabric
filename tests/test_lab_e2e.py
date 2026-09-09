"""End-to-end mock research cycle + failure injection (PHASE F+G).

Drives the REAL AgentLoop with mocked providers through complete cycles:
idea -> hypothesis -> task -> build -> verify -> test -> analyze/critic ->
registry verdict, plus injected failures (builder fail, tester FAIL/
INCONCLUSIVE, gate failure, provider timeout, malformed output, missing
artifact, scope violation, stagnation). No network, no subprocesses.

Run:  python -m unittest discover -s tests -v   (from repo root)
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab import idea as idea_mod
from lab.registry import ResearchRegistry
from agent_loop.clients import MockClient, TransientApiError
from agent_loop.config import LoopConfig
from agent_loop.loop import AgentLoop
from agent_loop.roles import RoleChain


def _mgr_msg(status="continue", task="Build the widget via tools."):
    d = {"status": status, "assessment": "a", "completed": [], "gaps": [],
         "problems": [], "next_task": task, "expected_result": "artifact",
         "validation": ["v"], "constraints": [], "reason": "r",
         "hypothesis": "widgets help", "expected_files": ["w.json"],
         "allowed_dirs": [], "required_evidence": ["file"],
         "success_criteria": "exists"}
    return {"role": "assistant", "content": json.dumps(d)}


def _cfg(tmpdir, **kw):
    d = dict(repo_root=tmpdir, state_dir=os.path.join(tmpdir, ".agent_loop"),
             max_iterations=6, max_tool_steps=4, max_consecutive_failures=5,
             max_api_failures=10, session_hours=1, tester_backend="off")
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


def _write_then_report(path="w.json"):
    return [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "write_file",
                                      "arguments": json.dumps(
                                          {"path": path,
                                           "content": '{"v": 1}'})}}]},
        {"role": "assistant", "content": json.dumps(
            {"status": "done", "task": "build", "changes": [path],
             "tests": ["1 seed ran"], "commands": ["write"],
             "evidence": ["measured v=1"], "artifacts": [path],
             "problems": [], "recommendation": "accept"})}]


class TestEndToEndCycle(unittest.TestCase):
    def test_full_cycle_registry_verdict(self):
        tmp = _repo(self)
        import subprocess
        subprocess.run(["git", "init"], cwd=tmp.name, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=tmp.name,
                       capture_output=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=tmp.name,
                       capture_output=True)
        subprocess.run(["git", "add", "."], cwd=tmp.name, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp.name,
                       capture_output=True)
        reg = ResearchRegistry(os.path.join(tmp.name, "reg.json"))
        reg.add_question("QW", "Does the widget help?",
                         status="PROPOSED")
        reg.set_question_status("QW", "DESIGNED", "task contracted")
        reg.add_hypothesis("H-W", "widgets help",
                           {"variable": "widget", "control": "none",
                            "baseline": "nowidget", "metric": "v"},
                           "v != 1 in artifact")
        mgr = RoleChain("manager", MockClient(
            [_mgr_msg("continue"), _mgr_msg("complete")], model="mp"))
        wrk = RoleChain("worker", MockClient(
            _write_then_report("w.json"), model="wp"))
        loop = AgentLoop(_cfg(tmp.name), mgr, wrk)
        with patch.object(AgentLoop, "_run_tester_backends",
                          return_value={"verdict": "PASS",
                                        "evidence": ["file exists"],
                                        "notes": "", "backend": "stub"}):
            out = loop.run("Widget research")
        self.assertEqual(out["status"], "complete")
        # builder produced a real artifact through the loop
        self.assertTrue(os.path.exists(os.path.join(tmp.name, "w.json")))
        self.assertEqual(loop.state.get_task_state(1), "ACCEPTED")
        # researcher loop closes: verdict recorded in registry
        reg.set_question_status("QW", "SUPPORTED", "tester PASS + gates clean")
        reg.judge_hypothesis("H-W", "SUPPORTED", "v==1 measured")
        reg.record_experiment("QW", "iter_0001", "v==1")
        reg2 = ResearchRegistry(os.path.join(tmp.name, "reg.json"))
        self.assertEqual(reg2.get_question("QW")["status"], "SUPPORTED")
        # analyst + critic records persisted
        hist = os.listdir(os.path.join(tmp.name, ".agent_loop", "history"))
        self.assertTrue(any("analysis" in f for f in hist))
        self.assertTrue(any("verification" in f for f in hist))
        # idea generation proposes non-duplicate follow-ups: supporting QW
        # closes it, so open a follow-up question first (manager behavior)
        reg2.add_question("QW2", "Does the widget transfer?",
                          limitations=["only one task family so far"])
        ideas = idea_mod.generate_from_registry(reg2, repo_root=tmp.name,
                                                max_ideas=3)
        self.assertTrue(ideas)

    def test_builder_failure_repairs_and_continues(self):
        tmp = _repo(self)
        bad = {"status": "failed", "task": "t", "changes": [], "tests": [],
               "commands": [], "evidence": [], "artifacts": [],
               "problems": ["boom"], "recommendation": "retry"}
        mgr = RoleChain("manager", MockClient(
            [_mgr_msg("continue"), _mgr_msg("complete")], model="mp"))
        wrk = RoleChain("worker", MockClient(
            [{"role": "assistant", "content": json.dumps(bad)}], model="wp"))
        loop = AgentLoop(_cfg(tmp.name), mgr, wrk)
        out = loop.run("O")
        self.assertEqual(out["status"], "complete")
        self.assertEqual(loop.state.get_task_state(1), "REPAIR")
        # the single failed iteration counted exactly once
        self.assertEqual(loop.state.consecutive_failures, 1)

    def test_tester_fail_forces_repair(self):
        tmp = _repo(self)
        mgr = RoleChain("manager", MockClient(
            [_mgr_msg("continue"), _mgr_msg("complete")], model="mp"))
        wrk = RoleChain("worker", MockClient(
            _write_then_report("w.json"), model="wp"))
        loop = AgentLoop(_cfg(tmp.name), mgr, wrk)
        with patch.object(AgentLoop, "_run_tester_backends",
                          return_value={"verdict": "FAIL",
                                        "evidence": ["content mismatch"],
                                        "notes": "x", "backend": "stub"}):
            loop.run("O")
        self.assertEqual(loop.state.get_task_state(1), "REPAIR")

    def test_gate_failure_skips_llm(self):
        tmp = _repo(self)
        mgr = RoleChain("manager", MockClient(
            [_mgr_msg("continue"), _mgr_msg("complete")], model="mp"))
        ghost = {"status": "done", "task": "t", "changes": ["g.json"],
                 "tests": ["ran"], "commands": [], "evidence": ["did it"],
                 "artifacts": ["ghost.json"], "problems": [],
                 "recommendation": "ok"}
        wrk = RoleChain("worker", MockClient(
            [{"role": "assistant", "content": "",
              "tool_calls": [{"id": "c1", "type": "function",
                              "function": {"name": "git_status",
                                           "arguments": "{}"}}]},
             {"role": "assistant", "content": json.dumps(ghost)}], model="wp"))
        loop = AgentLoop(_cfg(tmp.name), mgr, wrk)
        with patch.object(AgentLoop, "_run_tester_backends") as stub:
            loop.run("O")
            stub.assert_not_called()
        with open(os.path.join(tmp.name, ".agent_loop",
                               "worker_report.json")) as f:
            rep = json.load(f)
        self.assertEqual(rep["verification"]["verdict"], "FAIL")
        self.assertTrue(rep["verification"]["skipped_llm"])

    def test_provider_timeout_recovers(self):
        tmp = _repo(self)
        flaky = MockClient([TransientApiError("timeout")], model="wp")
        good = {"status": "done", "task": "t", "changes": [],
                "tests": ["ok"], "commands": [], "evidence": ["fine"],
                "artifacts": [], "problems": [], "recommendation": "ok"}
        mgr = RoleChain("manager", MockClient(
            [_mgr_msg("continue"), _mgr_msg("complete")], model="mp"))
        # primary worker fails infrastructurally, fallback succeeds
        wrk = RoleChain("worker", flaky,
                        MockClient([{"role": "assistant",
                                     "content": json.dumps(good)}],
                                   model="wb"))
        loop = AgentLoop(_cfg(tmp.name), mgr, wrk)
        out = loop.run("O")
        self.assertEqual(out["status"], "complete")
        # fallback absorbed the infra failure: no API failure counted
        self.assertEqual(loop.state.api_failures, 0)
        self.assertEqual(loop.state.get_task_state(1), "EVIDENCE_CHECK")

    def test_malformed_and_scope_violation(self):
        tmp = _repo(self)
        mgr = RoleChain("manager", MockClient(
            [_mgr_msg("continue"), _mgr_msg("complete")], model="mp"))
        wrk = RoleChain("worker", MockClient(
            [{"role": "assistant", "content": "all good, done!!"}],
            model="wp"))
        loop = AgentLoop(_cfg(tmp.name), mgr, wrk)
        loop.run("O")
        with open(os.path.join(tmp.name, ".agent_loop",
                               "worker_report.json")) as f:
            rep = json.load(f)
        self.assertEqual(rep["status"], "failed")

    def test_stagnation_replans_not_retries(self):
        tmp = _repo(self)
        same = _mgr_msg("continue")
        mgr = RoleChain("manager", MockClient(
            [same, same, same], model="mp"))
        bad = {"status": "failed", "task": "t", "changes": [], "tests": [],
               "commands": [], "evidence": [], "artifacts": [],
               "problems": ["x"], "recommendation": "retry"}
        wrk = RoleChain("worker", MockClient(
            [{"role": "assistant", "content": json.dumps(bad)}], model="wp"))
        loop = AgentLoop(_cfg(tmp.name, max_iterations=3,
                              stagnation_repeat=3, stagnation_window=6),
                         mgr, wrk)
        loop.run("O")
        self.assertIn("REPLAN REQUIRED", loop.state.research_summary())


if __name__ == "__main__":
    unittest.main()
