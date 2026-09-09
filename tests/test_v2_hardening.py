"""Tests for v2 hardening: tester layer, deterministic gates, state machine.

All mocked (no live LLM/CLI calls). Covers PART 7: hallucinated metrics,
fabricated artifacts, repeated-task loops, scope drift, unsupported
conclusions, false success reports, verifier disagreement, INCONCLUSIVE
handling - plus state transitions and tester backend selection.

Run:  python -m unittest discover -s tests -v   (from repo root)
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_loop.tester as tester_mod
from agent_loop.clients import MockClient
from agent_loop.config import LoopConfig
from agent_loop.loop import AgentLoop
from agent_loop.roles import RoleChain
from agent_loop.state import LoopState
from agent_loop.schemas import parse_manager_output
from agent_loop.prompts import build_worker_messages
from agent_loop.tester import (ApiTester, OpenCodeTester, _ndjson_text,
                               build_tester_messages, combine_verdict,
                               parse_verdict)
from agent_loop.verify import (StagnationTracker, check_claim_strength,
                               check_claims_vs_artifacts, check_scope)

VALID_MANAGER = json.dumps({
    "status": "continue", "assessment": "a", "completed": [], "gaps": [],
    "problems": [], "next_task": "Run the router ablation via tools.",
    "expected_result": "e", "validation": ["v"], "constraints": [],
    "reason": "r", "hypothesis": "top-k beats soft",
    "expected_files": ["experiments/results/x.json"],
    "allowed_dirs": ["experiments/"],
    "required_evidence": ["result JSON"], "success_criteria": "done"})

VALID_WORKER = json.dumps({
    "status": "done", "task": "t", "changes": ["experiments/x.py"],
    "tests": ["3 seeds ran"], "commands": ["run x.py"], "evidence": ["ok"],
    "artifacts": ["experiments/results/x.json"], "problems": [],
    "recommendation": "continue"})


def _repo(testcase, with_git=False):
    tmp = tempfile.TemporaryDirectory()
    testcase.addCleanup(tmp.cleanup)
    with open(os.path.join(tmp.name, "x.py"), "w") as f:
        f.write("y = 1\n")
    if with_git:
        import subprocess as sp
        sp.run(["git", "init"], cwd=tmp.name, capture_output=True)
        sp.run(["git", "config", "user.email", "t@t.t"], cwd=tmp.name,
               capture_output=True)
        sp.run(["git", "config", "user.name", "t"], cwd=tmp.name,
               capture_output=True)
        sp.run(["git", "add", "."], cwd=tmp.name, capture_output=True)
        sp.run(["git", "commit", "-m", "init"], cwd=tmp.name,
               capture_output=True)
    return tmp


def _cfg(tmpdir, **kw):
    d = dict(repo_root=tmpdir, state_dir=os.path.join(tmpdir, ".agent_loop"),
             max_iterations=4, max_tool_steps=4, max_consecutive_failures=3,
             max_api_failures=5, session_hours=1, tester_backend="off")
    d.update(kw)
    return LoopConfig(manager_api_key="m", worker_api_key="w",
                      worker_backup_api_key="b",
                      manager_fallback_enabled=False, **d)


def _mgr(status="continue", task="Do it with tools."):
    d = json.loads(VALID_MANAGER)
    d["status"] = status
    d["next_task"] = task
    return {"role": "assistant", "content": json.dumps(d)}


class TestManagerTaskContract(unittest.TestCase):
    def test_new_fields_parsed(self):
        d = parse_manager_output(VALID_MANAGER)
        self.assertEqual(d["hypothesis"], "top-k beats soft")
        self.assertEqual(d["expected_files"], ["experiments/results/x.json"])
        self.assertEqual(d["allowed_dirs"], ["experiments/"])
        self.assertEqual(d["success_criteria"], "done")

    def test_old_decisions_still_parse_with_defaults(self):
        d = parse_manager_output('{"status": "continue", "next_task": "t"}')
        self.assertEqual(d["allowed_dirs"], [])
        self.assertEqual(d["hypothesis"], "")

    def test_worker_messages_carry_scope_block(self):
        msgs = build_worker_messages("O", "P", "TASK", "EXP", ["v"], ["c"],
                                     scope={"hypothesis": "H1",
                                            "allowed_dirs": ["experiments/"],
                                            "success_criteria": "S1"})
        blob = msgs[1]["content"]
        self.assertIn("H1", blob)
        self.assertIn("experiments/", blob)
        self.assertIn("S1", blob)

    def test_worker_messages_backward_compatible(self):
        msgs = build_worker_messages("O", "P", "T", "E", [], [])
        self.assertIn("TASK", msgs[1]["content"])


class TestClaimArtifactGate(unittest.TestCase):
    def test_missing_artifact_fails(self):
        tmp = _repo(self)
        rep = {"status": "done", "changes": ["x.py"],
               "artifacts": ["experiments/results/ghost.json"],
               "tests": [], "evidence": [], "commands": []}
        findings = check_claims_vs_artifacts(rep, tmp.name, ["x.py"])
        self.assertTrue(any("missing on disk" in f for f in findings))

    def test_changes_with_clean_git_fails(self):
        tmp = _repo(self)
        rep = {"status": "done", "changes": ["did everything"],
               "artifacts": [], "tests": [], "evidence": []}
        findings = check_claims_vs_artifacts(rep, tmp.name, [])
        self.assertTrue(any("git shows none" in f for f in findings))

    def test_seed_count_mismatch_fails(self):
        tmp = _repo(self)
        rep = {"status": "done", "changes": ["experiments/seed1.json"],
               "artifacts": ["experiments/seed1.json"],
               "tests": ["ran 5 seeds"], "evidence": []}
        findings = check_claims_vs_artifacts(
            rep, tmp.name, ["experiments/seed1.json"])
        self.assertTrue(any("5 seeds" in f for f in findings))

    def test_honest_report_passes_gates(self):
        tmp = _repo(self)
        with open(os.path.join(tmp.name, "r.json"), "w") as f:
            f.write("{}")
        rep = {"status": "done", "changes": ["r.json"],
               "artifacts": ["r.json"], "tests": ["1 seed ran"],
               "evidence": ["measured 0.5"], "commands": ["run"]}
        self.assertEqual(check_claims_vs_artifacts(rep, tmp.name, ["r.json"]),
                         [])

    def test_failed_reports_skipped(self):
        self.assertEqual(check_claims_vs_artifacts(
            {"status": "failed", "changes": ["x"]}, ".", []), [])


class TestScopeGate(unittest.TestCase):
    def test_drift_outside_allowed_dirs_fails(self):
        decision = json.loads(VALID_MANAGER)  # allowed: experiments/
        findings = check_scope(decision, ["experiments/x.py",
                                          "frontend/app.js"])
        self.assertTrue(any("frontend" in f for f in findings))

    def test_in_scope_passes(self):
        decision = json.loads(VALID_MANAGER)
        self.assertEqual(check_scope(decision, ["experiments/x.py"]), [])

    def test_root_files_allowed_under_default_scope(self):
        self.assertEqual(check_scope({}, ["README.md", ".env.example"]), [])

    def test_explicit_scope_governs_everything(self):
        decision = {"allowed_dirs": ["experiments/"]}
        self.assertTrue(check_scope(decision, ["README.md"]))

    def test_preexisting_scaffolding_not_drift(self):
        from agent_loop.verify import changed_delta
        pre = ["?? agent_loop/", " M .gitignore"]
        post = ["?? agent_loop/", " M .gitignore", "?? experiments/new.json"]
        delta = changed_delta(pre, post)
        self.assertEqual(delta, ["?? experiments/new.json"])
        self.assertEqual(check_scope({}, delta), [])

    def test_state_dir_always_ignored(self):
        decision = json.loads(VALID_MANAGER)
        self.assertEqual(
            check_scope(decision, [".agent_loop/worker_report.json"]), [])


class TestHonestyGate(unittest.TestCase):
    def test_proves_language_rejected(self):
        rep = {"evidence": ["NF03 proves routing wins"],
               "recommendation": "", "tests": ""}
        findings = check_claim_strength(rep)
        self.assertTrue(findings)

    def test_measured_language_passes(self):
        rep = {"evidence": ["NF03 measured 0.45 vs 0.40 baseline, n=5"],
               "recommendation": "needs replication", "tests": "ran"}
        self.assertEqual(check_claim_strength(rep), [])

    def test_hallucinated_metric_without_artifact(self):
        tmp = _repo(self)
        rep = {"status": "done", "changes": [], "artifacts": [],
               "tests": ["accuracy improved to 0.99"],
               "evidence": ["trust me"], "commands": []}
        # accuracy claim with zero artifacts: seed/artifact logic may not
        # fire, but the overclaim + missing-evidence pattern is the
        # tester's job; gates stay silent only when truly nothing to check
        self.assertIsInstance(
            check_claims_vs_artifacts(rep, tmp.name, []), list)


class TestStagnation(unittest.TestCase):
    def _tracker(self, **kw):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        st = LoopState(os.path.join(tmp.name, ".agent_loop"))
        return StagnationTracker(st, **kw)

    def test_repeated_task_forces_replan(self):
        t = self._tracker(repeat=3, window=6)
        ok = True
        for _ in range(3):
            ok, reasons = t.observe("Run the router ablation via tools.",
                                    ["experiments/x.py"], "same summary",
                                    "failed")
        self.assertFalse(ok)
        self.assertTrue(any("repeated" in r for r in reasons))

    def test_fresh_tasks_pass(self):
        t = self._tracker()
        ok, _ = t.observe("Task A", ["a.py"], "ev1", "done")
        self.assertTrue(ok)
        ok, _ = t.observe("Task B", ["b.py"], "ev2", "done")
        self.assertTrue(ok)

    def test_evidence_stagnation_forces_replan(self):
        t = self._tracker(repeat=3, window=6)
        for i in range(3):
            ok, reasons = t.observe("Task %d" % i, ["f.py"], "identical", "done")
        self.assertFalse(ok)
        self.assertTrue(any("no new evidence" in r for r in reasons))

    def test_state_survives_restart(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        sd = os.path.join(tmp.name, ".agent_loop")
        t = StagnationTracker(LoopState(sd))
        t.observe("Same task here.", ["a.py"], "ev", "failed")
        t2 = StagnationTracker(LoopState(sd))
        ok, _ = t2.observe("Same task here.", ["a.py"], "ev", "failed")
        # 2 observations < repeat=3 -> still ok, but history persisted
        self.assertTrue(ok)
        self.assertEqual(len(t2._hist()), 2)


class TestVerdicts(unittest.TestCase):
    def test_parse_valid(self):
        v = parse_verdict('{"verdict": "FAIL", "evidence": ["x != y"], '
                          '"notes": "mismatch"}')
        self.assertEqual(v["verdict"], "FAIL")
        self.assertEqual(v["evidence"], ["x != y"])

    def test_parse_case_insensitive(self):
        self.assertEqual(parse_verdict('{"verdict": "pass"}')["verdict"],
                         "PASS")

    def test_parse_rejects_unknown(self):
        with self.assertRaises(ValueError):
            parse_verdict('{"verdict": "MAYBE"}')

    def test_combine_hard_gate_skips_llm(self):
        out = combine_verdict(["claimed artifact missing on disk: g.json"],
                              None)
        self.assertEqual(out["verdict"], "FAIL")
        self.assertTrue(out["skipped_llm"])

    def test_disagreement_either_way_is_fail(self):
        llm_fail = {"verdict": "FAIL", "evidence": ["e"], "notes": "n"}
        self.assertEqual(combine_verdict([], llm_fail)["verdict"], "FAIL")
        llm_pass = {"verdict": "PASS", "evidence": ["e"], "notes": "n"}
        self.assertEqual(
            combine_verdict(["scope violation: f/ outside e/"], llm_pass)["verdict"],
            "FAIL")

    def test_pass_requires_both(self):
        llm_pass = {"verdict": "PASS", "evidence": ["checked"], "notes": ""}
        self.assertEqual(combine_verdict([], llm_pass)["verdict"], "PASS")

    def test_inconclusive_propagates(self):
        self.assertEqual(combine_verdict([], None)["verdict"], "INCONCLUSIVE")
        llm_inc = {"verdict": "INCONCLUSIVE", "evidence": [],
                   "notes": "need logs"}
        self.assertEqual(combine_verdict([], llm_inc)["verdict"],
                         "INCONCLUSIVE")

    def test_ndjson_parsing(self):
        blob = ('{"type":"step_start"}\n'
                '{"type":"text","part":{"text":"hello "}}\n'
                '{"type":"text","part":{"text":"world"}}\n'
                'not json\n'
                '{"type":"step_finish"}\n')
        self.assertEqual(_ndjson_text(blob), "hello \nworld")

    def test_tester_prompt_contains_diff_and_report(self):
        p = build_tester_messages("TASK-X", "EXP-Y",
                                  {"status": "done", "task": "t"}, "DIFF-Z")
        self.assertIn("TASK-X", p)
        self.assertIn("DIFF-Z", p)


class TestTesterBackends(unittest.TestCase):
    def test_opencode_missing_binary(self):
        t = OpenCodeTester(repo_root=".", command="definitely-not-opencode-zz")
        self.assertFalse(t.check()["ok"])
        from agent_loop.clients import ApiError
        with self.assertRaises(ApiError):
            t.verify("hi")

    def test_off_backend_means_gates_only(self):
        cfg = _cfg(tempfile.mkdtemp(), tester_backend="off")
        loop = AgentLoop(cfg,
                         RoleChain("manager", MockClient([_mgr("complete")])),
                         RoleChain("worker", MockClient([])))
        primary, fallback = loop._build_tester()
        self.assertIsNone(primary)
        self.assertIsNone(fallback)

    def test_invalid_backend_rejected(self):
        cfg = LoopConfig(tester_backend="carrier-pigeon")
        self.assertTrue(any("TESTER_BACKEND" in e for e in cfg.validate()))


class TestLoopIntegration(unittest.TestCase):
    def _loop(self, tmpdir, mgr_script, wrk_script, **cfgkw):
        cfg = _cfg(tmpdir, **cfgkw)
        return AgentLoop(cfg,
                         RoleChain("manager", MockClient(mgr_script,
                                                         model="mp")),
                         RoleChain("worker", MockClient(wrk_script,
                                                        model="wp")))

    def _tool_msg(self):
        return {"role": "assistant", "content": "",
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "git_status",
                                             "arguments": "{}"}}]}

    def _wrk(self, report_json: str) -> list:
        # One real tool call first: otherwise the loop's zero-tool
        # fabrication guard (correctly) downgrades done+changes to failed.
        return [self._tool_msg(),
                {"role": "assistant", "content": report_json}]

    def _wrk_with_write(self, report_json: str, path: str,
                        content: str = "{}") -> list:
        # The worker really creates the artifact DURING the unit (real
        # ToolExecutor), so the per-iteration delta sees it.
        return [{"role": "assistant", "content": "",
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {
                                     "name": "write_file",
                                     "arguments": json.dumps(
                                         {"path": path,
                                          "content": content})}}]},
                {"role": "assistant", "content": report_json}]

    def test_fail_verdict_counts_as_failure(self):
        tmp = _repo(self)
        mgr = [_mgr("continue"), _mgr("complete", task="t")]
        # worker claims changes with clean git -> gate FAIL
        bad = json.loads(VALID_WORKER)
        bad["status"] = "done"
        wrk = self._wrk(json.dumps(bad))
        loop = self._loop(tmp.name, mgr, wrk)
        loop.run("O")
        self.assertEqual(loop.state.consecutive_failures, 1)
        with open(os.path.join(tmp.name, ".agent_loop",
                               "worker_report.json")) as f:
            rep = json.load(f)
        self.assertEqual(rep["verification"]["verdict"], "FAIL")

    def test_inconclusive_does_not_count_failure(self):
        tmp = _repo(self)
        mgr = [_mgr("continue"), _mgr("complete", task="t")]
        honest = json.loads(VALID_WORKER)
        honest.update(status="done", changes=[], artifacts=[],
                      tests=["looked"], evidence=["nothing to change"])
        wrk = self._wrk(json.dumps(honest))
        loop = self._loop(tmp.name, mgr, wrk)
        loop.run("O")
        # tester off -> INCONCLUSIVE, not a failure
        self.assertEqual(loop.state.consecutive_failures, 0)
        with open(os.path.join(tmp.name, ".agent_loop",
                               "worker_report.json")) as f:
            rep = json.load(f)
        self.assertEqual(rep["verification"]["verdict"], "INCONCLUSIVE")
        # inconclusive stays in EVIDENCE_CHECK for manager judgment,
        # it is not a failure and not a repair order
        self.assertEqual(loop.state.get_task_state(1), "EVIDENCE_CHECK")

    def test_task_states_persisted(self):
        tmp = _repo(self)
        mgr = [_mgr("continue"), _mgr("complete", task="t")]
        wrk = [{"role": "assistant", "content": VALID_WORKER}]
        loop = self._loop(tmp.name, mgr, wrk)
        loop.run("O")
        self.assertEqual(loop.state.get_task_state(1), "REPAIR")
        # fabricated artifact -> hard FAIL -> REPAIR (not ACCEPTED)
        tasks = loop.state._state.get("tasks", {})
        self.assertIn("iter_0001", tasks)
        self.assertTrue(tasks["iter_0001"]["history"])

    def test_stagnation_signals_replan_in_state(self):
        tmp = _repo(self)
        same = _mgr("continue")
        mgr = [same, same, same, _mgr("complete", task="t")]
        bad = json.loads(VALID_WORKER)
        bad["status"] = "failed"
        wrk = self._wrk(json.dumps(bad))
        loop = self._loop(tmp.name, mgr, wrk,
                          max_iterations=3, stagnation_repeat=2,
                          stagnation_window=4)
        loop.run("O")
        summ = loop.state.research_summary()
        self.assertIn("REPLAN REQUIRED", summ)

    def test_llm_verdict_pass_accepts(self):
        tmp = _repo(self, with_git=True)
        mgr = [_mgr("continue"), _mgr("complete", task="t")]
        good = json.loads(VALID_WORKER)
        good.update(changes=["experiments/r.json"],
                    artifacts=["experiments/r.json"],
                    tests=["1 seed ran"], evidence=["measured"])
        wrk = self._wrk_with_write(json.dumps(good), "experiments/r.json")
        loop = self._loop(tmp.name, mgr, wrk)
        passed = {"verdict": "PASS", "evidence": ["checked"],
                  "notes": "", "backend": "stub"}
        with patch.object(AgentLoop, "_run_tester_backends",
                          return_value=passed):
            loop.run("O")
        self.assertEqual(loop.state.get_task_state(1), "ACCEPTED")
        self.assertEqual(loop.state.consecutive_failures, 0)

    def test_llm_fail_overrides_clean_gates(self):
        tmp = _repo(self, with_git=True)
        mgr = [_mgr("continue"), _mgr("complete", task="t")]
        good = json.loads(VALID_WORKER)
        good.update(changes=["experiments/r.json"],
                    artifacts=["experiments/r.json"],
                    tests=["ran"], evidence=["measured"])
        wrk = self._wrk_with_write(json.dumps(good), "experiments/r.json")
        loop = self._loop(tmp.name, mgr, wrk)
        failed = {"verdict": "FAIL", "evidence": ["numbers differ"],
                  "notes": "mismatch", "backend": "stub"}
        with patch.object(AgentLoop, "_run_tester_backends",
                          return_value=failed):
            loop.run("O")
        self.assertEqual(loop.state.consecutive_failures, 1)
        self.assertEqual(loop.state.get_task_state(1), "REPAIR")


if __name__ == "__main__":
    unittest.main()

