"""Tests for the lab/ research extensions (mocked; hermetic temp dirs).

Covers: registry CRUD + hypothesis enforcement, experiment envelopes +
multi-seed honesty, claim classification, doc reconciliation, critic
alternatives + verdicts, researcher novelty attack, idea gates, and the
analyst/critic loop integration (incl. self-audit on doc changes).

Run:  python -m unittest discover -s tests -v   (from repo root)
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab import analyze, critic, experiment, idea, registry, researcher
from lab.registry import ResearchRegistry
from agent_loop.clients import MockClient
from agent_loop.config import LoopConfig
from agent_loop.loop import AgentLoop
from agent_loop.roles import RoleChain


def _tmp(testcase):
    tmp = tempfile.TemporaryDirectory()
    testcase.addCleanup(tmp.cleanup)
    return tmp


class TestRegistry(unittest.TestCase):
    def test_question_lifecycle(self):
        tmp = _tmp(self)
        reg = ResearchRegistry(os.path.join(tmp.name, "reg.json"))
        reg.add_question("Q1", "Does X beat Y?", status="PROPOSED")
        reg.set_question_status("Q1", "EXPERIMENTING", "started")
        reg.record_experiment("Q1", "nf99", "acc 0.9")
        q = reg.get_question("Q1")
        self.assertEqual(q["status"], "EXPERIMENTING")
        self.assertIn("nf99", q["experiments"])
        self.assertTrue(any(h["status"] == "EXPERIMENTING"
                            for h in q["history"]))
        self.assertIn(q, reg.open_questions())
        # persistence across instances
        reg2 = ResearchRegistry(os.path.join(tmp.name, "reg.json"))
        self.assertEqual(reg2.get_question("Q1")["status"], "EXPERIMENTING")

    def test_bad_status_rejected(self):
        tmp = _tmp(self)
        reg = ResearchRegistry(os.path.join(tmp.name, "r.json"))
        with self.assertRaises(ValueError):
            reg.add_question("Q", "t", status="DONE-ISH")

    def test_hypothesis_requires_falsification(self):
        tmp = _tmp(self)
        reg = ResearchRegistry(os.path.join(tmp.name, "r.json"))
        with self.assertRaises(ValueError):
            reg.add_hypothesis("H1", "X wins", {"variable": "v"}, "")
        with self.assertRaises(ValueError):
            reg.add_hypothesis("H2", "X wins", {"variable": "v"}, "none")
        h = reg.add_hypothesis(
            "H3", "X wins",
            {"variable": "routing", "control": "fixed",
             "baseline": "mlp", "metric": "acc"},
            "no compute reduction at matched accuracy")
        self.assertEqual(h["verdict"], "UNTESTED")
        reg.judge_hypothesis("H3", "REFUTED", "p=0.4 null")
        self.assertEqual(
            ResearchRegistry(os.path.join(tmp.name, "r.json"))
            ._data["hypotheses"]["H3"]["verdict"], "REFUTED")

    def test_failure_patterns_accumulate(self):
        tmp = _tmp(self)
        reg = ResearchRegistry(os.path.join(tmp.name, "r.json"))
        reg.record_failure_pattern("routing collapse", 1, "gate: entropy")
        reg.record_failure_pattern("routing collapse", 2)
        pats = reg.failure_patterns()
        self.assertEqual(pats[0]["occurrences"], 3)


class TestExperiment(unittest.TestCase):
    def test_run_seeds_reports_all_seeds(self):
        out = experiment.run_seeds(
            lambda seed: {"acc": 0.9 if seed != 2 else 0.5}, seeds=[1, 2, 3])
        self.assertEqual(len(out["per_seed"]), 3)
        self.assertEqual(out["seeds"], [1, 2, 3])
        # instability preserved, not averaged away
        self.assertLess(out["acc"]["mean"], 0.9)
        self.assertGreater(out["acc"]["std"], 0.1)

    def test_run_seeds_rejects_nondict(self):
        with self.assertRaises(ValueError):
            experiment.run_seeds(lambda seed: 0.9, seeds=[1])

    def test_budget_parity_flags_mismatch(self):
        base = {"params": 1000, "flops": 2000, "epochs": 30,
                "dataset": "syn", "split": "a"}
        cand = {"params": 5000, "flops": 2000, "epochs": 30,
                "dataset": "syn", "split": "a"}
        findings = experiment.check_budget_parity(base, cand)
        self.assertTrue(any("params" in f for f in findings))
        self.assertEqual(experiment.check_budget_parity(base, dict(base)), [])

    def test_envelope_keys(self):
        env = experiment.build_envelope("nf99", "H?", {"lr": 0.1},
                                        {"name": "syn"}, {"name": "mlp"},
                                        {"epochs": 1}, {"acc": 0.5},
                                        {"flops": 10}, ["a.json"])
        for k in ("experiment_id", "git_commit", "metrics", "timestamp",
                  "status", "hypothesis", "budgets", "artifacts"):
            self.assertIn(k, env)

    def test_write_artifact_roundtrip(self):
        tmp = _tmp(self)
        p = experiment.write_artifact(
            os.path.join(tmp.name, "sub", "a.json"), {"x": 1})
        self.assertEqual(json.load(open(p))["x"], 1)


class TestAnalyze(unittest.TestCase):
    def test_claim_kinds(self):
        self.assertEqual(analyze.claim_kind("measured accuracy 0.9 on 5 seeds"),
                         "MEASURED")
        self.assertEqual(analyze.claim_kind("ratio computed from flops"),
                         "DERIVED")
        self.assertEqual(analyze.claim_kind("this suggests specialization"),
                         "INFERRED")
        self.assertEqual(analyze.claim_kind("might transfer to real data"),
                         "SPECULATIVE")

    def test_reconcile_catches_stale_integer(self):
        tmp = _tmp(self)
        import pathlib
        (pathlib.Path(tmp.name) / "doc.md").write_text(
            "top-1 uses 9966 FLOPs and 41.9 percent\n", encoding="utf-8")
        findings = analyze.reconcile_numbers(
            ["doc.md"], {"flops": 10004.8}, repo_root=tmp.name)
        self.assertTrue(any("9966" in f for f in findings))

    def test_reconcile_accepts_matching(self):
        tmp = _tmp(self)
        import pathlib
        (pathlib.Path(tmp.name) / "doc.md").write_text(
            "accuracy 0.94 at 10004.8 FLOPs\n", encoding="utf-8")
        findings = analyze.reconcile_numbers(
            ["doc.md"], {"acc": 0.94, "flops": 10004.8}, repo_root=tmp.name)
        self.assertEqual(findings, [])

    def test_reconcile_iteration(self):
        tmp = _tmp(self)
        import pathlib
        (pathlib.Path(tmp.name) / "r.json").write_text('{"v": 5}',
                                                       encoding="utf-8")
        (pathlib.Path(tmp.name) / "doc.md").write_text(
            "value is 5 and 9999\n", encoding="utf-8")
        findings = analyze.reconcile_iteration(
            ["doc.md"], {"artifacts": ["r.json"], "evidence_summary": ""},
            repo_root=tmp.name)
        self.assertTrue(any("9999" in f for f in findings))
        self.assertFalse(any(": 5 [" in f for f in findings))


class TestCritic(unittest.TestCase):
    def test_weak_evidence_generates_alternatives(self):
        alts = critic.generate_alternatives({"n_seeds": 1})
        names = [a["alternative"] for a in alts]
        self.assertIn("seed_artifact", names)
        self.assertIn("implementation_bug", names)
        self.assertTrue(all(a["discriminating_experiment"] for a in alts))

    def test_strong_evidence_still_gets_baseline_checks(self):
        alts = critic.generate_alternatives({"n_seeds": 5,
                                             "budgets_matched": True,
                                             "splits_shared": False,
                                             "variance_reported": True,
                                             "mechanism": ""})
        names = [a["alternative"] for a in alts]
        self.assertNotIn("seed_artifact", names)
        self.assertIn("implementation_bug", names)

    def test_reject_strong_language(self):
        out = critic.critique_conclusion("NF03 proves routing wins",
                                         {"n_seeds": 5})
        self.assertEqual(out["verdict"], "REJECT")

    def test_ok_honest_conclusion(self):
        out = critic.critique_conclusion(
            "measured 0.94 vs 0.948 baseline, null result",
            {"n_seeds": 5, "baseline_compared": True, "replicated": True,
             "negative_checked": True})
        self.assertEqual(out["verdict"], "OK")


class TestResearcher(unittest.TestCase):
    def test_finds_local_implementation(self):
        tmp = _tmp(self)
        os.makedirs(os.path.join(tmp.name, "fabric"))
        with open(os.path.join(tmp.name, "fabric", "router.py"), "w") as f:
            f.write("def route_topk(x, k): ...\n")
        with open(os.path.join(tmp.name, "notes.md"), "w") as f:
            f.write("# routing survey\n")
        atk = researcher.attack_novelty("top-k routing study", ["topk", "rout"],
                                        repo_root=tmp.name)
        self.assertTrue(atk["code_hits"])
        self.assertIn(atk["verdict"], ("ESTABLISHED", "NARROW_GAP"))

    def test_open_when_nothing_found(self):
        tmp = _tmp(self)
        with open(os.path.join(tmp.name, "a.py"), "w") as f:
            f.write("x = 1\n")
        atk = researcher.attack_novelty("zzz quark routing", ["zzzquark"],
                                        repo_root=tmp.name)
        self.assertEqual(atk["verdict"], "OPEN")
        self.assertFalse(atk["web_searched"])

    def test_web_fetcher_used_when_given(self):
        tmp = _tmp(self)
        atk = researcher.attack_novelty(
            "i", ["i"], repo_root=tmp.name,
            web_fetcher=lambda kw: [{"source": "arxiv:1", "claim": "exists"}])
        self.assertTrue(atk["web_searched"])
        self.assertTrue(atk["web_hits"])


class TestIdea(unittest.TestCase):
    def test_generates_checked_ideas(self):
        from lab.idea import generate_from_registry, passes
        tmp = _tmp(self)
        from lab.registry import ResearchRegistry
        reg = ResearchRegistry(os.path.join(tmp.name, "r.json"))
        reg.add_question("Q1", "Does routing transfer?",
                         limitations=["only synthetic so far"])
        ideas = generate_from_registry(reg, repo_root=tmp.name, max_ideas=3)
        self.assertTrue(ideas)
        for i in ideas:
            for k in ("novelty", "testability", "not_duplicate",
                      "feasibility", "baseline", "falsification"):
                self.assertIn(k, i["checks"])
            self.assertEqual(i["passes"], passes(i["checks"]))

    def test_no_duplicate_ideas(self):
        from lab.idea import run_checks
        from lab.registry import ResearchRegistry
        tmp = _tmp(self)
        reg = ResearchRegistry(os.path.join(tmp.name, "r.json"))
        reg.add_idea("Does X transfer?", {})
        checks = run_checks("Does X transfer?", "transfer study",
                            repo_root=tmp.name, registry=reg)
        self.assertFalse(checks["not_duplicate"])


def _cfg(tmpdir, **kw):
    d = dict(repo_root=tmpdir, state_dir=os.path.join(tmpdir, ".agent_loop"),
             max_iterations=3, max_tool_steps=4, max_consecutive_failures=3,
             max_api_failures=5, session_hours=1, tester_backend="off")
    d.update(kw)
    return LoopConfig(manager_api_key="m", worker_api_key="w",
                      worker_backup_api_key="b",
                      manager_fallback_enabled=False, **d)


def _mgr_msg(status="continue"):
    d = {"status": status, "assessment": "a", "completed": [], "gaps": [],
         "problems": [], "next_task": "Do the thing via git_status.",
         "expected_result": "e", "validation": ["v"], "constraints": [],
         "reason": "r"}
    return {"role": "assistant", "content": json.dumps(d)}


class TestAnalystLoopIntegration(unittest.TestCase):
    def test_analysis_recorded_and_fed_to_manager(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with open(os.path.join(tmp.name, "x.py"), "w") as f:
            f.write("y = 1\n")
        good = {"status": "done", "task": "t", "changes": [],
                "tests": ["measured 0.9"], "commands": ["git_status"],
                "evidence": ["measured routing entropy 0.3"],
                "artifacts": [], "problems": [], "recommendation":
                "continue, null result vs baseline"}
        mgr_client = MockClient([_mgr_msg("continue"), _mgr_msg("complete")],
                                  model="mp")
        mgr = RoleChain("manager", mgr_client)
        wrk = RoleChain("worker", MockClient(
            [{"role": "assistant",
              "content": "",
              "tool_calls": [{"id": "c1", "type": "function",
                              "function": {"name": "git_status",
                                           "arguments": "{}"}}]},
             {"role": "assistant", "content": json.dumps(good)}],
            model="wp"))
        loop = AgentLoop(_cfg(tmp.name), mgr, wrk)
        loop.run("Objective O")
        with open(os.path.join(tmp.name, ".agent_loop",
                               "worker_report.json")) as f:
            rep = json.load(f)
        self.assertIn("analysis", rep)
        self.assertIn("MEASURED", rep["analysis"]["claim_kinds"])
        self.assertTrue(os.path.exists(os.path.join(
            tmp.name, ".agent_loop", "history",
            "iter_0001_analysis.json")))
        # manager saw it: second decision context included verification
        self.assertIn("verification",
                      mgr_client.calls[1]["messages"][1]["content"])

    def test_self_audit_flags_doc_number_mismatch(self):
        from lab import analyze as _analyze
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with open(os.path.join(tmp.name, "r.json"), "w") as f:
            json.dump({"flops": 100}, f)
        with open(os.path.join(tmp.name, "notes.md"), "w") as f:
            f.write("uses 100 flops, also 7777 mystery\n")
        findings = _analyze.reconcile_iteration(
            ["notes.md"], {"artifacts": ["r.json"], "evidence_summary": ""},
            repo_root=tmp.name)
        self.assertTrue(any("7777" in x for x in findings))
        self.assertFalse(any(x.startswith("doc number without iteration "
                                          "evidence: 100.0")
                             for x in findings))


if __name__ == "__main__":
    unittest.main()
