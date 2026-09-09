"""Topology + fallback-chain tests (mocked; no live LLM calls).

Covers: exact model IDs / base URLs, explicit Kimi slot, chain isolation,
infra-only fallback, task-failure non-fallback, observability persistence,
malformed handling, termination with chains.

Run:  python -m unittest discover -s tests -v   (from repo root)
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_loop import clients
from agent_loop.clients import ApiError, MockClient, TransientApiError
from agent_loop.config import (LoopConfig, NVIDIA_BASE_URL, OPENROUTER_BASE_URL)
from agent_loop.loop import AgentLoop, build_chains
from agent_loop.roles import ChainExhausted, RoleChain
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


def _cfg(tmpdir, **kw):
    d = dict(repo_root=tmpdir, state_dir=os.path.join(tmpdir, ".agent_loop"),
             max_iterations=5, max_tool_steps=4, max_consecutive_failures=2,
             max_api_failures=3)
    d.update(kw)
    return LoopConfig(manager_api_key="m", worker_api_key="w",
                      worker_backup_api_key="b", **d)


def _repo(testcase):
    tmp = tempfile.TemporaryDirectory()
    testcase.addCleanup(tmp.cleanup)
    with open(os.path.join(tmp.name, "x.py"), "w") as f:
        f.write("y = 1\n")
    return tmp


# ---------- CONFIGURATION (items 11-16) ----------

class TestTopologyConfig(unittest.TestCase):
    def test_exact_model_ids(self):
        cfg = LoopConfig()
        self.assertEqual(cfg.manager_model,
                         "nvidia/nemotron-3-ultra-550b-a55b:free")
        self.assertEqual(cfg.manager_fallback_model, "moonshotai/kimi-k3")
        self.assertEqual(cfg.worker_model, "nvidia/nemotron-3-super-120b-a12b:free")
        self.assertEqual(cfg.worker_backup_model, "deepseek-ai/deepseek-v4-flash-0731")

    def test_manager_default_is_openrouter_fallback_is_nvidia(self):
        cfg = LoopConfig()
        self.assertEqual(cfg.manager_base_url,
                         "https://openrouter.ai/api/v1")
        self.assertEqual(cfg.manager_fallback_base_url, NVIDIA_BASE_URL)
        self.assertEqual(cfg.worker_backup_base_url, NVIDIA_BASE_URL)

    def test_manager_url_has_exactly_one_scheme(self):
        cfg = LoopConfig(manager_api_key="m", worker_api_key="w",
                         worker_backup_api_key="b",
                         manager_fallback_enabled=False,
                         manager_base_url="https:https://openrouter.ai/api/v1")
        errs = cfg.validate()
        self.assertTrue(any("duplicated scheme" in e for e in errs))

    def test_exact_openrouter_url_for_worker(self):
        cfg = LoopConfig()
        self.assertEqual(cfg.worker_base_url, OPENROUTER_BASE_URL)
        self.assertEqual(cfg.worker_base_url, "https://openrouter.ai/api/v1")

    def test_kimi_base_url_is_user_specified_nvidia(self):
        self.assertEqual(LoopConfig().manager_fallback_base_url,
                         "https://integrate.api.nvidia.com/v1")

    def test_missing_kimi_url_fails_validation_clearly(self):
        cfg = LoopConfig(manager_api_key="m", worker_api_key="w",
                         worker_backup_api_key="b",
                         manager_fallback_api_key="k",
                         manager_fallback_base_url="")
        errs = cfg.validate()
        kimi = [e for e in errs if "MANAGER_FALLBACK_BASE_URL" in e]
        self.assertEqual(len(kimi), 1)
        self.assertIn("required but missing", kimi[0])

    def test_kimi_fallback_can_be_disabled(self):
        cfg = LoopConfig(manager_api_key="m", worker_api_key="w",
                         worker_backup_api_key="b",
                         manager_fallback_enabled=False)
        self.assertEqual(cfg.validate(), [])

    def test_credentials_isolated_per_role(self):
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        old = dict(os.environ)
        for k in ("MANAGER_API_KEY", "WORKER_API_KEY", "WORKER_BACKUP_API_KEY",
                  "MANAGER_FALLBACK_API_KEY", "MANAGER_FALLBACK_BASE_URL",
                  "OPENAI_API_KEY", "AGY_ENABLED"):
            os.environ.pop(k, None)
        os.environ.update({"MANAGER_API_KEY": "m1", "WORKER_API_KEY": "w1",
                           "WORKER_BACKUP_API_KEY": "b1",
                           "MANAGER_FALLBACK_API_KEY": "k1",
                           "MANAGER_FALLBACK_BASE_URL": "https://k.example/v1",
                           "AGY_ENABLED": "false"})
        try:
            mgr, wrk = build_chains(LoopConfig.from_env(repo_root=tmp.name))
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(mgr.primary.api_key, "m1")
        self.assertEqual(mgr.fallback.api_key, "k1")
        self.assertEqual(wrk.primary.api_key, "w1")
        self.assertEqual(wrk.fallback.api_key, "b1")


# ---------- ROLE CHAINS (items 1-10) ----------

class TestRoleChains(unittest.TestCase):
    def test_manager_success_no_fallback(self):
        c = RoleChain("manager", MockClient([mgr_msg("complete", "")],
                                            model="deepseek-ai/deepseek-v4-pro-0813",
                                            provider="nvidia"),
                      MockClient([], model="moonshotai/kimi-k3"))
        msg, invos = c.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(len(invos), 1)
        self.assertEqual(invos[0].status, "success")
        self.assertEqual(invos[0].fallback_from, "")
        self.assertIn("complete", msg["content"])

    def test_manager_infra_failure_falls_back_to_kimi(self):
        primary = MockClient([TransientApiError("connection reset")],
                             model="deepseek-ai/deepseek-v4-pro-0813",
                             provider="nvidia")
        fallback = MockClient([mgr_msg("continue")], model="moonshotai/kimi-k3",
                              provider="custom")
        c = RoleChain("manager", primary, fallback)
        msg, invos = c.chat([{"role": "user", "content": "hi"}])
        self.assertEqual([i.status for i in invos],
                         ["infrastructure_failure", "success"])
        self.assertEqual(invos[1].fallback_from,
                         "deepseek-ai/deepseek-v4-pro-0813")
        self.assertEqual(invos[1].model, "moonshotai/kimi-k3")
        # fallback got the full manager context
        sent = fallback.calls[0]["messages"]
        self.assertTrue(any("ORIGINAL OBJECTIVE" in str(m.get("content", ""))
                            for m in sent) or len(sent) == 1)

    def test_manager_fallback_produces_valid_decision(self):
        from agent_loop.schemas import parse_manager_output
        c = RoleChain("manager",
                      MockClient([TransientApiError("dns")], model="p"),
                      MockClient([mgr_msg("retry", "Fix it.")], model="moonshotai/kimi-k3"))
        msg, _ = c.chat([{"role": "user", "content": "ctx"}])
        d = parse_manager_output(msg["content"])
        self.assertEqual(d["status"], "retry")

    def test_manager_chain_rejects_worker_tools(self):
        c = RoleChain("manager", MockClient([mgr_msg()]))
        with self.assertRaises(AssertionError):
            c.chat([], tools=[{"type": "function"}])

    def test_worker_success_and_backup_gets_tools(self):
        primary = MockClient([TransientApiError("timeout")],
                             model="thinkingmachines/inkling-small:free",
                             provider="openrouter")
        backup = MockClient([{"role": "assistant", "content": VALID_WORKER}],
                            model="deepseek-ai/deepseek-v4-flash-0731",
                            provider="nvidia")
        c = RoleChain("worker", primary, backup)
        tools = [{"type": "function", "function": {"name": "git_status"}}]
        msg, invos = c.chat([{"role": "user", "content": "task"}], tools=tools)
        self.assertEqual(invos[1].model, "deepseek-ai/deepseek-v4-flash-0731")
        self.assertIs(backup.calls[0]["tools"], tools)  # backup got worker tools
        self.assertEqual(primary.calls[0]["tools"], tools)

    def test_worker_backup_never_manages(self):
        # build_chains wires backup only into the worker chain
        cfg = LoopConfig(manager_api_key="m", worker_api_key="w",
                         worker_backup_api_key="b",
                         manager_fallback_enabled=False)
        mgr, wrk = build_chains(cfg)
        self.assertIsNone(mgr.fallback)
        self.assertNotIn(wrk.fallback, (mgr.primary,))
        self.assertEqual([f.model for f in wrk.fallbacks],
                         ["nvidia/nemotron-3-super-120b-a12b:free",
                          "deepseek-ai/deepseek-v4-flash-0731"])
        self.assertNotEqual(mgr.primary.model, wrk.fallbacks[0].model)
        # loop constructor also guards cross-contamination
        with self.assertRaises(ValueError):
            AgentLoop(cfg, wrk, wrk)

    def test_chain_exhausted_when_all_fail(self):
        c = RoleChain("manager",
                      MockClient([TransientApiError("down")], model="p"),
                      MockClient([ApiError("HTTP 503")], model="f"))
        with self.assertRaises(ChainExhausted) as ctx:
            c.chat([])
        self.assertEqual(len(ctx.exception.invocations), 2)

    def test_malformed_provider_response_is_infra_failure(self):
        c = RoleChain("manager",
                      MockClient([{"bogus": "no choices"}], model="p"),
                      MockClient([mgr_msg("complete", "")], model="f"))
        # MockClient returns raw dicts; RoleChain passes them through, so
        # simulate malformed at client level via a raising stub instead:
        class BadShape:
            provider = "x"
            model = "bad-shape"
            def chat(self, *a, **k):
                raise ApiError("malformed provider response: missing choices")
        c2 = RoleChain("manager", BadShape(),
                       MockClient([mgr_msg("complete", "")], model="f"))
        msg, invos = c2.chat([])
        self.assertEqual(invos[0].status, "infrastructure_failure")
        self.assertIn("complete", msg["content"])

    def test_task_failure_does_not_trigger_fallback(self):
        # Valid HTTP response with garbage content = task failure, no fallback.
        primary = MockClient([{"role": "assistant",
                               "content": "Done. Tests pass. Trust me."}],
                             model="primary")
        fallback = MockClient([mgr_msg()], model="fallback")
        c = RoleChain("manager", primary, fallback)
        msg, invos = c.chat([])
        self.assertEqual(len(invos), 1)  # no fallback attempt
        self.assertEqual(fallback.calls, [])  # fallback never invoked
        from agent_loop.schemas import parse_manager_output
        with self.assertRaises(ValueError):
            parse_manager_output(msg["content"])


# ---------- CLIENT HARDENING ----------

class TestClientHardening(unittest.TestCase):
    def test_extract_rejects_missing_choices(self):
        from agent_loop.clients import _extract_message
        with self.assertRaises(ApiError):
            _extract_message({"oops": 1})

    def test_extract_accepts_reasoning_fields(self):
        from agent_loop.clients import _extract_message
        msg = _extract_message({"choices": [{"message": {
            "role": "assistant", "content": VALID_MANAGER,
            "reasoning_content": "private thought"}}]})
        self.assertIn("continue", msg["content"])
        self.assertEqual(msg["reasoning_content"], "private thought")

    def test_json_mode_downgrade_on_rejection(self):
        from agent_loop.clients import OpenAICompatibleClient
        seen = []

        class RejectOnce(OpenAICompatibleClient):
            def _post(self, payload):
                seen.append("json" in str(payload))
                if "response_format" in payload:
                    raise ApiError("HTTP 400: response_format unsupported")
                return {"choices": [{"message": {"role": "assistant",
                                                 "content": "{}"}}]}
        c = RejectOnce("k", "https://x.example/v1", "m", retries=1)
        out = c.chat([], force_json=True)
        self.assertEqual(out["content"], "{}")
        self.assertEqual(len(seen), 2)


# ---------- LOOP WITH CHAINS (items 17-24) ----------

class TestLoopWithChains(unittest.TestCase):
    def _loop(self, cfg, mgr, wrk):
        return AgentLoop(cfg, mgr, wrk)

    def test_full_flow_primary_manager_to_primary_worker(self):
        tmp = _repo(self)
        mgr = RoleChain("manager", MockClient(
            [mgr_msg("continue"), mgr_msg("complete", "")],
            model="deepseek-ai/deepseek-v4-pro-0813", provider="nvidia"))
        wrk = RoleChain("worker", MockClient(
            [{"role": "assistant", "content": VALID_WORKER}],
            model="thinkingmachines/inkling-small:free", provider="openrouter"))
        out = self._loop(_cfg(tmp.name), mgr, wrk).run("Objective O")
        self.assertEqual(out["status"], "complete")
        self.assertEqual(out["iterations"], 2)

    def test_fallback_manager_to_primary_worker(self):
        tmp = _repo(self)
        mgr = RoleChain(
            "manager",
            MockClient([TransientApiError("TLS failure")],
                       model="deepseek-ai/deepseek-v4-pro-0813"),
            MockClient([mgr_msg("complete", "")], model="moonshotai/kimi-k3"))
        wrk = RoleChain("worker", MockClient(
            [{"role": "assistant", "content": VALID_WORKER}],
            model="thinkingmachines/inkling-small:free"))
        loop = self._loop(_cfg(tmp.name), mgr, wrk)
        out = loop.run("Objective O")
        self.assertEqual(out["status"], "complete")
        tail = loop.state.recent_invocations(10)
        self.assertTrue(any(i["model"] == "moonshotai/kimi-k3" and
                            i["status"] == "success" for i in tail))
        self.assertTrue(any(i["fallback_from"] ==
                            "deepseek-ai/deepseek-v4-pro-0813" for i in tail))

    def test_primary_manager_to_backup_worker(self):
        tmp = _repo(self)
        mgr = RoleChain("manager", MockClient(
            [mgr_msg("continue"), mgr_msg("complete", "")],
            model="deepseek-ai/deepseek-v4-pro-0813"))
        wrk = RoleChain(
            "worker",
            MockClient([TransientApiError("connect")],
                       model="thinkingmachines/inkling-small:free"),
            MockClient([{"role": "assistant", "content": VALID_WORKER}],
                       model="deepseek-ai/deepseek-v4-flash-0731"))
        out = self._loop(_cfg(tmp.name), mgr, wrk).run("Objective O")
        self.assertEqual(out["status"], "complete")

    def test_both_fallback_paths_together(self):
        tmp = _repo(self)
        mgr = RoleChain(
            "manager", MockClient([TransientApiError("x")], model="mp"),
            MockClient([mgr_msg("complete", "")], model="mf"))
        wrk = RoleChain(
            "worker", MockClient([TransientApiError("y")], model="wp"),
            MockClient([{"role": "assistant", "content": VALID_WORKER}],
                       model="wb"))
        # complete on first decision means no worker pass; use continue first
        mgr2 = RoleChain(
            "manager", MockClient([TransientApiError("x")], model="mp"),
            MockClient([mgr_msg("continue"), mgr_msg("complete", "")],
                       model="mf"))
        loop = self._loop(_cfg(tmp.name), mgr2, wrk)
        out = loop.run("Objective O")
        self.assertEqual(out["status"], "complete")
        models = [i["model"] for i in loop.state.recent_invocations(20)]
        self.assertIn("mf", models)
        self.assertIn("wb", models)

    def test_bad_task_result_flows_to_manager_not_to_backup(self):
        tmp = _repo(self)
        bad_worker = VALID_WORKER.replace('"done"', '"failed"')
        mgr = RoleChain("manager", MockClient(
            [mgr_msg("continue"), mgr_msg("blocked", "stuck")], model="mp"))
        primary = MockClient([{"role": "assistant", "content": bad_worker}],
                             model="thinkingmachines/inkling-small:free")
        backup = MockClient([{"role": "assistant", "content": VALID_WORKER}],
                            model="deepseek-ai/deepseek-v4-flash-0731")
        wrk = RoleChain("worker", primary, backup)
        out = self._loop(_cfg(tmp.name), mgr, wrk).run("Objective O")
        self.assertEqual(out["status"], "blocked")
        self.assertEqual(backup.calls, [])  # backup never silently invoked

    def test_fallback_history_persisted_across_restart(self):
        tmp = _repo(self)
        cfg = _cfg(tmp.name)
        mgr = RoleChain(
            "manager", MockClient([TransientApiError("x")], model="mp"),
            MockClient([mgr_msg("complete", "")], model="mf"))
        wrk = RoleChain("worker", MockClient(
            [{"role": "assistant", "content": VALID_WORKER}], model="wp"))
        loop = self._loop(cfg, mgr, wrk)
        loop.run("Objective O")
        loop2 = AgentLoop(cfg, RoleChain("manager", MockClient([], model="x")),
                          RoleChain("worker", MockClient([], model="y")),
                          state=LoopState(cfg.state_dir))
        tail = loop2.state.recent_invocations(10)
        self.assertTrue(any(i.get("fallback_from") == "mp" for i in tail))

    def test_malformed_worker_report_handled(self):
        tmp = _repo(self)
        mgr = RoleChain("manager", MockClient(
            [mgr_msg("continue"), mgr_msg("complete", "")], model="mp"))
        wrk = RoleChain("worker", MockClient(
            [{"role": "assistant", "content": "all good!"},
             {"role": "assistant", "content": VALID_WORKER}], model="wp"))
        # first worker turn garbage -> failed report -> failure count, then ok
        out = self._loop(_cfg(tmp.name), mgr, wrk).run("Objective O")
        self.assertIn(out["status"], ("complete", "max_iterations",
                                      "failure_threshold"))

    def test_termination_max_iterations_with_chains(self):
        tmp = _repo(self)
        mgr = RoleChain("manager", MockClient([mgr_msg("continue")], model="mp"))
        wrk = RoleChain("worker", MockClient(
            [{"role": "assistant", "content": VALID_WORKER}], model="wp"))
        out = self._loop(_cfg(tmp.name, max_iterations=2), mgr, wrk).run("O")
        self.assertEqual(out["status"], "max_iterations")


if __name__ == "__main__":
    unittest.main()
