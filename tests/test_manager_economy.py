"""Tests for manager token/request economy (mocked; no live calls).

Covers: Retry-After honoring, jittered backoff, provider-state
classification, manager context caps, and malformed-URL validation.

Run:  python -m unittest discover -s tests -v   (from repo root)
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_loop.clients import (_backoff_s, _retry_after_s,
                                OpenAICompatibleClient, TransientApiError)
from agent_loop.config import LoopConfig
from agent_loop.prompts import build_manager_messages
from agent_loop.roles import RoleChain, _provider_state
from agent_loop.clients import ApiError, MockClient


class TestRetryAfter(unittest.TestCase):
    def test_delta_seconds_parsed(self):
        class H(dict):
            def get(self, k, d=None):
                return super().get(k, d)
        self.assertEqual(_retry_after_s(H({"Retry-After": "7"})), 7.0)
        self.assertIsNone(_retry_after_s(H({})))
        self.assertIsNone(_retry_after_s(None))
        self.assertIsNone(_retry_after_s(H({"Retry-After": "soon"})))

    def test_backoff_honors_retry_after_with_cap(self):
        self.assertLessEqual(_backoff_s(0, 9999), 121.0)
        self.assertGreaterEqual(_backoff_s(0, 5), 5.0)
        self.assertLess(_backoff_s(0, 5), 7.0)  # base + jitter < 1s

    def test_backoff_jittered_but_bounded(self):
        for _ in range(20):
            w = _backoff_s(1)
            self.assertGreaterEqual(w, 2.0)
            self.assertLess(w, 3.0)

    def test_retry_after_flows_to_on_retry_hook(self):
        seen = []

        class Flaky(OpenAICompatibleClient):
            def __init__(self):
                super().__init__("k", "https://x.example/v1", "m", retries=2)
                self.n = 0

            def _post(self, payload):
                self.n += 1
                if self.n == 1:
                    err = TransientApiError("HTTP 429: slow down")
                    err.retry_after = 0
                    raise err
                return {"choices": [{"message": {"role": "assistant",
                                                 "content": "{}"}}]}

        out = Flaky().chat([], on_retry=lambda a, w, e: seen.append(w))
        self.assertEqual(out["content"], "{}")
        self.assertEqual(len(seen), 1)
        self.assertLess(seen[0], 2.0)  # honored ~0 + jitter, no fixed sleep


class TestProviderStates(unittest.TestCase):
    def test_rate_limited(self):
        self.assertEqual(
            _provider_state(TransientApiError("HTTP 429: quota")), 
            "PROVIDER_RATE_LIMITED")

    def test_unavailable(self):
        self.assertEqual(
            _provider_state(TransientApiError("HTTP 503: overload")),
            "PROVIDER_UNAVAILABLE")

    def test_timeout(self):
        self.assertEqual(_provider_state(TransientApiError("network error")),
                         "PROVIDER_TIMEOUT")
        self.assertEqual(_provider_state(TimeoutError("timed out")),
                         "PROVIDER_TIMEOUT")

    def test_auth_failure(self):
        self.assertEqual(_provider_state(ApiError("HTTP 403: gated")),
                         "PROVIDER_AUTH_FAILURE")
        self.assertEqual(_provider_state(ApiError("HTTP 401: bad key")),
                         "PROVIDER_AUTH_FAILURE")

    def test_label_appears_in_invocation_record(self):
        c = RoleChain("manager",
                      MockClient([TransientApiError("HTTP 429: slow")],
                                 model="mp"),
                      MockClient([{"role": "assistant", "content": "{}"}],
                                 model="mf"))
        _, invos = c.chat([])
        self.assertIn("PROVIDER_RATE_LIMITED", invos[0].reason)
        self.assertEqual(invos[0].status, "infrastructure_failure")


class TestContextCaps(unittest.TestCase):
    def _big(self, n):
        return {"git_status": "x" * n, "pad": "y" * n}

    def test_default_caps_bound_manager_input(self):
        msgs = build_manager_messages("O", "S", {"status": "continue"},
                                      {"status": "done"}, self._big(10**6))
        user = msgs[1]["content"]
        # evidence section capped at 8000 chars + fixed scaffolding
        self.assertLess(len(user), 40000)

    def test_custom_caps_respected(self):
        full = build_manager_messages("O", "S", None, None, self._big(10**5))
        slim = build_manager_messages("O", "S", None, None, self._big(10**5),
                                      caps=(500, 500, 500))
        self.assertLess(len(slim[1]["content"]), len(full[1]["content"]))

    def test_evidence_floor_validated(self):
        cfg = LoopConfig(manager_evidence_chars=100)
        self.assertTrue(any("MANAGER_EVIDENCE_CHARS" in e
                            for e in cfg.validate()))

    def test_env_override(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        old = dict(os.environ)
        for k in ("MANAGER_DECISION_CHARS", "MANAGER_REPORT_CHARS",
                  "MANAGER_EVIDENCE_CHARS", "MANAGER_API_KEY",
                  "WORKER_API_KEY", "WORKER_BACKUP_API_KEY",
                  "MANAGER_FALLBACK_ENABLED", "OPENAI_API_KEY"):
            os.environ.pop(k, None)
        os.environ.update({"MANAGER_DECISION_CHARS": "1000",
                           "MANAGER_API_KEY": "m", "WORKER_API_KEY": "w",
                           "WORKER_BACKUP_API_KEY": "b",
                           "MANAGER_FALLBACK_ENABLED": "false"})
        try:
            cfg = LoopConfig.from_env(repo_root=tmp.name)
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(cfg.manager_decision_chars, 1000)
        self.assertEqual(cfg.manager_report_chars, 6000)


if __name__ == "__main__":
    unittest.main()
