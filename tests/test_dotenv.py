"""Regression tests for `.env` configuration loading.

All tests use temporary directories/files and fake secrets. The developer's
real `<repo>/.env` is NEVER read here: every `from_env` call gets an isolated
empty `repo_root`.

Run:  python -m unittest discover -s tests -v   (from repo root)
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_loop import dotenv as dotenv_mod
from agent_loop.config import LoopConfig
from agent_loop.loop import AgentLoop
from agent_loop.roles import RoleChain
from agent_loop.clients import MockClient

FAKE = "fake-secret-0123456789abcdef"


def _clean_env(extra=None):
    old = dict(os.environ)
    for k in ("MANAGER_API_KEY", "WORKER_API_KEY", "WORKER_BACKUP_API_KEY",
              "MANAGER_FALLBACK_API_KEY", "MANAGER_FALLBACK_BASE_URL",
              "MANAGER_FALLBACK_ENABLED", "MANAGER_FALLBACK_MODEL",
              "MANAGER_MODEL", "WORKER_MODEL", "OPENAI_API_KEY",
              "OPENAI_BASE_URL"):
        os.environ.pop(k, None)
    if extra:
        os.environ.update(extra)
    return old


def _restore(old):
    os.environ.clear()
    os.environ.update(old)


def _tmp_repo(testcase, dotenv_text=None):
    tmp = tempfile.TemporaryDirectory()
    testcase.addCleanup(tmp.cleanup)
    if dotenv_text is not None:
        with open(os.path.join(tmp.name, ".env"), "w", encoding="utf-8") as f:
            f.write(dotenv_text)
    return tmp.name


class TestDotenvParser(unittest.TestCase):
    def test_comments_and_blanks_ignored(self):
        out = dotenv_mod.parse_dotenv_text(
            "# full line comment\n\n   \nA=1\n\t# indented comment\nB=2\n")
        self.assertEqual(out, {"A": "1", "B": "2"})

    def test_quotes_and_export_and_equals_in_value(self):
        out = dotenv_mod.parse_dotenv_text(
            'A="quoted value"\nB=\'single\'\nexport C=plain\nD=abc=def==\n')
        self.assertEqual(out, {"A": "quoted value", "B": "single",
                               "C": "plain", "D": "abc=def=="})

    def test_malformed_lines_skipped(self):
        out = dotenv_mod.parse_dotenv_text("NOEQUALS\n=noname\n9BAD=x\nOK=1\n")
        self.assertEqual(out, {"OK": "1"})

    def test_missing_file_is_noop(self):
        self.assertEqual(dotenv_mod.load_dotenv("/nonexistent/.env"), [])
        self.assertEqual(dotenv_mod.dotenv_info("/nonexistent/.env"), (False, 0))


class TestDotenvLoading(unittest.TestCase):
    def test_1_env_values_are_loaded(self):
        root = _tmp_repo(self, "MANAGER_API_KEY=%s\nWORKER_API_KEY=w1\n" % FAKE)
        old = _clean_env()
        try:
            cfg = LoopConfig.from_env(repo_root=root)
        finally:
            _restore(old)
        self.assertEqual(cfg.manager_api_key, FAKE)
        self.assertEqual(cfg.worker_api_key, "w1")

    def test_2_existing_env_overrides_dotenv(self):
        root = _tmp_repo(self, "MANAGER_API_KEY=from-dotenv\n")
        old = _clean_env({"MANAGER_API_KEY": "from-process"})
        try:
            cfg = LoopConfig.from_env(repo_root=root)
        finally:
            _restore(old)
        self.assertEqual(cfg.manager_api_key, "from-process")

    def test_5_secrets_never_in_diagnostics(self):
        root = _tmp_repo(self, "MANAGER_API_KEY=%s\n" % FAKE)
        old = _clean_env()
        try:
            cfg = LoopConfig.from_env(repo_root=root)
            blob = json.dumps(cfg.redacted()) + str(cfg.validate())
            exists, count = dotenv_mod.dotenv_info(os.path.join(root, ".env"))
            diag = blob + "%s %d" % (exists, count)
        finally:
            _restore(old)
        self.assertNotIn(FAKE, diag)

    def test_6_missing_keys_still_detected(self):
        root = _tmp_repo(self, "# empty on purpose\n")
        old = _clean_env()
        try:
            errs = LoopConfig.from_env(repo_root=root).validate()
        finally:
            _restore(old)
        self.assertTrue(any("MANAGER_API_KEY" in e for e in errs))
        self.assertTrue(any("WORKER_API_KEY" in e for e in errs))
        self.assertTrue(any("WORKER_BACKUP_API_KEY" in e for e in errs))

    def test_7_all_four_roles_recognize_credentials(self):
        root = _tmp_repo(self, "\n".join([
            "MANAGER_API_KEY=m-%s" % FAKE,
            "MANAGER_FALLBACK_API_KEY=k-%s" % FAKE,
            "WORKER_API_KEY=w-%s" % FAKE,
            "WORKER_BACKUP_API_KEY=b-%s" % FAKE,
            "MANAGER_FALLBACK_BASE_URL=https://k.example/v1",
            "AGY_ENABLED=false",
            ""]))
        old = _clean_env()
        try:
            from agent_loop.loop import build_chains
            mgr, wrk = build_chains(LoopConfig.from_env(repo_root=root))
        finally:
            _restore(old)
        self.assertEqual(mgr.primary.api_key, "m-" + FAKE)
        self.assertEqual(mgr.fallback.api_key, "k-" + FAKE)
        self.assertEqual(wrk.primary.api_key, "w-" + FAKE)
        self.assertEqual(wrk.fallback.api_key, "b-" + FAKE)

    def test_7b_agy_primary_wired_by_default(self):
        root = _tmp_repo(self, "\n".join([
            "MANAGER_API_KEY=m-%s" % FAKE,
            "WORKER_API_KEY=w-%s" % FAKE,
            "WORKER_BACKUP_API_KEY=b-%s" % FAKE,
            "MANAGER_FALLBACK_ENABLED=false",
            ""]))
        old = _clean_env()
        try:
            from agent_loop.loop import build_chains
            mgr, wrk = build_chains(LoopConfig.from_env(repo_root=root))
        finally:
            _restore(old)
        self.assertEqual(wrk.primary.provider, "agy")
        self.assertEqual([f.model for f in wrk.fallbacks],
                         ["nvidia/nemotron-3-super-120b-a12b:free",
                          "deepseek-ai/deepseek-v4-flash-0731"])
        self.assertEqual(mgr.primary.model,
                         "nvidia/nemotron-3-ultra-550b-a55b:free")

    def test_8_disabled_kimi_does_not_block_primary_run(self):
        root = _tmp_repo(self, "\n".join([
            "MANAGER_API_KEY=m1", "WORKER_API_KEY=w1",
            "WORKER_BACKUP_API_KEY=b1", "MANAGER_FALLBACK_ENABLED=false", ""]))
        old = _clean_env()
        try:
            cfg = LoopConfig.from_env(repo_root=root)
            errs = cfg.validate()
        finally:
            _restore(old)
        self.assertEqual(errs, [])
        self.assertFalse(cfg.manager_fallback_enabled)

    def test_9_env_never_written_to_state(self):
        root = _tmp_repo(self)
        secret = "state-leak-probe-%s" % FAKE
        old = _clean_env({"MANAGER_API_KEY": secret,
                          "WORKER_API_KEY": "w",
                          "WORKER_BACKUP_API_KEY": "b",
                          "MANAGER_FALLBACK_ENABLED": "false"})
        try:
            cfg = LoopConfig.from_env(repo_root=root)
            mgr = RoleChain("manager", MockClient(
                [{"role": "assistant",
                  "content": '{"status": "complete", "assessment": "ok"}'}],
                model="m"))
            wrk = RoleChain("worker", MockClient([], model="w"))
            AgentLoop(cfg, mgr, wrk).run("probe objective")
            found = []
            for dirpath, _, files in os.walk(os.path.join(root, ".agent_loop")):
                for fn in files:
                    with open(os.path.join(dirpath, fn), encoding="utf-8",
                              errors="replace") as f:
                        if secret in f.read():
                            found.append(fn)
        finally:
            _restore(old)
        self.assertEqual(found, [])


class TestCheckEntryPoint(unittest.TestCase):
    def test_10_check_reports_configured_without_leaking(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ)
        for k in ("MANAGER_API_KEY", "WORKER_API_KEY", "WORKER_BACKUP_API_KEY",
                  "MANAGER_FALLBACK_API_KEY", "OPENAI_API_KEY"):
            env.pop(k, None)
        fake = "entrypoint-probe-%s" % FAKE
        env.update({"MANAGER_API_KEY": "m-" + fake,
                    "WORKER_API_KEY": "w-" + fake,
                    "WORKER_BACKUP_API_KEY": "b-" + fake,
                    "MANAGER_FALLBACK_ENABLED": "false"})
        proc = subprocess.run(
            [sys.executable, "-m", "agent_loop", "--check"],
            cwd=repo, capture_output=True, text=True, timeout=120, env=env)
        out = (proc.stdout or "") + (proc.stderr or "")
        self.assertEqual(proc.returncode, 0, msg=out[-2000:])
        self.assertIn("MANAGER PRIMARY", out)
        self.assertIn("MANAGER FALLBACK", out)
        self.assertIn("disabled", out)
        self.assertEqual(out.count("-> configured"), 3)
        self.assertNotIn(fake, out)


if __name__ == "__main__":
    unittest.main()
