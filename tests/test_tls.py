"""Regression tests for TLS trust behavior (no live network dependency).

Proves: verification always on, no bypasses, platform guards, env-CA
precedence delegation, no secret/cert logging. Suite must stay green.

Run:  python -m unittest discover -s tests -v   (from repo root)
"""
import ast
import io
import os
import ssl
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_loop import tls as tls_mod

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_LOOP_DIR = os.path.join(REPO_ROOT, "agent_loop")

ON_WINDOWS_STORE = (os.name == "nt" and hasattr(ssl, "enum_certificates"))


def _package_sources():
    srcs = {}
    for fn in os.listdir(AGENT_LOOP_DIR):
        if fn.endswith(".py"):
            with open(os.path.join(AGENT_LOOP_DIR, fn), encoding="utf-8") as f:
                srcs[fn] = f.read()
    return srcs


def _docstring_nodes(tree):
    """AST nodes that are docstrings (excluded from the code scan)."""
    found = set()

    def _check(body):
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            found.add(id(body[0].value))

    _check(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _check(node.body)
    return found


def _forbidden_constructs(tree):
    """Find verification-bypass / downgrade constructs in REAL code."""
    skip = _docstring_nodes(tree)
    hits = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.keyword) and node.arg == "verify"
                and isinstance(node.value, ast.Constant)
                and node.value.value is False):
            hits.append("verify=False")
        if isinstance(node, ast.Attribute) and node.attr == "CERT_NONE":
            hits.append("CERT_NONE")
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_create_unverified_context"):
            hits.append("_create_unverified_context")
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Name)
                        and target.id == "check_hostname"
                        and isinstance(node.value, ast.Constant)
                        and node.value.value is False):
                    hits.append("check_hostname=False")
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and "http://" in node.value and id(node) not in skip):
            hits.append("http:// URL literal: %.60s" % node.value)
    return hits


class TestVerificationAlwaysOn(unittest.TestCase):
    def test_context_is_fully_verified(self):
        ctx = tls_mod.create_ssl_context()
        self.assertIsInstance(ctx, ssl.SSLContext)
        self.assertEqual(ctx.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(ctx.check_hostname)

    def test_no_bypass_or_downgrade_in_package_code(self):
        for fn, src in _package_sources().items():
            hits = _forbidden_constructs(ast.parse(src))
            self.assertEqual(hits, [], "%s: %s" % (fn, hits))

    def test_no_logging_of_certs_or_secrets(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            ctx = tls_mod.create_ssl_context()
            pem = tls_mod.windows_store_pem()
        self.assertEqual(buf.getvalue(), "")
        self.assertIsInstance(ctx, ssl.SSLContext)
        self.assertTrue(pem is None or isinstance(pem, str))


class TestPlatformGuards(unittest.TestCase):
    def test_off_windows_returns_none(self):
        with mock.patch.object(tls_mod.os, "name", "posix"):
            self.assertIsNone(tls_mod.windows_store_pem())

    def test_missing_enum_api_returns_none(self):
        with mock.patch.object(tls_mod.os, "name", "nt"):
            with mock.patch.dict(tls_mod.ssl.__dict__):
                tls_mod.ssl.__dict__.pop("enum_certificates", None)
                self.assertIsNone(tls_mod.windows_store_pem())

    def test_store_errors_degrade_to_none(self):
        with mock.patch.object(tls_mod.os, "name", "nt"), \
                mock.patch.object(tls_mod.ssl, "enum_certificates",
                                  side_effect=OSError("store locked"), create=True):
            self.assertIsNone(tls_mod.windows_store_pem())

    def test_non_windows_path_adds_nothing(self):
        # Faithful non-Windows simulation: stdlib keys its own store loading
        # off sys.platform while our guard keys off os.name; patch both.
        with mock.patch.object(os, "name", "posix"), \
                mock.patch.object(sys, "platform", "linux"):
            expected = ssl.create_default_context().get_ca_certs()
            ctx = tls_mod.create_ssl_context()
        self.assertEqual(ctx.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(ctx.check_hostname)
        self.assertEqual(ctx.get_ca_certs(), expected)

    @unittest.skipUnless(ON_WINDOWS_STORE, "needs Windows certificate store")
    def test_windows_pem_is_well_formed_and_loadable(self):
        pem = tls_mod.windows_store_pem()
        self.assertTrue(pem and "-----BEGIN CERTIFICATE-----" in pem)
        ctx = ssl.create_default_context()
        ctx.load_verify_locations(cadata=pem)  # must not raise
        self.assertEqual(ctx.verify_mode, ssl.CERT_REQUIRED)

    @unittest.skipUnless(ON_WINDOWS_STORE, "needs Windows certificate store")
    def test_windows_context_carries_store_cas(self):
        self.assertGreater(len(tls_mod.create_ssl_context().get_ca_certs()), 0)


class TestEnvPrecedence(unittest.TestCase):
    def test_does_not_override_or_consume_env_ca_config(self):
        before = dict(os.environ)
        with mock.patch.object(tls_mod.ssl, "create_default_context",
                               wraps=ssl.create_default_context) as spy:
            tls_mod.create_ssl_context()
        spy.assert_called_once_with()  # no cafile/capath args: env fully honored
        self.assertEqual(dict(os.environ), before)  # env untouched, nothing stored

    def test_explicit_env_keys_survive_context_creation(self):
        os.environ["AGENT_LOOP_TLS_TEST_SENTINEL"] = "present"
        try:
            tls_mod.create_ssl_context()
            self.assertEqual(os.environ["AGENT_LOOP_TLS_TEST_SENTINEL"], "present")
        finally:
            del os.environ["AGENT_LOOP_TLS_TEST_SENTINEL"]


class TestClientWiring(unittest.TestCase):
    def test_clients_use_verified_context(self):
        from agent_loop import clients as clients_mod
        seen = {}

        class FakeResp:
            status = 200

            def __enter__(self): return self

            def __exit__(self, *a): return False

            def read(self): return b'{"data": []}'

        def fake_urlopen(req, timeout=None, context=None):
            seen["url"] = req.full_url
            seen["context"] = context
            return FakeResp()

        with mock.patch.object(clients_mod.urllib.request, "urlopen",
                               side_effect=fake_urlopen):
            c = clients_mod.OpenAICompatibleClient(
                "k", "https://example.invalid/v1", "m", retries=1)
            c.check_connectivity()
        self.assertTrue(seen["url"].startswith("https://"))
        self.assertIsInstance(seen["context"], ssl.SSLContext)
        self.assertEqual(seen["context"].verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(seen["context"].check_hostname)


if __name__ == "__main__":
    unittest.main()
