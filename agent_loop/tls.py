"""TLS trust configuration (stdlib only, verification ALWAYS on).

Root cause it fixes: on Windows, Python's OpenSSL often has no CA bundle
(default `cert.pem`/`certs` paths absent, no `SSL_CERT_FILE`/`SSL_CERT_DIR`
set), while the Windows certificate stores (used by curl, PowerShell,
SChannel) do trust the endpoint. Plain `ssl.create_default_context()` then
fails with `CERTIFICATE_VERIFY_FAILED` although the OS trusts the chain.

Fix: on Windows only, build the default verified context and ADDITIVELY load
the Windows ROOT + CA store certificates (DER -> PEM via stdlib `base64`).

Security properties (all covered by tests in `tests/test_tls.py`):
- Verification is never disabled: `CERT_REQUIRED` + `check_hostname=True`.
- No verification bypasses and no plain-HTTP downgrade anywhere in this
  package (AST-scanned by tests, docstrings excluded).
- Explicit `SSL_CERT_FILE` / `SSL_CERT_DIR` configuration keeps working:
  `create_default_context()` honors it first; Windows store certs are added
  on top, never substituted. Precedence: explicit env CA config, then
  platform defaults, plus the Windows store on Windows.
- The Windows certificate store is only read, never modified; no global OS
  security configuration is changed; no cert material or secret is logged.
- Non-Windows platforms: byte-for-byte the old behavior
  (`create_default_context()` untouched).
"""
from __future__ import annotations

import base64
import os
import ssl
import textwrap

_WINDOWS_STORES = ("ROOT", "CA")


def windows_store_pem() -> str | None:
    """PEM bundle of Windows trusted certs, or None when unavailable.

    Platform-guarded: returns None off Windows or when the stdlib Windows
    store API is absent. Never raises: store enumeration problems yield None
    so callers fall back to the default context (current behavior).
    """
    if os.name != "nt" or not hasattr(ssl, "enum_certificates"):
        return None
    blocks: list = []
    try:
        for store in _WINDOWS_STORES:
            try:
                certs = ssl.enum_certificates(store)
            except OSError:
                continue
            for der, encoding, _trust in certs:
                if encoding != "x509_asn" or not der:
                    continue
                b64 = base64.b64encode(bytes(der)).decode("ascii")
                blocks.append(
                    "-----BEGIN CERTIFICATE-----\n"
                    + "\n".join(textwrap.wrap(b64, 64))
                    + "\n-----END CERTIFICATE-----\n")
    except Exception:  # noqa: BLE001 - store problems must not break TLS setup
        return None
    return "".join(blocks) or None


def create_ssl_context() -> ssl.SSLContext:
    """Verified client context: defaults (+ env CA overrides) + Windows store.

    Always: `verify_mode == CERT_REQUIRED` and `check_hostname is True`.
    """
    ctx = ssl.create_default_context()
    pem = windows_store_pem()
    if pem:
        ctx.load_verify_locations(cadata=pem)
    return ctx
