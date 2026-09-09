"""Minimal stdlib-only `.env` loader.

Fills `os.environ` from a `<repo>/.env` file WITHOUT ever overwriting an
already-exported process variable (explicit environment always wins over
`.env`, which wins over compiled defaults).

Security rules (load-bearing):
- Only key NAMES are ever reported; values are never printed, logged,
  returned in diagnostics, or written anywhere.
- The loader only mutates `os.environ` for missing keys; it never writes files.
- Malformed lines are skipped silently (never echoed back).
"""
from __future__ import annotations

import os
import re

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_dotenv_text(text: str) -> dict:
    """Parse KEY=VALUE lines. Blank lines and `#` comments ignored.

    - Splits on the FIRST `=` only (values may contain `=`).
    - Strips surrounding whitespace; removes one layer of matching single or
      double quotes around the value.
    - Tolerates a leading `export ` prefix.
    - Anything without a valid KEY= shape is skipped.
    """
    values: dict = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not _KEY_RE.match(key):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def load_dotenv(path) -> list:
    """Load missing keys from `path` into `os.environ`.

    Returns the NAMES of keys actually set (never values). Missing/unreadable
    files are a silent no-op returning [].
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return []
    loaded: list = []
    for key, value in parse_dotenv_text(text).items():
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def dotenv_info(path) -> tuple:
    """Non-mutating probe: (exists: bool, parseable_key_count: int).

    Reports nothing about values.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return False, 0
    return True, len(parse_dotenv_text(text))
