"""Independent tester layer (manager -> builder -> tester -> manager).

The tester is adversarial: it assumes the builder may be wrong and attempts
to falsify every material claim. It NEVER modifies repository files.

Backends (provider-agnostic interface, configured via TESTER_BACKEND):
- "opencode": headless `opencode run --format json --dir <repo>` subprocess
  (proven live). Read-only instructions; permission denials surface as
  INCONCLUSIVE, never PASS. No --auto flag is ever passed.
- "api": OpenAI-compatible chat client on the manager-fallback provider,
  review-only reuse (never executes, never decides).
- "off": deterministic gates only; LLM verdict recorded as INCONCLUSIVE.

run_verification() runs deterministic gates first (verify.py), then the LLM
tester unless hard fabrication findings already settle FAIL. Combination:
any FAIL -> FAIL; gates pass + LLM INCONCLUSIVE/unavailable -> INCONCLUSIVE;
otherwise PASS. Every verdict carries concrete evidence.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time

try:
    from .clients import ApiError, OpenAICompatibleClient, TransientApiError
    from .schemas import extract_json
except ImportError:  # pragma: no cover - direct script use
    from clients import ApiError, OpenAICompatibleClient, TransientApiError
    from schemas import extract_json

VERDICTS = ("PASS", "FAIL", "INCONCLUSIVE")

TESTER_SYSTEM = """\
You are the TESTER, an independent adversarial reviewer of autonomous
engineering work. Assume the builder may be wrong. Attempt to falsify every
material claim. NEVER modify the repository - you have no write tools and
must not run mutating commands; inspect only (read files, review diffs and
artifacts).

Evidence outranks prose. A builder statement is not evidence.

Inspect: the git diff, changed files, experiment artifacts, JSON outputs,
test outputs, and whether claimed commands could have produced the observed
state. Check: do claimed files exist with plausible contents? Do numbers in
reports match numbers in artifacts? Were tests actually executed (exit codes,
not descriptions)? Are conclusions ("X proves Y") supported by controls,
baselines, and seeds - or stronger than the evidence?

Return EXACTLY ONE JSON object, no prose outside it:
{"verdict": "PASS|FAIL|INCONCLUSIVE", "evidence": [str], "notes": str}
- PASS: every material claim is supported by concrete evidence you inspected.
- FAIL: at least one material claim is contradicted or unsupported (say which,
  with the specific mismatch).
- INCONCLUSIVE: evidence is insufficient to decide (say precisely what is
  missing). Never guess.
"""

MAX_DIFF_CHARS = 12000


def build_tester_messages(task: str, expected: str, report: dict,
                          diff_text: str) -> str:
    """Assemble the single verification prompt (returned as one string)."""
    rep = dict(report or {})
    rep.pop("_executor", None)
    body = (
        "ORIGINAL BUILDER TASK:\n" + str(task or "(none)")
        + "\n\nEXPECTED RESULT:\n" + str(expected or "(none)")
        + "\n\nBUILDER REPORT (claims to verify):\n"
        + json.dumps(rep, indent=2)[:8000]
        + "\n\nGIT DIFF (objective, truncated):\n"
        + (str(diff_text or "(no diff)")[:MAX_DIFF_CHARS])
        + "\n\nVerify every material claim against the evidence above. "
           "Return exactly one verdict JSON object."
    )
    return body


def parse_verdict(text: str) -> dict:
    """Parse an LLM tester verdict; strict on schema, lenient on wrapping."""
    obj = extract_json(text)
    verdict = str(obj.get("verdict", "")).strip().upper()
    if verdict not in VERDICTS:
        raise ValueError("tester verdict must be one of %s, got %r"
                         % (list(VERDICTS), obj.get("verdict")))
    ev = obj.get("evidence", [])
    if isinstance(ev, str):
        ev = [ev]
    return {"verdict": verdict,
            "evidence": [str(x)[:500] for x in (ev or [])][:10],
            "notes": str(obj.get("notes", ""))[:1000]}


class OpenCodeTester:
    """Headless `opencode run` verification backend (read-only)."""

    def __init__(self, repo_root: str = ".", command: str = "opencode",
                 model: str = "", timeout_s: float = 300.0):
        self.repo_root = repo_root
        self.command = command
        self.model = model
        self.timeout_s = timeout_s
        self.provider = "opencode"
        self.backend = "opencode"

    def check(self) -> dict:
        exe = shutil.which(self.command)
        if not exe:
            return {"ok": False, "reason": "opencode not on PATH"}
        return {"ok": True, "reason": "opencode available", "exe": exe}

    def verify(self, prompt: str) -> str:
        """Run one headless verification; return raw text for parsing."""
        if not shutil.which(self.command):
            raise ApiError("opencode executable not found")
        args = [self.command, "run", "--format", "json",
                "--dir", self.repo_root]
        if self.model:
            args += ["--model", self.model]
        args.append("Read-only verification task. Do not modify any files, "
                    "do not run mutating commands, do not approve anything. "
                    + prompt)
        try:
            proc = subprocess.run(args, cwd=self.repo_root,
                                  capture_output=True, text=True,
                                  timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            raise TransientApiError(
                "opencode tester timed out after %.0fs" % self.timeout_s)
        except OSError as e:
            raise ApiError("cannot execute opencode: %s" % e)
        text = _ndjson_text(proc.stdout or "")
        if text.strip():
            return text
        tail = ((proc.stdout or "") + (proc.stderr or ""))[-1000:]
        raise ApiError("opencode tester produced no text (exit %d): %s"
                       % (proc.returncode, tail.strip()[:500]))


def _ndjson_text(blob: str) -> str:
    """Collect text parts from `opencode run --format json` NDJSON events."""
    parts = []
    for line in (blob or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        if ev.get("type") == "text":
            part = ev.get("part", {}) if isinstance(ev.get("part"), dict) else {}
            t = part.get("text", "")
            if isinstance(t, str) and t:
                parts.append(t)
        elif ev.get("type") == "message":
            msg = ev.get("message", {}) if isinstance(ev.get("message"), dict) else {}
            for key in ("text", "content"):
                t = msg.get(key, "")
                if isinstance(t, str) and t:
                    parts.append(t)
    return "\n".join(parts)


class ApiTester:
    """API-backed adversarial reviewer (manager-fallback provider)."""

    def __init__(self, api_key: str, base_url: str, model: str,
                 timeout_s: float = 120.0, retries: int = 2):
        self.client = OpenAICompatibleClient(api_key, base_url, model,
                                             timeout_s=timeout_s,
                                             retries=retries)
        self.provider = "api-tester"
        self.backend = "api"
        self.model = model

    def check(self) -> dict:
        return self.client.check_connectivity()

    def verify(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": TESTER_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        msg = self.client.chat(messages, force_json=True)
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(b.get("text", "") for b in content
                              if isinstance(b, dict))
        return str(content or "")


def combine_verdict(gate_findings: list, llm: dict | None) -> dict:
    """Merge deterministic gates with the LLM verdict.

    Hard fabrication findings (missing artifacts/changes) settle FAIL even
    without an LLM pass. Any FAIL wins ties; INCONCLUSIVE propagates doubt;
    PASS requires gates clean AND an explicit LLM PASS.
    """
    hard = [f for f in gate_findings
            if f.startswith("claimed artifact missing")
            or f.startswith("worker claims file changes")]
    if hard:
        return {"verdict": "FAIL", "gate_findings": gate_findings,
                "llm": llm, "skipped_llm": True,
                "notes": "deterministic fabrication check failed"}
    if any(f.startswith("scope violation") or f.startswith("overclaim")
           for f in gate_findings):
        base_fail = True
    else:
        base_fail = False
    if llm is None:
        return {"verdict": ("FAIL" if base_fail else "INCONCLUSIVE"),
                "gate_findings": gate_findings, "llm": None,
                "notes": "no LLM tester verdict available"}
    if llm.get("verdict") == "FAIL" or base_fail:
        return {"verdict": "FAIL", "gate_findings": gate_findings,
                "llm": llm, "notes": llm.get("notes", "")}
    if llm.get("verdict") == "INCONCLUSIVE":
        return {"verdict": "INCONCLUSIVE", "gate_findings": gate_findings,
                "llm": llm, "notes": llm.get("notes", "")}
    return {"verdict": "PASS", "gate_findings": gate_findings,
            "llm": llm, "notes": llm.get("notes", "")}
