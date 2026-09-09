"""Deterministic verification gates (no LLM calls).

These run BEFORE any LLM tester review, on every completed builder task:

1. claim_vs_artifact: worker claims must match on-disk reality (claimed
   artifact paths exist; claimed seed counts have matching result files;
   claimed changes coincide with actual git changes).
2. scope: edits must stay inside the task's allowed directories.
3. claim_strength: conclusions must not outrun evidence ("proves", ...).

StagnationTracker detects repeated tasks / repeated file churn / evidence
stagnation / manager oscillation and forces REPLAN after thresholds.

All functions are pure (inputs in, findings out) except the tracker, which
persists compact fingerprints in LoopState.
"""
from __future__ import annotations

import os
import re

# Directories the tester itself may always ignore (loop machinery, VCS).
_ALWAYS_IGNORED_TOP = (".agent_loop", ".git", "__pycache__", ".venv",
                       "node_modules")

# Default allowed scope when the manager decision carries no explicit list
# (backward compatibility with pre-v2 decisions).
DEFAULT_ALLOWED_DIRS = ("experiments/", "evaluation/", "fabric/", "docs/",
                        "tests/", "backend/", "scripts/")

# Language that claims more than evidence can support.
_OVERCLAIM_PATTERNS = (
    r"\bproves?\b", r"\bproven\b", r"\bconclusively?\b", r"\bdefinitively\b",
    r"\bbreakthrough\b", r"\bstate-of-the-art\b", r"\bSOTA\b",
    r"\bsignificant (improvement|gain|speedup|better)\b",
)

_SEED_CLAIM = re.compile(r"(\d+)\s+seeds?\b", re.IGNORECASE)


def _top_dir(path: str) -> str:
    return (path or "").replace("\\", "/").split("/")[0] + "/"


def porcelain_path(line: str) -> str:
    """Strip `git status --porcelain` XY prefixes (evidence uses raw lines).

    The XY field is positional (leading spaces are significant), so the
    prefix match runs BEFORE any whitespace stripping.
    """
    norm = str(line or "").replace("\\", "/")
    m = re.match(r"^(?:[ MADRCU?!]{2}|!!)[ \t]+(.*)$", norm)
    if m:
        norm = m.group(1).strip().strip('"')
    norm = re.sub(r"^(?:\./|/)+", "", norm.strip())
    return norm


def check_claims_vs_artifacts(report: dict, repo_root: str,
                              changed_files: list) -> list:
    """Verify worker claims against on-disk reality. Returns findings."""
    findings = []
    status = str(report.get("status", ""))
    if status in ("failed", "blocked", "partial"):
        return findings  # nothing completed -> nothing to falsify here
    changed = " ".join(changed_files or [])

    for art in report.get("artifacts", []) or []:
        name = str(art).split(" ")[0].strip().strip(",;")
        if not name or name.startswith("("):
            continue
        full = os.path.join(repo_root, name.replace("/", os.sep))
        if not os.path.exists(full):
            findings.append("claimed artifact missing on disk: %s" % name[:120])

    claimed_changes = [c for c in (report.get("changes", []) or []) if str(c).strip()]
    if claimed_changes and not (changed_files or []):
        findings.append("worker claims file changes but git shows none")

    blob = " ".join(str(report.get(k, "")) for k in
                    ("tests", "evidence", "commands", "recommendation"))
    for m in _SEED_CLAIM.finditer(blob):
        need = int(m.group(1))
        artifacts = report.get("artifacts", []) or []
        seed_hits = sum(1 for a in artifacts
                        if re.search(r"seed|run\d|s\d+\b", str(a), re.IGNORECASE))
        if seed_hits == 0 and not artifacts:
            findings.append("claims %d seeds but lists no result artifacts"
                            % need)
        elif 0 < seed_hits < need:
            findings.append("claims %d seeds but only %d seed-like artifacts"
                            % (need, seed_hits))
    return findings


def check_scope(decision: dict, changed_files: list) -> list:
    """Reject edits outside the task's allowed directories."""
    explicit = bool(decision.get("allowed_dirs"))
    allowed = list(decision.get("allowed_dirs") or list(DEFAULT_ALLOWED_DIRS))
    allowed_norm = [a.replace("\\", "/").rstrip("/") + "/" for a in allowed]
    findings = []
    for f in changed_files or []:
        norm = porcelain_path(f)
        if not norm or norm in ("", "/"):
            continue
        if norm.startswith("fatal:") or norm.startswith("("):
            continue  # git-error/placeholder lines are not paths
        if _top_dir(norm) in [d.split("/")[0] + "/" for d in
                              _ALWAYS_IGNORED_TOP]:
            continue
        if "/" not in norm and not explicit:
            continue  # root-level project files (README, configs) are
            # allowed under the default scope; explicit scopes govern all
        if not any(norm == d.rstrip("/") or norm.startswith(d)
                   for d in allowed_norm):
            findings.append("scope violation: %s outside %s"
                            % (norm[:120], ",".join(allowed)[:200]))
    return findings


def check_claim_strength(report: dict) -> list:
    """Reject conclusions stronger than the evidence."""
    findings = []
    blob = " ".join(str(report.get(k, "")) for k in
                    ("evidence", "recommendation", "tests"))
    for pat in _OVERCLAIM_PATTERNS:
        m = re.search(pat, blob, re.IGNORECASE)
        if m:
            findings.append("overclaim language %r requires measured proof "
                            "(artifacts + controls)" % m.group(0)[:60])
    return findings


def changed_top_dirs(changed_files: list) -> list:
    out = []
    for f in changed_files or []:
        norm = porcelain_path(f)
        top = _top_dir(norm)
        if top not in ("", "/") and top not in out:
            out.append(top)
    return out


def changed_delta(pre_lines: list, post_lines: list) -> list:
    """Per-iteration change delta: post entries absent from the pre-image.

    The scope gate must judge what THIS work unit changed, not pre-existing
    untracked scaffolding. Comparison uses normalized paths; original post
    lines are returned for evidence fidelity.
    """
    pre = {porcelain_path(l) for l in pre_lines or []}
    return [l for l in post_lines or [] if porcelain_path(l) not in pre]


class StagnationTracker:
    """Frontier tracking with REPLAN signals (state-persisted fingerprints)."""

    def __init__(self, state, repeat: int = 3, window: int = 6):
        self.state = state
        self.repeat = max(2, repeat)
        self.window = max(repeat, window)

    def _hist(self) -> list:
        st = self.state._state.setdefault("stagnation", {"observations": []})
        return st["observations"]

    def _save(self, obs: list):
        self.state._state["stagnation"] = {"observations": obs[-self.window:]}
        self.state._save()

    @staticmethod
    def _sig(task: str) -> str:
        return " ".join(str(task).lower().split())[:200]

    def observe(self, task: str, changed_files: list, evidence_summary: str,
                worker_status: str) -> tuple:
        """Record one iteration; return (ok: bool, reasons: list)."""
        obs = self._hist()
        entry = {"task": self._sig(task),
                 "files": sorted(changed_top_dirs(changed_files)),
                 "evidence": " ".join(str(evidence_summary).split())[:200],
                 "status": str(worker_status)}
        obs.append(entry)
        self._save(obs)
        tail = obs[-self.window:]
        reasons = []
        same_task = sum(1 for o in tail if o["task"] == entry["task"])
        if same_task >= self.repeat:
            reasons.append("task repeated %dx in last %d iterations"
                           % (same_task, len(tail)))
        if entry["status"] in ("failed", "blocked") and len(tail) >= 2:
            same_files = sum(1 for o in tail
                             if o["files"] == entry["files"] and o["status"]
                             in ("failed", "blocked"))
            if same_files >= self.repeat:
                reasons.append("same files churned without progress %dx"
                               % same_files)
        if len(tail) >= self.repeat:
            same_ev = sum(1 for o in tail if o["evidence"] == entry["evidence"])
            if same_ev >= self.repeat:
                reasons.append("no new evidence for %d iterations" % same_ev)
        return (not reasons, reasons)
