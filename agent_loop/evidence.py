"""Objective repository evidence collection.

After every worker pass the orchestrator snapshots what ACTUALLY happened -
git status, diff stat, changed files, experiment artifacts, and the worker's
own command/test claims - so the manager reviews facts, not prose.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

MAX_DIFF_CHARS = 12000


def _git(repo_root: Path, *args: str, timeout: int = 30) -> str:
    try:
        proc = subprocess.run(["git", *args], cwd=str(repo_root),
                              capture_output=True, text=True, timeout=timeout)
        return ((proc.stdout or "") + (proc.stderr or "")).strip()
    except Exception as e:  # noqa: BLE001 - evidence must never crash the loop
        return "(unavailable: %s)" % e


def porcelain_lines(repo_root: str | Path) -> list:
    """Raw `git status --porcelain` lines (pre-image for delta computation)."""
    root = Path(repo_root).resolve()
    return [l for l in _git(root, "status", "--porcelain").splitlines() if l.strip()]


def collect_evidence(repo_root: str | Path) -> dict:
    """Return a JSON-serializable evidence snapshot for the manager."""
    root = Path(repo_root).resolve()
    status = _git(root, "status", "--short", "--branch")
    diff_stat = _git(root, "diff", "--stat")
    diff = _git(root, "diff", "--", ".")
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n...[diff truncated]..."
    changed = [l for l in _git(root, "status", "--porcelain").splitlines()]
    artifacts = []
    for sub in ("experiments/results", "experiments/runs"):
        d = root / sub
        if d.is_dir():
            for f in sorted(d.glob("*.json")):
                try:
                    artifacts.append("%s (%d bytes, mtime %d)"
                                     % (f.relative_to(root), f.stat().st_size,
                                        int(f.stat().st_mtime)))
                except OSError:
                    pass
    head = _git(root, "log", "--oneline", "-3")
    return {
        "git_status": status or "(clean)",
        "git_diff_stat": diff_stat or "(no diff)",
        "git_diff": diff or "(no diff)",
        "changed_files": changed,
        "artifacts": artifacts,
        "recent_commits": head or "(no history)",
    }


def summarize_evidence(evidence: dict) -> str:
    """One-screen human summary for the terminal UI."""
    lines = [
        "git: %s" % _one_line(evidence.get("git_status", "")),
        "diff: %s" % _one_line(evidence.get("git_diff_stat", "")),
        "changed: %d file(s)" % len(evidence.get("changed_files", [])),
        "artifacts: %d" % len(evidence.get("artifacts", [])),
    ]
    return " | ".join(lines)


def _one_line(s: str, limit: int = 160) -> str:
    s = " ".join(str(s).split())
    return (s[:limit] + "...") if len(s) > limit else (s or "-")
