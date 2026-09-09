"""Researcher: novelty attack against local + documented prior art (stdlib).

Local-first: searches the repository (implementations) and docs/research/
for related work. An optional web fetcher (injectable, offline by default)
extends the attack to external literature. Never invents literature: every
cited item must come from an observed source (file hit or fetcher result).

Verdicts: ESTABLISHED (already exists - pivot required), NARROW_GAP (broad
idea exists; a scoped mechanism/comparison/evaluation is open),
OPEN (no prior art found in searched sources).
"""
from __future__ import annotations

import os
import re
from pathlib import Path


def search_code(repo_root: str | Path, keywords: list,
                include=(".py",), exclude_dirs=(".git", ".venv", "__pycache__",
                                                ".agent_loop", "node_modules")) -> list:
    """Return [{file, line, snippet}] hits for any keyword (case-insensitive)."""
    hits = []
    root = Path(repo_root)
    pats = [re.compile(re.escape(k), re.IGNORECASE) for k in keywords]
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for fn in filenames:
            if not any(fn.endswith(ext) for ext in include):
                continue
            full = Path(dirpath) / fn
            try:
                text = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if any(p.search(line) for p in pats):
                    hits.append({"file": str(full.relative_to(root)),
                                 "line": i, "snippet": line.strip()[:200]})
                    if len(hits) >= 50:
                        return hits
    return hits


def search_docs(repo_root: str | Path, keywords: list) -> list:
    """Search markdown docs (research notes, reports, READMEs)."""
    return search_code(repo_root, keywords, include=(".md",))


def attack_novelty(idea: str, keywords: list, repo_root: str | Path = ".",
                   related_notes: list | None = None,
                   web_fetcher=None) -> dict:
    """Attempt to destroy the novelty claim. Returns the attack record.

    related_notes: [{source, claim}] from curated docs (e.g. related_work.md).
    web_fetcher: optional callable(keywords) -> [{source, claim}] for
    external literature. Offline default (None) limits scope honestly.
    """
    code_hits = search_code(repo_root, keywords)
    doc_hits = search_docs(repo_root, keywords)
    notes = list(related_notes or [])
    web_hits = list(web_fetcher(keywords)) if web_fetcher else []
    total_prior = len(code_hits) + len(doc_hits) + len(notes) + len(web_hits)
    if total_prior == 0:
        verdict = "OPEN"
        guidance = ("No prior art found in searched sources. Scope is "
                    "limited to this repo + provided notes; do not claim "
                    "global novelty without a web literature pass.")
    elif any("novel" in str(n.get("claim", "")).lower() and "not" not in
             str(n.get("claim", "")).lower() for n in notes) and not code_hits:
        verdict = "NARROW_GAP"
        guidance = ("Related work exists but no local implementation found; "
                    "frame the contribution as the missing comparison, "
                    "evaluation, or mechanism.")
    elif code_hits and doc_hits:
        verdict = "ESTABLISHED"
        guidance = ("Implemented AND documented locally: do not claim novelty. "
                    "Ask what narrower question remains (different regime, "
                    "stronger baseline, falsifying comparison).")
    else:
        verdict = "NARROW_GAP"
        guidance = ("Partial prior art. Scope the claim to the specific "
                    "mechanism/comparison/evaluation that is missing.")
    return {"idea": idea, "verdict": verdict, "guidance": guidance,
            "code_hits": code_hits[:10], "doc_hits": doc_hits[:10],
            "notes": notes[:10], "web_hits": web_hits[:10],
            "web_searched": web_fetcher is not None}
