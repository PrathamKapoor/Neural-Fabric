"""Analyst: evidence classification + doc↔artifact reconciliation (stdlib).

claim_kind(): MEASURED / DERIVED / INFERRED / SPECULATIVE heuristics.
reconcile_numbers(): extract (number, context) pairs from markdown docs and
check each against canonical artifact values (caller-supplied mapping of
name -> value). Mismatches are findings, never silent.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

MEASURED_HINTS = ("measured", "recorded", "observed", "executed", "ran ",
                  "reproduced", "artifact")
DERIVED_HINTS = ("ratio", "computed from", "derived", "calculated",
                 "difference", "reduction", "% less", "speedup")
INFERRED_HINTS = ("suggests", "indicates", "appears", "likely", "consistent with",
                  "interpretation", "hypothesis")
SPECULATIVE_HINTS = ("might", "could generalize", "future", "would",
                     "should transfer", "promise", "believe", "expect that")


def claim_kind(text: str) -> str:
    """Classify a conclusion sentence. Never collapse categories silently."""
    low = str(text or "").lower()
    scores = {
        "MEASURED": sum(1 for h in MEASURED_HINTS if h in low),
        "DERIVED": sum(1 for h in DERIVED_HINTS if h in low),
        "INFERRED": sum(1 for h in INFERRED_HINTS if h in low),
        "SPECULATIVE": sum(1 for h in SPECULATIVE_HINTS if h in low),
    }
    best = max(scores, key=lambda k: (scores[k], k))
    return best if scores[best] > 0 else "INFERRED"


_NUMBER = re.compile(r"(?<!\w)(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")


def _parse_number(token: str, frac: str | None) -> float | None:
    try:
        whole = float(token.replace(",", ""))
        if frac is not None:
            whole += float("0." + frac)
        return whole
    except ValueError:
        return None


def extract_numbers(markdown: str) -> list:
    """Return [(value, is_int, context)] for doc numbers (excl. years/refs)."""
    out = []
    for m in _NUMBER.finditer(str(markdown or "")):
        val = _parse_number(m.group(1), m.group(2))
        if val is None:
            continue
        if 1900 <= val <= 2100 and m.group(2) is None:
            continue  # years, versions, ports
        start = max(0, m.start() - 60)
        out.append((val, m.group(2) is None,
                    markdown[start:m.end() + 40].replace("\n", " ")))
    return out


def _close_enough(doc_val: float, doc_is_int: bool, canon: float) -> bool:
    # Integers (FLOPs, counts, seeds) must match EXACTLY - this catches
    # stale-figure bugs like 9966 vs the real 10004.8. Decimals (accuracies,
    # percentages as written) get 1% relative tolerance for prose rounding.
    if doc_is_int:
        return doc_val == canon
    denom = max(abs(doc_val), abs(canon), 1e-9)
    return abs(doc_val - canon) / denom <= 0.01


def reconcile_numbers(doc_paths: list, canonical: dict,
                      repo_root: str | Path = ".") -> list:
    """Check doc numbers against canonical {label: value} artifact values.

    A doc number matches if it is close to ANY canonical value (2% or exact);
    otherwise it is reported with context. Returns findings (empty = clean).
    canonical: {label: numeric_value} built by the caller from artifacts.
    """
    findings = []
    canon_vals = list(canonical.values())
    for rel in doc_paths:
        try:
            text = (Path(repo_root) / rel).read_text(encoding="utf-8")
        except OSError:
            findings.append("cannot read doc for reconciliation: %s" % rel)
            continue
        for val, is_int, ctx in extract_numbers(text):
            if not any(isinstance(c, (int, float)) and _close_enough(val, is_int, c)
                       for c in canon_vals):
                # Skip small integers (counts/years-like) to limit noise.
                if is_int and val < 100:
                    continue
                findings.append("doc number without artifact match: %s [%s: %s]"
                                % (val, rel, ctx.strip()[:120]))
    return findings


def load_canonical_from_artifacts(specs: list,
                                  repo_root: str | Path = ".") -> dict:
    """Build {label: value} from [(label, artifact_relpath, json_path)] specs.

    json_path is a dotted path into the artifact (e.g. "mean_mi_nats" or
    "cells.0.mean_accuracy"; list indices supported).
    """
    out = {}
    for label, rel, jpath in specs:
        try:
            obj = json.loads((Path(repo_root) / rel).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        cur = obj
        try:
            for part in str(jpath).split("."):
                cur = cur[int(part)] if isinstance(cur, list) else cur[part]
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if isinstance(cur, (int, float)) and not isinstance(cur, bool):
            out[label] = cur
    return out


def _artifact_numbers(path: Path) -> list:
    """All numeric leaves of a JSON artifact (for iteration self-audit)."""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []

    def _walk(o):
        if isinstance(o, dict):
            for v in o.values():
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            out.append(o)

    _walk(obj)
    return out


def reconcile_iteration(changed_md: list, evidence: dict,
                        repo_root: str | Path = ".") -> list:
    """Self-audit: numbers in docs changed THIS iteration vs its evidence.

    Canonical values = numeric leaves of artifacts listed in the evidence
    plus numbers in the evidence summary. Findings for unmatched doc numbers
    (int-exact / float-1% rule). Returns findings (empty = clean).
    """
    root = Path(repo_root)
    canon: list = []
    for a in (evidence.get("artifacts", []) if isinstance(evidence, dict) else []):
        name = str(a).split(" ")[0].strip().strip(",;")
        if name.endswith(".json"):
            canon += _artifact_numbers(root / name)
    for val, _, _ in extract_numbers(
            str((evidence or {}).get("evidence_summary", ""))):
        canon.append(val)
    findings = []
    for rel in changed_md:
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except OSError:
            findings.append("cannot read changed doc: %s" % rel)
            continue
        for val, is_int, ctx in extract_numbers(text):
            if is_int and val < 100:
                continue
            if not any(_close_enough(val, is_int, c) for c in canon):
                findings.append("doc number without iteration evidence: %s [%s]"
                                % (val, ctx.strip()[:120]))
    return findings
