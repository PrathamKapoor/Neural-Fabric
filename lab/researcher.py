"""Researcher: novelty attack against local + documented prior art (stdlib).

Local-first: searches the repository (implementations) and docs/research/
for related work. An optional web fetcher (injectable, offline by default)
extends the attack to external literature. Never invents literature: every
cited item must come from an observed source (file hit or fetcher result).

Real source-backed path: arxiv_fetcher() queries the arXiv API over HTTPS,
parses Atom with stdlib ElementTree, caches raw responses with retrieval
dates, and returns [{source, title, published, summary, retrieved_at}].
Failures yield [] (never synthesized entries).

Verdicts: ESTABLISHED (already exists - pivot required), NARROW_GAP (broad
idea exists; a scoped mechanism/comparison/evaluation is open),
OPEN (no prior art found in searched sources).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
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


ARXIV_API = "https://export.arxiv.org/api/query"
CROSSREF_API = "https://api.crossref.org/works"
CACHE_MAX_AGE_S = 7 * 24 * 3600


def _ssl_context():
    try:
        from agent_loop.tls import create_ssl_context
        return create_ssl_context()
    except Exception:
        import ssl as _ssl
        return _ssl.create_default_context()


def _cache_path(cache_dir: str | Path, query: str) -> Path:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / ("arxiv_%s.json" % digest)


def fetch_arxiv(query: str, max_results: int = 5,
                cache_dir: str | Path | None = None,
                timeout_s: float = 30.0,
                max_age_s: float = CACHE_MAX_AGE_S) -> list:
    """Query arXiv, cached. Returns [{source, title, published, summary,
    retrieved_at}]. Any failure -> [] (never synthesized)."""
    if cache_dir is not None:
        path = _cache_path(cache_dir, query)
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if (isinstance(cached, dict)
                    and time.time() - cached.get("retrieved_at", 0) < max_age_s
                    and isinstance(cached.get("items"), list)):
                return cached["items"]
        except (OSError, ValueError):
            pass
    params = urllib.parse.urlencode(
        {"search_query": "all:%s" % query, "start": 0,
         "max_results": max(1, min(int(max_results), 20)),
         "sortBy": "relevance", "sortOrder": "descending"})
    req = urllib.request.Request(ARXIV_API + "?" + params,
                                 headers={"User-Agent": "nn-lab-researcher/1.0"},
                                 method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s,
                                    context=_ssl_context()) as resp:
            blob = resp.read().decode("utf-8", "replace")
    except Exception:
        return []
    items = []
    try:
        root = ET.fromstring(blob)
    except ET.ParseError:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    now = int(time.time())
    for entry in root.findall("a:entry", ns)[:max_results]:
        def _text(tag):
            el = entry.find("a:" + tag, ns)
            return " ".join((el.text or "").split()) if el is not None else ""
        url = _text("id")
        if not url:
            continue
        items.append({"source": url, "title": _text("title"),
                      "published": _text("published")[:10],
                      "summary": _text("summary")[:1500],
                      "retrieved_at": now})
    if cache_dir is not None and items:
        try:
            _cache_path(cache_dir, query).write_text(
                json.dumps({"query": query, "retrieved_at": now,
                            "items": items})[:200000], encoding="utf-8")
        except OSError:
            pass
    return items


def arxiv_novelty_fetcher(query: str, cache_dir: str | Path | None = None,
                          max_results: int = 5):
    """Adapter: arxiv fetch results as attack_novelty web hits."""
    def _fetch(keywords):
        q = query or " ".join(keywords[:4])
        return [{"source": it["source"],
                 "claim": "%s — %s" % (it["title"], it["summary"][:300])}
                for it in fetch_arxiv(q, max_results=max_results,
                                      cache_dir=cache_dir)]
    return _fetch


def fetch_crossref(query: str, max_results: int = 5,
                   cache_dir: str | Path | None = None,
                   timeout_s: float = 30.0,
                   max_age_s: float = CACHE_MAX_AGE_S) -> list:
    """Query Crossref (TLS-verifiable host), cached. Same item shape.

    Returns [{source, title, published, summary, retrieved_at}]; source is
    the DOI URL, summary the abstract (may be empty - recorded, not faked).
    Any failure -> [].
    """
    if cache_dir is not None:
        path = _cache_path(cache_dir, "crossref:" + query)
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if (isinstance(cached, dict)
                    and time.time() - cached.get("retrieved_at", 0) < max_age_s
                    and isinstance(cached.get("items"), list)):
                return cached["items"]
        except (OSError, ValueError):
            pass
    params = urllib.parse.urlencode(
        {"query": query, "rows": max(1, min(int(max_results), 20)),
         "select": "DOI,title,abstract,published,URL",
         "mailto": "nn-lab@localhost"})
    req = urllib.request.Request(CROSSREF_API + "?" + params,
                                 headers={"User-Agent": "nn-lab-researcher/1.0",
                                          "Accept": "application/json"},
                                 method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s,
                                    context=_ssl_context()) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:
        return []
    items = []
    now = int(time.time())
    try:
        records = payload["message"]["items"]
    except (KeyError, TypeError):
        return []
    for rec in records[:max_results]:
        if not isinstance(rec, dict):
            continue
        doi = str(rec.get("DOI", ""))
        if not doi:
            continue
        title = " ".join(rec.get("title", []) or [])
        abstract = re.sub(r"<[^>]+>", "",
                          str(rec.get("abstract", "") or ""))[:1500]
        pub = ""
        try:
            pub = "-".join(str(p) for p in
                           rec.get("published", {}).get("date-parts", [[]])[0][:3])
        except (IndexError, AttributeError):
            pass
        items.append({"source": "https://doi.org/" + doi, "title": title,
                      "published": pub, "summary": abstract,
                      "retrieved_at": now})
    if cache_dir is not None and items:
        try:
            _cache_path(cache_dir, "crossref:" + query).write_text(
                json.dumps({"query": query, "retrieved_at": now,
                            "items": items})[:200000], encoding="utf-8")
        except OSError:
            pass
    return items


def crossref_novelty_fetcher(query: str, cache_dir: str | Path | None = None,
                             max_results: int = 5):
    """Adapter: crossref fetch results as attack_novelty web hits."""
    def _fetch(keywords):
        q = query or " ".join(keywords[:4])
        return [{"source": it["source"],
                 "claim": "%s — %s" % (it["title"], it["summary"][:300])}
                for it in fetch_crossref(q, max_results=max_results,
                                         cache_dir=cache_dir)]
    return _fetch
