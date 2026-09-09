"""Research registry: the laboratory's persistent memory (stdlib only).

Tracks research questions, hypotheses with falsification conditions, and
process-level failure patterns (self-improvement log). Persisted as JSON
inside the loop state dir so it survives crashes and resumes.

Question statuses: PROPOSED RESEARCHING DESIGNED IMPLEMENTING TESTING
EXPERIMENTING SUPPORTED PARTIALLY_SUPPORTED REFUTED INCONCLUSIVE COMPLETE
Hypothesis verdicts: UNTESTED SUPPORTED REFUTED INCONCLUSIVE
"""
from __future__ import annotations

import json
import time
from pathlib import Path

QUESTION_STATUSES = ("PROPOSED", "RESEARCHING", "DESIGNED", "IMPLEMENTING",
                     "TESTING", "EXPERIMENTING", "SUPPORTED",
                     "PARTIALLY_SUPPORTED", "REFUTED", "INCONCLUSIVE",
                     "COMPLETE")
HYPOTHESIS_VERDICTS = ("UNTESTED", "SUPPORTED", "REFUTED", "INCONCLUSIVE")


def _now() -> int:
    return int(time.time())


class ResearchRegistry:
    """Persistent registry of questions, hypotheses, and failure lessons."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data = self._load()

    # -- persistence ---------------------------------------------------
    def _blank(self) -> dict:
        return {"questions": {}, "hypotheses": {}, "failure_patterns": [],
                "ideas": [], "updated_at": _now()}

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return self._blank()
            for key in ("questions", "hypotheses"):
                if not isinstance(data.get(key), dict):
                    data[key] = {}
            for key in ("failure_patterns", "ideas"):
                if not isinstance(data.get(key), list):
                    data[key] = []
            return data
        except (OSError, ValueError):
            return self._blank()

    def save(self) -> str:
        self._data["updated_at"] = _now()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2)[:500000],
                       encoding="utf-8")
        tmp.replace(self.path)
        return str(self.path)

    # -- questions ------------------------------------------------------
    def add_question(self, qid: str, title: str, hypothesis: str = "",
                     status: str = "PROPOSED", **fields) -> dict:
        if status not in QUESTION_STATUSES:
            raise ValueError("bad question status %r" % status)
        entry = {"id": qid, "title": title, "hypothesis": hypothesis,
                 "status": status, "related_work": [],
                 "experiments": [], "baselines": [], "results": [],
                 "evidence": [], "confidence": "low", "limitations": [],
                 "next_questions": [], "history": [{"t": _now(), "status": status}],
                 "created_at": _now()}
        entry.update({k: v for k, v in fields.items()
                      if k in ("related_work", "experiments", "baselines",
                               "results", "evidence", "confidence",
                               "limitations", "next_questions")})
        self._data["questions"][qid] = entry
        self.save()
        return entry

    def set_question_status(self, qid: str, status: str, note: str = "") -> dict:
        if status not in QUESTION_STATUSES:
            raise ValueError("bad question status %r" % status)
        entry = self._data["questions"].get(qid)
        if entry is None:
            raise KeyError("unknown question %r" % qid)
        entry["status"] = status
        entry["history"].append({"t": _now(), "status": status,
                                "note": str(note)[:300]})
        entry["history"] = entry["history"][-40:]
        self.save()
        return entry

    def record_experiment(self, qid: str, experiment_id: str,
                          result_summary: str = "") -> dict:
        entry = self._data["questions"].get(qid)
        if entry is None:
            raise KeyError("unknown question %r" % qid)
        if experiment_id not in entry["experiments"]:
            entry["experiments"].append(experiment_id)
        if result_summary and result_summary not in entry["results"]:
            entry["results"].append(str(result_summary)[:500])
        self.save()
        return entry

    def get_question(self, qid: str) -> dict | None:
        return self._data["questions"].get(qid)

    def open_questions(self) -> list:
        terminal = ("SUPPORTED", "REFUTED", "COMPLETE")
        return [q for q in self._data["questions"].values()
                if q.get("status") not in terminal]

    # -- hypotheses ------------------------------------------------------
    def add_hypothesis(self, hid: str, statement: str, variables: dict,
                       falsification: str, question_id: str = "") -> dict:
        """variables: {variable, control, baseline, metric, expected_effect,
        limitations}. falsification: what observation would kill it."""
        required = ("variable", "control", "baseline", "metric")
        missing = [k for k in required if not variables.get(k)]
        if missing:
            raise ValueError("hypothesis missing %s" % ",".join(missing))
        if not str(falsification).strip():
            raise ValueError("hypothesis requires a falsification condition")
        entry = {"id": hid, "statement": statement,
                 "variables": dict(variables),
                 "falsification": falsification,
                 "verdict": "UNTESTED", "question_id": question_id,
                 "evidence": [], "created_at": _now()}
        self._data["hypotheses"][hid] = entry
        self.save()
        return entry

    def judge_hypothesis(self, hid: str, verdict: str, evidence: str = "") -> dict:
        if verdict not in HYPOTHESIS_VERDICTS:
            raise ValueError("bad verdict %r" % verdict)
        entry = self._data["hypotheses"].get(hid)
        if entry is None:
            raise KeyError("unknown hypothesis %r" % hid)
        entry["verdict"] = verdict
        if evidence:
            entry["evidence"].append({"t": _now(), "text": str(evidence)[:500]})
        self.save()
        return entry

    # -- failure patterns (self-improvement log) --------------------------
    def record_failure_pattern(self, pattern: str, occurrences: int = 1,
                               gate_suggestion: str = "") -> dict:
        for p in self._data["failure_patterns"]:
            if p["pattern"] == pattern:
                p["occurrences"] += occurrences
                if gate_suggestion and gate_suggestion not in p.get("gates", []):
                    p.setdefault("gates", []).append(gate_suggestion[:300])
                self.save()
                return p
        entry = {"pattern": str(pattern)[:300], "occurrences": occurrences,
                 "gates": [gate_suggestion[:300]] if gate_suggestion else [],
                 "first_seen": _now()}
        self._data["failure_patterns"].append(entry)
        self.save()
        return entry

    def failure_patterns(self, min_occurrences: int = 1) -> list:
        return [p for p in self._data["failure_patterns"]
                if p.get("occurrences", 0) >= min_occurrences]

    # -- ideas --------------------------------------------------------------
    def add_idea(self, title: str, checks: dict) -> dict:
        """Store a generated idea with its check outcomes."""
        entry = {"title": str(title)[:300], "checks": dict(checks),
                 "t": _now()}
        self._data["ideas"].append(entry)
        self._data["ideas"] = self._data["ideas"][-100:]
        self.save()
        return entry

    def idea_titles(self) -> list:
        return [i["title"] for i in self._data["ideas"]]
