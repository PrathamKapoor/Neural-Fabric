"""Structured-output parsing / validation for manager and worker messages.

Models are asked for JSON but often wrap it in fences or add prose, so parsing
is lenient on extraction yet strict on the resulting schema (with safe
defaults), so the orchestrator always gets a machine-readable decision.
"""
from __future__ import annotations

import json
import re

MANAGER_STATUSES = ("continue", "complete", "retry", "blocked")
WORKER_STATUSES = ("done", "partial", "blocked", "failed")


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of possibly-decorated model output."""
    if not isinstance(text, str):
        raise ValueError("model output is not text")
    t = text.strip()
    # Strip ```json ... ``` / ``` ... ``` fences (take the first fenced block).
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
    if fence:
        t = fence.group(1).strip()
    else:
        start, end = t.find("{"), t.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON object found in model output")
        t = t[start:end + 1]
    try:
        obj = json.loads(t)
    except json.JSONDecodeError as e:
        raise ValueError("model output is not valid JSON: %s" % e)
    if not isinstance(obj, dict):
        raise ValueError("model output JSON must be an object")
    return obj


def _str_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v else []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def parse_manager_output(text: str) -> dict:
    obj = extract_json(text)
    status = str(obj.get("status", "")).strip().lower()
    if status not in MANAGER_STATUSES:
        raise ValueError("manager status must be one of %s, got %r"
                         % (list(MANAGER_STATUSES), obj.get("status")))
    if status in ("continue", "retry") and not str(obj.get("next_task", "")).strip():
        raise ValueError("manager %s decision requires a non-empty next_task" % status)
    return {
        "status": status,
        "assessment": str(obj.get("assessment", "")),
        "completed": _str_list(obj.get("completed")),
        "gaps": _str_list(obj.get("gaps")),
        "problems": _str_list(obj.get("problems")),
        "next_task": str(obj.get("next_task", "")),
        "expected_result": str(obj.get("expected_result", "")),
        "validation": _str_list(obj.get("validation")),
        "constraints": _str_list(obj.get("constraints")),
        "reason": str(obj.get("reason", "")),
        # v2 task contract (optional; scope gate enforces when present):
        "hypothesis": str(obj.get("hypothesis", "")),
        "expected_files": _str_list(obj.get("expected_files")),
        "allowed_dirs": _str_list(obj.get("allowed_dirs")),
        "required_evidence": _str_list(obj.get("required_evidence")),
        "success_criteria": str(obj.get("success_criteria", "")),
    }


def parse_worker_output(text: str) -> dict:
    obj = extract_json(text)
    status = str(obj.get("status", "")).strip().lower()
    if status not in WORKER_STATUSES:
        raise ValueError("worker status must be one of %s, got %r"
                         % (list(WORKER_STATUSES), obj.get("status")))
    return {
        "status": status,
        "task": str(obj.get("task", "")),
        "changes": _str_list(obj.get("changes")),
        "tests": _str_list(obj.get("tests")),
        "commands": _str_list(obj.get("commands")),
        "evidence": _str_list(obj.get("evidence")),
        "artifacts": _str_list(obj.get("artifacts")),
        "problems": _str_list(obj.get("problems")),
        "recommendation": str(obj.get("recommendation", "")),
    }


def manager_fallback(reason: str) -> dict:
    """Safe decision used when the manager's output is unparseable."""
    return {
        "status": "retry",
        "assessment": "Manager output unparseable: %s" % reason,
        "completed": [],
        "gaps": ["manager produced malformed output; corrective re-task required"],
        "problems": [reason],
        "next_task": ("Your previous response was not valid structured JSON. "
                      "Re-read the task requirements and continue the assigned work "
                      "using repository tools; end with exactly one JSON report object."),
        "expected_result": "A valid structured worker report reflecting real tool use.",
        "validation": ["worker report parses as the required JSON schema"],
        "constraints": ["use tools; do not fabricate results"],
        "reason": "malformed manager output - automatic corrective retry",
        "hypothesis": "",
        "expected_files": [],
        "allowed_dirs": [],
        "required_evidence": [],
        "success_criteria": "",
    }


def worker_fallback(reason: str, task: str = "") -> dict:
    """Safe report used when the worker's output is unparseable."""
    return {
        "status": "failed",
        "task": task,
        "changes": [],
        "tests": [],
        "commands": [],
        "evidence": [],
        "artifacts": [],
        "problems": ["worker output unparseable: %s" % reason],
        "recommendation": "retry with a narrower task",
    }
