"""Standardized experiment abstraction (stdlib only).

Every experiment produces a machine-readable artifact envelope with at
least: experiment ID, hypothesis, configuration, git commit, environment,
dataset, split, seeds, model, parameter count, training/inference budgets,
metrics, compute metrics, runtime, artifact paths, timestamp, status.

Helpers: run_seeds() for multi-seed evaluation with mean/std/CI preserved
per-seed; record_budgets() for baseline-fairness comparison.
"""
from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def git_commit(repo_root: str | Path) -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=str(repo_root), capture_output=True,
                              text=True, timeout=20)
        out = (proc.stdout or "").strip()
        return out if out else "unknown"
    except Exception:
        return "unknown"


def environment() -> dict:
    return {"platform": platform.platform(),
            "python": sys.version.split()[0]}


def mean_std(values: list) -> dict:
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0, "ci95": [0.0, 0.0]}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
               7: 2.365, 8: 2.306, 9: 2.262}
    margin = t_table.get(n - 1, 1.96) * (std / math.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean, "std": std, "n": n,
            "ci95": [mean - margin, mean + margin]}


def run_seeds(fn, seeds: list, **kwargs) -> dict:
    """Run fn(seed=..., **kwargs) per seed; never hide instability.

    Returns {per_seed: [...], <metric>: mean_std(...)} merged across the
    metric dicts each call returns. Exceptions propagate (a crashed seed is
    a failed experiment, not a missing data point).
    """
    per_seed = []
    for s in seeds:
        started = time.time()
        metrics = fn(seed=s, **kwargs)
        if not isinstance(metrics, dict):
            raise ValueError("seed fn must return a metric dict")
        per_seed.append({"seed": s, "elapsed_s": round(time.time() - started, 2),
                         **metrics})
    agg: dict = {"per_seed": per_seed, "seeds": list(seeds)}
    keys = sorted({k for row in per_seed for k in row
                   if isinstance(row[k], (int, float)) and not isinstance(row[k], bool)})
    for k in keys:
        agg[k] = mean_std([row[k] for row in per_seed])
    return agg


def record_budgets(**kwargs) -> dict:
    """Record comparability budgets (params, flops, epochs, seeds...)."""
    allowed = ("params", "flops", "epochs", "seeds", "train_n", "test_n",
               "optimizer", "lr", "dataset", "split", "inference_budget")
    return {k: kwargs[k] for k in allowed if k in kwargs}


def check_budget_parity(baseline: dict, candidate: dict,
                        numeric_keys=("params", "flops", "epochs")) -> list:
    """Flag fairness mismatches between baseline and candidate budgets."""
    findings = []
    for k in numeric_keys:
        b, c = baseline.get(k), candidate.get(k)
        if isinstance(b, (int, float)) and isinstance(c, (int, float)):
            if b > 0 and abs(c - b) / b > 0.05:
                findings.append(
                    "budget mismatch %s: baseline=%s candidate=%s" % (k, b, c))
    for k in ("dataset", "split", "optimizer"):
        if (k in baseline and k in candidate
                and baseline[k] != candidate[k]):
            findings.append("budget mismatch %s: %r vs %r"
                            % (k, baseline[k], candidate[k]))
    return findings


def write_artifact(path: str | Path, envelope: dict) -> str:
    """Write the standardized envelope (creates parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope, indent=2)[:1000000],
                    encoding="utf-8")
    return str(path)


def build_envelope(experiment_id: str, hypothesis: str, config: dict,
                   dataset: dict, model: dict, budgets: dict,
                   metrics: dict, compute: dict, artifacts: list,
                   status: str = "complete", repo_root: str = ".") -> dict:
    """Assemble the standard artifact envelope (JSON-serializable)."""
    return {
        "experiment_id": experiment_id,
        "hypothesis": hypothesis,
        "config": config,
        "git_commit": git_commit(repo_root),
        "environment": environment(),
        "dataset": dataset,
        "model": model,
        "budgets": budgets,
        "metrics": metrics,
        "compute": compute,
        "artifacts": artifacts,
        "timestamp": int(time.time()),
        "status": status,
    }
