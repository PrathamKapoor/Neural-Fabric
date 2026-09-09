"""Critic: alternative explanations + discriminating experiments (stdlib).

Given a result summary, enumerate plausible alternative explanations that
would invalidate the claimed conclusion, each paired with a concrete
discriminating experiment. Deterministic rules (no LLM): the critic can
REJECT conclusions by producing an untested alternative, never by prose.
"""
from __future__ import annotations


ALTERNATIVES = (
    ("seed_artifact",
     "Result holds only for the tested seeds.",
     "Re-run on disjoint seeds; report range, not just mean."),
    ("split_artifact",
     "Train/test split construction favors the method.",
     "Swap or regenerate splits with new seeds; check stability."),
    ("parameter_advantage",
     "Winner simply has more parameters.",
     "Report params for both sides; rerun with matched capacity."),
    ("compute_advantage",
     "Winner simply used more training/inference compute.",
     "Record FLOPs/epochs for both sides; rerun matched."),
    ("hyperparameter_advantage",
     "Winner received more tuning (lr, epochs, init).",
     "Fix a shared budget; tune baselines first, then the method."),
    ("leakage",
     "Test information leaked into training (shared seeds/samples).",
     "Audit seed/data independence; rerun with fresh splits."),
    ("implementation_bug",
     "A bug (not the mechanism) produces the effect.",
     "Unit-test the mechanism in isolation; finite-difference gradients."),
    ("metric_bug",
     "The metric implementation mismeasures the effect.",
     "Recompute the metric independently; hand-derive edge cases."),
    ("benchmark_artifact",
     "The synthetic task structure predetermines the outcome.",
     "Vary task structure (new modes/distributions); check robustness."),
    ("overfitting",
     "Effect is train-set memorization, not generalization.",
     "Evaluate on held-out seeds/splits; compare train vs test gaps."),
    ("instability",
     "High seed variance makes the mean meaningless.",
     "Report per-seed outcomes and std; require stability for claims."),
    ("by_construction",
     "The effect follows trivially from the setup (e.g. same argmax).",
     "Disclose the mechanism; do not present it as an empirical win."),
    ("documentation_error",
     "Docs report a stale/rounded/misattributed number.",
     "Reconcile every doc number against artifacts (lab.analyze)."),
)


def generate_alternatives(result_summary: dict) -> list:
    """Return [{alternative, why_relevant, discriminating_experiment}].

    result_summary keys (all optional): metrics, seeds, baselines_compared,
    budgets_matched (bool), splits_shared (bool), variance_reported (bool),
    mechanism (str), docs_claim (str).
    Rules fire on missing safeguards: e.g. no budget record -> compute/
    parameter advantage alternatives; single seed set -> seed artifact.
    """
    summary = result_summary or {}
    out = []
    n_seeds = summary.get("n_seeds", 1)
    budgets = summary.get("budgets_matched", False)
    splits = summary.get("splits_shared", True)
    variance = summary.get("variance_reported", False)
    mechanism = str(summary.get("mechanism", ""))

    def add(name):
        for alt, why, exp in ALTERNATIVES:
            if alt == name:
                out.append({"alternative": alt, "why_relevant": why,
                            "discriminating_experiment": exp})

    if n_seeds is not None and n_seeds < 3:
        add("seed_artifact")
    if not budgets:
        add("parameter_advantage")
        add("compute_advantage")
        add("hyperparameter_advantage")
    if splits:
        add("split_artifact")
        add("leakage")
    if not variance:
        add("instability")
    if "argmax" in mechanism or "same model" in mechanism:
        add("by_construction")
    # Always-on checks: implementation, metric, benchmark, docs.
    add("implementation_bug")
    add("metric_bug")
    add("benchmark_artifact")
    add("overfitting")
    add("documentation_error")
    return out


def critique_conclusion(conclusion: str, evidence: dict) -> dict:
    """Judge whether a conclusion outruns its evidence.

    evidence: {n_seeds, effect_size, baseline_compared (bool),
    replicated (bool), negative_checked (bool)}.
    Returns {verdict: OK|WEAKEN|REJECT, reasons: [...]}.
    """
    reasons = []
    if evidence.get("n_seeds", 0) < 3:
        reasons.append("fewer than 3 seeds")
    if not evidence.get("baseline_compared", False):
        reasons.append("no baseline comparison")
    if not evidence.get("replicated", False):
        reasons.append("not independently reproduced")
    low = str(conclusion or "").lower()
    strong = any(w in low for w in ("prove", "sota", "novel", "breakthrough",
                                    "definitively", "guaranteed"))
    if strong and (reasons or not evidence.get("negative_checked", False)):
        reasons.append("strong language without full safeguards")
        return {"verdict": "REJECT", "reasons": reasons}
    if reasons:
        return {"verdict": "WEAKEN", "reasons": reasons}
    return {"verdict": "OK", "reasons": []}
