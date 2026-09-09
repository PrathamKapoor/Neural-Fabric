"""Idea generation with mandatory checks (stdlib).

Generated ideas must pass: NOVELTY (researcher attack), FEASIBILITY
(cheap falsification possible with local compute), TESTABILITY (measurable
outcome + falsification condition), BASELINE (an appropriate control
exists), FALSIFICATION (a concrete observation that would kill it).

The generator is template- and registry-driven (no LLM): it proposes
follow-ups from open registry questions (unresolved gaps, failed
alternatives, suggested next steps) and scores them. It never invents
speculative mega-ideas; every idea names its measurable question.
"""
from __future__ import annotations

from . import researcher


FOLLOWUP_TEMPLATES = (
    ("transfer", "Does {mechanism} transfer to {new_regime}?",
     "Run the existing protocol on a new regime; compare effect sizes."),
    ("lesion", "Which component causes {effect}? Remove {component} and rerun.",
     "Ablation with matched budgets; effect must shrink or the claim weakens."),
    ("stress", "Does {effect} survive {harder_condition}?",
     "Stress the finding (noise/shift/seeds); report breakage honestly."),
    ("baseline", "Does {effect} survive a stronger baseline ({baseline})?",
     "Strengthen the control first; rerun the comparison."),
    ("mechanism", "Is {effect} caused by {mechanism} or a confound?",
     "Design a discriminating experiment (see lab.critic)."),
)


def generate_from_registry(registry, repo_root: str = ".",
                           max_ideas: int = 5) -> list:
    """Propose follow-ups from open questions; each fully check-gated."""
    ideas = []
    for q in registry.open_questions():
        gaps = list(q.get("limitations", [])) + list(q.get("next_questions", []))
        mechs = [q.get("hypothesis", "") or q.get("title", "")]
        for mech in mechs[:1]:
            for kind, title_t, how in FOLLOWUP_TEMPLATES:
                title = title_t.format(mechanism=mech[:60] or "the effect",
                                       new_regime="a harder task family",
                                       effect="the effect",
                                       component="one component",
                                       harder_condition="stress conditions",
                                       baseline="stronger controls")
                checks = run_checks(title, mech, repo_root, registry)
                ideas.append({"title": title, "kind": kind, "how": how,
                              "checks": checks,
                              "passes": passes(checks)})
                if len(ideas) >= max_ideas:
                    return ideas
        for gap in gaps[:2]:
            title = "Close gap: %s" % str(gap)[:120]
            checks = run_checks(title, gap, repo_root, registry)
            ideas.append({"title": title, "kind": "gap", "how": str(gap)[:200],
                          "checks": checks, "passes": passes(checks)})
            if len(ideas) >= max_ideas:
                return ideas
    return ideas


def run_checks(title: str, focus: str, repo_root: str = ".",
               registry=None) -> dict:
    """The five mandatory gates.

    novelty/testability/not_duplicate are decided here. feasibility,
    baseline, and falsification CANNOT be verified statically, so they are
    marked "deferred" (must be satisfied at hypothesis-design time via
    lab.registry.add_hypothesis, which enforces them) rather than faked True.
    """
    keywords = [w for w in str(focus).lower().split()
                if len(w) > 4][:6] or [str(title).lower()[:20]]
    attack = researcher.attack_novelty(title, keywords, repo_root)
    already = False
    if registry is not None:
        already = any(str(title).lower() == t.lower()
                      for t in registry.idea_titles())
    testable = bool("?" in title or "whether" in title.lower()
                    or "does" in title.lower())
    return {
        "novelty": attack["verdict"] in ("NARROW_GAP", "OPEN"),
        "novelty_detail": attack["verdict"],
        "feasibility": "deferred: confirm cheap falsification at design",
        "testability": testable,
        "baseline": "deferred: baseline engine check at design",
        "falsification": "deferred: enforced by add_hypothesis at design",
        "not_duplicate": not already,
    }


def passes(checks: dict) -> bool:
    """Statically decidable gates must pass; deferred ones must be honored."""
    return bool(checks.get("novelty") and checks.get("testability")
                and checks.get("not_duplicate"))
