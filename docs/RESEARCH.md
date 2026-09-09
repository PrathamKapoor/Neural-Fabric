# Research Registry (living document; canonical state is `.agent_loop/research_registry.json`)

Seeded from the completed Neural Fabric program. Statuses: PROPOSED …
COMPLETE (see `lab/registry.py`).

## RQ-NF (COMPLETE): heterogeneous conditional composition vs fixed baselines
- Verdict: comparative null (paired p≈0.37/0.21); top-1 efficiency real
  (41.7% vs fixed-MLP); specialization measurable (MI 0.87).
- Evidence: `experiments/results/nf*.json`, `docs/PROJECT_COMPLETE.md`.

## RQ-HIER (TESTED): transfer to hierarchical composition
- Question: does the NF null result generalize to 2-level hierarchical
  modes, or does depth change the routing advantage?
- Falsification: top-1 execution accuracy drops >5pp vs soft on hierarchical
  modes, or fixed baselines win significantly.
- Findings (`experiments/results/nf14_hierarchical.json`, 5 seeds):
  transfer REFUTED (flat router 0.56 on moved signals vs 0.94 flat,
  paired p=0.003 — specialization is location-bound); hier-trained dynamic
  ties fixed (null p=0.47 — main null extends to hierarchical structure).
- Status: verdicts recorded; benchmark (`evaluation/compositional.py`)
  retained for future transfer studies.

## Process lessons (failure patterns log)
- Stale doc numbers (9966 vs 10004.8) → doc↔artifact reconciliation gate.
- RNG-ordering in test construction → replication-flag checks.
- Degenerate predict-all policies → non-degeneracy review in policy tests.
- Zero-tool done+changes → fabrication guard (pre-existing).
