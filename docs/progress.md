# Progress — FINAL (research-complete)

## Completed
- NF-00 research grounding (`docs/research/`).
- NF-01 fixed baselines, 8 seeds, Pareto + t-tests.
- NF-02 soft routing, 5 seeds, no collapse, reproduced byte-identically.
- NF-03 hard baseline (untrained inference comparison).
- NF-04 top-k compute ablation (extended by NF-12 trained comparison).
- NF-07 compute-aware λ sweep, real 3×3×3 grid, gradients verified.
- NF-09 robustness noise curve (5 levels).
- NF-10 distribution shift (negative result kept).
- NF-11 true-FLOP scalable logging.
- NF-12 trained routing modes (soft/top-k/Gumbel-hard) + paired tests.
- NF-13 routing specialization (MI 0.87 nats).
- Integrity: fabricated `bench()` 0.92 removed (engine + API), regression-tested.
- 254/254 tests passing. Docs current. `docs/PROJECT_COMPLETE.md` marks DONE.

## Deliberately not pursued (documented in PROJECT_COMPLETE)
- Real-world datasets, STE variants, DAG branching, GPU scaling — see SUGGESTED.
