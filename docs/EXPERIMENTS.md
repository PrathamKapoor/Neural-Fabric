# Experiments

Every experiment writes a machine-readable artifact (see
`lab/experiment.py::build_envelope`): ID, hypothesis, config, git commit,
environment, dataset, split, seeds, model, budgets, metrics, compute,
artifacts, timestamp, status. Seeds are never averaged away: per-seed rows
persist alongside mean/std/CI.

## Neural Fabric battery (all CPU-only, deterministic by seed)

| ID | Question | Seeds | Artifact |
|---|---|---|---|
| NF-01 | fixed baselines | 8 | `experiments/results/nf01_fixed_baselines.json` |
| NF-02 | soft routing | 5 | `.../nf02_soft_training.json` |
| NF-03 | hard baseline (untrained) | 3 | `.../nf03_hard_baseline_*.json` |
| NF-04 | top-k compute ablation | 10 samples | `.../nf04_topk_ablation.json` |
| NF-07 | compute-aware λ sweep | 3×9 cells | `.../nf07_sweep_matrix.json` |
| NF-09 | noise curve (5 levels) | 5 | `.../nf09_robustness.json` |
| NF-10 | distribution shift | — | `.../nf10_distribution_shift.json` |
| NF-11 | true-FLOP scaling | 100 samples | `.../nf11_scalable_100.json` |
| NF-12 | routing modes + paired tests | 5 | `.../nf12_routing_modes.json` |
| NF-13 | specialization (MI) | 5 | `.../nf13_specialization.json` |
| NF-14 | hierarchical transfer | 5 | `.../nf14_hierarchical.json` |
| NF-15 | fresh-seed extension (6–10) | 5 | `.../nf15_seed_extension.json` |

Rerun any script directly (e.g. `.venv\Scripts\python.exe
experiments\nf02_soft_training.py`); artifacts regenerate byte-identically
modulo timestamps and wall-clock fields.
