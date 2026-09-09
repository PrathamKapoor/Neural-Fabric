# NEURAL FABRIC — research-complete

> Input-conditioned composition of heterogeneous computational primitives:
> does dynamic routing beat fixed architectures under comparable budgets?
> **Answer on the synthetic benchmark: no — with reduced compute at matched
> accuracy via top-1 execution, and interpretable route specialization.**
> See `docs/PROJECT_COMPLETE.md` for the full evidence ledger.

## Status: RESEARCH-COMPLETE (all findings measured, negative results kept)

| Area | State | Evidence |
|---|---|---|
| Fixed baselines (MLP/CNN/Attention, 8 seeds) | VALIDATED | `experiments/results/nf01_fixed_baselines.json` |
| Soft routing (5 seeds, no collapse) | VALIDATED | `experiments/results/nf02_soft_training.json` |
| Top-k execution efficiency (k=1,2,3) | VALIDATED | `experiments/results/nf12_routing_modes.json` |
| Gumbel-hard trained routing (5 seeds) | VALIDATED | same artifact (null vs fixed, p=0.21) |
| Compute-aware λ sweep (3×3×3 seeds) | VALIDATED | `experiments/results/nf07_sweep_matrix.json` |
| Route specialization (MI 0.87/1.10) | VALIDATED | `experiments/results/nf13_specialization.json` |
| Robustness curve (stable ≤0.05 noise) | VALIDATED | `experiments/results/nf09_robustness.json` |
| Distribution shift (0.45→0.0, negative) | VALIDATED | `experiments/results/nf10_distribution_shift.json` |
| Conditional execution | VALIDATED | `.agent_loop/verification_conditional_exec.json` |
| Unit/integrity suites (254 tests) | VALIDATED | `tests/` |

## Headline results (all reproduced byte-identically except wall-clock noise)

- Best fixed: MLP **0.9475±0.0399** @ 17,152 FLOPs (8 seeds).
- Soft dynamic: **0.94±0.051** @ 30,976 FLOPs — paired diff −0.005, **p=0.35 (null)**.
- Top-1 execution of the same router: **0.94 @ 9,966 FLOPs** — identical
  accuracy by construction (same argmax as soft); 41.9% less compute than
  the fixed-MLP baseline (17,152 FLOPs) and 67.8% less than full-soft
  execution (30,976 FLOPs). Efficiency without superiority.
- Gumbel-hard: 0.888±0.104, no collapse, p=0.21 vs fixed (null, higher variance).
- Specialization is real and interpretable: P(correct primitive | input mode) = 0.93, MI = 0.87 nats (max 1.10).

## Quick Run

```bash
# Full test suite (stdlib + torch CPU only)
.venv\Scripts\python.exe -m unittest discover -s tests
# Key experiments (seconds each unless noted)
.venv\Scripts\python.exe experiments\nf02_soft_training.py
.venv\Scripts\python.exe experiments\nf01_fixed_baselines.py   # ~80s
.venv\Scripts\python.exe experiments\nf12_routing_modes.py
.venv\Scripts\python.exe experiments\nf07_sweep.py
.venv\Scripts\python.exe experiments\nf13_specialization.py
.venv\Scripts\python.exe experiments\nf09_robustness.py
# Backend demo
.venv\Scripts\python.exe backend\api\app.py   # :8080, open frontend/index.html
```

## Reproducibility

Every result artifact regenerates deterministically from its script (fixed
seeds recorded in-file); NF-01/NF-02 artifacts were re-run and verified
byte-identical excluding timestamps and wall-clock timings. Provenance
scripts live in `experiments/provenance/`. No GPU required.

## Limitations (honest)

- Synthetic 64-dim benchmark with aligned primitive/mode structure; real-world
  transfer NOT TESTED.
- Pure-Python numerics (torch variants exist for real-gradient runs).
- Hard routing trained only via Gumbel-softmax; STE variants NOT TESTED.
- `bench().fabric_accuracy_est` is intentionally `None` — accuracy has no
  cheap estimator; see `docs/PROJECT_COMPLETE.md`.
- Frontend is a demo, not an evaluation harness.

## Novelty over MoE

Heterogeneous (not homogeneous) primitives, instance-level topological
graphs, compute-aware training, and measured route interpretability — with
the null comparative result preserved rather than hidden.
