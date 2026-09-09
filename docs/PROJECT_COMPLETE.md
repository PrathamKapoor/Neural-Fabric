# Neural Fabric — PROJECT COMPLETE

**Status: RESEARCH-COMPLETE. Marked DONE. No git push performed.**

Central question: *can input-conditioned composition of heterogeneous
computational primitives improve accuracy, efficiency, robustness, or
interpretability vs fixed architectures under comparable budgets?*

**Conclusion (evidence-backed): on the controlled synthetic benchmark, the
dynamic fabric does NOT beat strong fixed baselines on accuracy (all paired
comparisons null), but DOES deliver matched accuracy at substantially lower
executed compute via top-1 execution, with genuinely interpretable routing.
A null comparative result with a real efficiency finding — not a victory.**

## IMPLEMENTED (code exists and runs)

- Heterogeneous primitives (MLP/CNN/Attention, pure-Python + torch variants)
  with `forward/params/flops`.
- Router: soft / hard / top-k, k-plumbing, Gumbel-softmax hard training.
- `ComputationGraph` (sequential/DAG execution, JSON-serializable).
- `FabricEngine`: predict (soft + graph modes), route, bench, conditional
  execution verified (executed FLOPs == selected-primitive FLOPs).
- Backend API (`/predict /route /bench /sequential`), frontend demo.
- Experiments NF-01–NF-04, NF-07, NF-09–NF-13 (+ torch demo, sweep stub
  replaced by real NF-07).
- Autonomous loop v2 (manager→builder→tester, gates, state machine).

## VALIDATED (measured, reproduced, falsification-attempted)

| Claim | Evidence |
|---|---|
| NF-01 baselines (8 seeds) | `nf01_fixed_baselines*.json`, reruns byte-identical |
| NF-02 soft 0.94±0.051, no collapse | `nf02_soft_training*.json`, reruns byte-identical |
| Soft vs best-fixed null (p≈0.35–0.37) | independent paired recomputation (NF-12) |
| Top-1 execution: 0.94 @ 9,966 FLOPs | `nf12_routing_modes.json` |
| Gumbel-hard trains, no collapse, null (p=0.21) | same artifact; diversity 0.36–0.52 max-ratio |
| NF-07 tradeoff (~4pp acc ↔ ~10% top-1 FLOPs) | `nf07_sweep_matrix.json`; gradients finite-diff verified 1e-10 |
| Specialization MI 0.87/1.10, 93% correct-prim | `nf13_specialization.json`; MI independently recomputed exact |
| Robustness stable ≤0.05 noise, degrades ≥0.1 | `nf09_robustness.json` (5-level curve) |
| Conditional execution works | operator-local probe + `.agent_loop/verification_conditional_exec.json` |
| 254/254 tests pass | `python -m unittest discover -s tests` |

## Negative / null findings (kept, not hidden)

- Dynamic accuracy does not exceed fixed baselines (soft p≈0.37, hard p≈0.21).
- Distribution shift destroys routing (match 0.45→0.0): no shift robustness.
- Hard routing has high seed variance (0.74–0.98): less stable than soft.
- NF-07 penalties move compute modestly; no collapse induced or cured.
- `bench().fabric_accuracy_est` is `None` by design (was hardcoded 0.92 —
  removed, regression-tested; same for the backend `/bench` copy).

## SUGGESTED (not started, would be next)

- Real-world datasets (CIFAR-scale) to test transfer of the null result.
- STE hard-routing variants; learned halting/depth; DAG branching beyond chains.
- Stronger shift/robustness batteries; calibration analysis.
- GPU scaling and wall-clock efficiency claims.

## NOT TESTED

- Real quantum/hardware execution (N/A — classical project).
- Production deployment, load, security review (explicitly out of scope).
- Frontend as evaluation harness (demo only; display baseline corrected to
  measured 17,152 FLOPs).

## Reproducibility

Fixed seeds in every artifact; `experiments/provenance/` holds generation
scripts; reruns verified byte-identical modulo timestamps/wall-clock.
CPU-only, ~2 min for the full experimental battery.

## What remains

Nothing required. Optional next steps are listed under SUGGESTED. Further
polishing without a new research question would violate the stop rule —
hence DONE.
