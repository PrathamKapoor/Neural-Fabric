# Limitations — FINAL (research-complete)

## Scientific (residual, accepted)
- Synthetic 64-dim benchmark with aligned structure; transfer NOT TESTED.
- Dynamic-vs-fixed comparisons are null (p≈0.2–0.4), not victories.
- No shift robustness (0.45→0.0); hard routing seed-unstable (0.74–0.98).
- Accuracy has no cheap estimator (`bench().fabric_accuracy_est is None`).

## Resolved during completion
- Router now trained (was random); multi-seed everywhere it matters.
- Deterministic CNN forward (was random noise); k-plumbing; true FLOPs.
- Fabricated 0.92 removed from engine + API, regression-tested.
- Frontend display baseline corrected to measured 17,152 FLOPs.

## Technical (accepted)
- Pure-Python numerics (torch variants available); CPU-only.
- No CI/deployment packaging review (out of scope, no push).
