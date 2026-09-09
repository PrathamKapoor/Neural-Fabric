# Results (measured; see `docs/PROJECT_COMPLETE.md` for interpretation)

- Fixed MLP 0.9475±0.0399 @ 17,152 FLOPs (8 seeds).
- Soft dynamic 0.94±0.051 @ 30,976 FLOPs; paired diff −0.005, p≈0.37 (null).
- Top-1 execution 0.94 @ 10,004.8 FLOPs (identical accuracy by construction).
- Gumbel-hard 0.888±0.104, no collapse, p≈0.21 (null).
- Specialization MI 0.869/1.099 nats, 93% correct-primitive rate.
- Robustness: 0/5 flips ≤0.05 noise; 1/5 at 0.1; 2/5 at 0.2.
- Shift: match rate 0.45 → 0.0 (negative).
- Hierarchical transfer: hier-trained dynamic 0.74 ties fixed 0.75 (p=0.47);
  flat-trained router collapses on moved signals (0.56, p=0.003) —
  specialization is location-bound.
- Fresh-seed extension (NF-15, seeds 6–10): diffs [-0.04,-0.02,-0.06,+0.02,
  -0.04], mean −0.028; combined 10-seed paired p=0.054 (null holds, one
  soft win on seed 9). Run detached via the job system with heartbeats.
- All artifacts under `experiments/results/` + `experiments/runs/`.

## Cost-model correction (v2-corrected; figures above are the v1 record)

- Primitive costs: MLP 17,152 (unchanged) · CNN 1,152→8,256 (was
  undercounted ~7×) · legacy-"attention" 12,288→2,048 (was overcounted ~6×;
  it is a truncated Wq projection, not QKV attention). Ordering flips to
  attention < CNN < MLP. Convention: 1 FLOP = 1 mul/add; bias + CNN pooling
  counted; ReLU/router-softmax excluded (see
  `fabric/primitives/cost_model.py`).
- Soft dynamic (corrected rerun): 0.94±0.051 @ 27,840 FLOPs (was 30,976).
- Top-1 execution (corrected rerun): 0.94 @ 8,835.2 FLOPs (was 10,004.8) —
  identical accuracy by construction; 48.5% below the fixed-MLP baseline
  (was 41.7%), 68.3% below full-soft execution (was 67.7%).
- NF-07 tradeoff survives correction: λ_comp=2.0 → ~11% less top-1 compute
  for −6pp accuracy (was ~10% for −4pp); no collapse in either run.
- NF-01 pareto (reporting-only recomputation): {Fixed CNN, Fixed Attention,
  Fixed MLP}; efficiency winner flips Fixed CNN→Fixed Attention
  (0.44 acc/kFLOP @ 2,096 FLOPs, acc 0.9125). All accuracies unchanged.
- NF-13 MI, NF-14/NF-12/NF-09/NF-10 accuracies + p-values: unchanged
  (bit-identical reruns where applicable). Historical v1 artifacts preserved
  alongside `*_corrected.json`; see `docs/EXPERIMENTS.md`.
