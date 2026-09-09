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
- All artifacts under `experiments/results/` + `experiments/runs/`.
