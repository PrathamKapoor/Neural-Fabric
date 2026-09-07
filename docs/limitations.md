# Limitations — NF-01

## Scientific
- Router weights untrained; routing distributions reflect initialization, not learned specialization. Claim "router learns" is unsupported until training loop completes.
- No multi-seed evaluation; reported variance unknown.
- Synthetic dataset small (n=30); statistical power low.

## Technical
- Pure-Python primitives slower than optimized CUDA kernels.
- No batch training loop implemented; only inference mode.
- No adversarial / corruption robustness tests executed yet.

## Deployment
- Dockerfile exists but not tested on clean machine.
- Frontend assumes localhost:8080; production needs HTTPS + proxy.

## Next Steps
- Install PyTorch, run training with entropy + balance + compute penalties.
- Scale synthetic benchmark, add corruption/shifts.
- Profile and optimize router overhead.
