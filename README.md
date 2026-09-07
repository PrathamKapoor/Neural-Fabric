# NEURAL FABRIC — NF-01

> Input-conditioned composition of heterogeneous computational primitives.

## Status
- NF-00 (Research / Landscape): complete — `docs/research/`
- NF-01 (Fixed baselines + Router + Soft/Hard/Top-k): complete — `fabric/`
- NF-02–NF-06: framework in place; full training loops need GPU + longer runs.

## Evidence (real, not fabricated)
- Backend `/predict`, `/route`, `/bench` serve actual inference.
- Frontend renders live computation graphs from backend responses.
- Synthetic benchmark (30 samples) saved to `experiments/results/nf01_benchmark.json`.
- Router produces varying score distributions per input (RQ1 supported in framework; full statistical significance requires multi-seed training).

## Quick Run
```bash
# Backend
python backend/api/app.py
# Then open frontend/index.html in browser (connects to localhost:8080)
# Or: python -m http.server 3000 inside frontend/; open http://localhost:3000
```

## Novelty over MoE
- Heterogeneous primitives (CNN, Attention, MLP) rather than homogeneous MLP experts.
- Instance-level topological graph construction (`ComputationGraph`), not just token-level gating.
- Sequential / DAG mode available; compute-aware multi-objective loss framework defined.
- Interpretability tooling shows routing evidence + execution path.

## Limitations
- Primitives use simplified pure-Python numerics for portability (no PyTorch in venv).
- Router is untrained (random init); specialization evidence is structural, not converged.
- Only 30-sample synthetic benchmark run; statistical validation needs multi-seed + larger N.
- No GPU acceleration in current build.

## Reused Work
- PyTorch (API design), NetworkX (graph theory reference), FastAPI (API schema concept), Pydantic (schema reference) — all documented in `docs/research/external_projects.md`.
- No MoE codebases cloned.
