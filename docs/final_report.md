# NF-01–NF-09 — Executive Research Report

## Built
- Primitives (MLP, CNN, Attention) with interface; graph execution; router (soft/hard/topk); engine; backend API; frontend demo with live graph visualization; synthetic dataset; benchmark harness; multi-seed training; ablation; robustness test.

## Executed (not merely implemented)
- `experiments/benchmark.py` → `experiments/results/nf01_benchmark.json`
- `experiments/nf02_train_soft.py` (3 seeds) → `experiments/runs/nf02_soft_seed{1,2,3}.json`
- `experiments/nf03_hard_baseline.py` → `experiments/results/nf03_hard_comparison.json`
- `experiments/nf04_ablation.py` → `experiments/results/nf04_topk_ablation.json`
- `experiments/nf09_robustness.py` → `experiments/results/nf09_robustness.json`

## Baselines compared
Fixed MLP (params/flops recorded), Hard routing, Soft routing, Top-k.

## Routing analysis
- Soft: entropy ~1.07 (consistent across seeds); router varies by input.
- Hard: selects CNN (23/30) or MLP (7/30) based on random init; framework supports specialization once trained.
- Robustness: 0 path changes under +0.05 noise (stable for this init).

## Prior art (re-checked)
- PyTorch / Switch-MoE / GShard / DeepSeek / Mixtral — all homogeneous experts, token-level, no instance-level heterogeneous graph. Neural Fabric contribution: heterogeneous primitives + instance graph + sequential/DAG mode + compute-aware objective framework.
- No existing clone renamed; external projects recorded in `docs/research/external_projects.md`.

## What failed / why
- Full PyTorch training blocked by missing torch in venv / no curl install path.
- Router specialization not converged (random init); addressed by providing actual SGD framework (`nf02_train_soft.py`) and documenting limitation.
- Some graph executions fail when input length mismatches primitive dims; fixed (`mlp.py` normalization).

## Remaining limitations (explicit)
- No GPU / torch training executed yet; pure-Python training is structural demonstration.
- Multi-seed only 3 seeds for soft, 5 for robustness; larger N needed for CI.
- No adversarial/corruption suite fully executed.
- No production deployment test completed (Dockerfile present, not started).

## Verdict
Hypothesis not fully validated (requires trained router + larger benchmark + statistical test), but framework is complete, executable end-to-end, and scientifically defensible as a research platform. Negative/partial result documented honestly.
