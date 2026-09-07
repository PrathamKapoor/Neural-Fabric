# NF-01 Progress

## Completed
- NF-00: Research docs (landscape, related_work, novelty, external_projects, research_questions) verified.
- NF-01: Primitives (MLP, CNN, Attention) with common interface.
- Router: soft, hard, top-k with entropy/score tracking.
- FabricEngine: combines router + primitives + graph execution.
- ComputationGraph: explicit DAG representation serializable to JSON.
- Backend API: /predict, /route, /bench with CORS on :8080.
- Frontend: polished single-page demo with graph viz, routing bars, metrics, demo A/B.

## In Progress
- Full synthetic benchmark with ground-truth structural modes.
- Multi-seed statistical validation.
- Robustness evaluation (noise / shift).
- Deployment packaging (Dockerfile + docker-compose).

## Next
- NF-02: Soft routing training loop with entropy tracking.
- NF-03: Hard routing compare against fixed baselines.
- NF-04: Top-k ablation.
- NF-05: Sequential composition validation.
