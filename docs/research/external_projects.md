# Neural Fabric: External Project Reuse & Open-Source Attribution

In compliance with the project reuse policy (Section 4), this document maintains a complete, auditable record of mature open-source tools, foundational libraries, and frameworks integrated into Neural Fabric.

---

## Reused Infrastructure Matrix

| Project | Repository / Package | License | What Was Reused | Why Reused | Neural Fabric Contribution & Novelty |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch** | pytorch/pytorch | BSD-3-Clause | Tensor autograd engine, nn.Module base classes, CUDA kernels | Industry-standard deep learning runtime; avoids error-prone reimplementation of low-level autograd | Heterogeneous primitive interface, dynamic execution graph manager, straight-through routing controllers |
| **NetworkX** | 
etworkx/networkx | BSD-3-Clause | DAG representation, topological sort, cycle validation | Robust graph theory primitives with verified asymptotic correctness | Custom ComputationGraph wrapper, tensor-state passing along DAG edges, graph complexity FLOPs accounting |
| **FastAPI** | 	iangolo/fastapi | MIT | High-performance ASGI HTTP server scaffolding | Standard production API framework with OpenAPI schema generation | Real-time /predict, /route, /graph, /benchmarks endpoints streaming live execution graphs and routing decisions |
| **Pydantic** | pydantic/pydantic | MIT | Schema validation and serialization | Strict type contracts for input tensors, graph JSONs, and benchmark records | Custom schemas for PrimitiveNode, GraphExecutionResult, RoutingEvidence, AblationRun |
| **Bun** | oven-sh/bun | MIT | High-speed JavaScript/TypeScript runtime & package manager | Direct native execution, SOTA developer experience | Rapid benchmark test runners, visualization scripts |

---

## Explicit Assurance of Originality
- No existing MoE codebases (e.g. Megatron-MoE, DeepSpeed-MoE, Tutel) were cloned or cosmetically renamed.
- All router mechanisms (Soft, Gumbel Hard, Top-K, Sequential, Dynamic DAG controller), primitives, benchmarking harnesses, and ablation pipelines are authored natively for this investigation.
