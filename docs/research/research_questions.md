# Neural Fabric: Research Questions & Scientific Hypotheses

## Core Hypothesis
\mathcal{H}_{core}: \text{Input-conditioned dynamic composition of heterogeneous primitives achieves superior Pareto efficiency (accuracy vs. FLOPs)} \\ \text{and out-of-distribution robustness compared to fixed homogeneous architectures under identical compute constraints.}

---

### RQ1: Input-Conditioned Primitive Selection
- **Question**: Can a lightweight parameterized router learn to select computational primitives conditioned on individual sample features rather than adopting a static uniform choice?
- **Null Hypothesis (^{(1)}$)**: Routing probability distributions (r|x)$ are statistically indistinguishable from a constant marginal distribution (r) = \mathbb{E}_x[p(r|x)]$, with mutual information (X; R) \approx 0$.
- **Alternative (^{(1)}$)**: Router entropy varies systematically with input characteristics; inputs with distinct inductive requirements activate distinct primitives with high confidence ((X; R) > 0$).

### RQ2: Heterogeneous Composition vs. Homogeneous Baselines
- **Question**: Does composing heterogeneous primitives (CNN + Attention + MLP) yield lower task loss than parameter-matched homogeneous baselines (All-CNN, All-Attention, All-MLP)?
- **Evaluation**: Compare parameter-matched fixed models against Neural Fabric across hybrid multi-modal / multi-bias benchmarks.

### RQ3: Accuracy Under Fixed Compute Budgets
- **Question**: Under an identical inference FLOP budget, does hard conditional execution of Neural Fabric outperform static compute-matched baselines?
- **Metric**: Accuracy vs. Executed FLOPs (Pareto frontier).

### RQ4: Emergence of Inductive Specialization
- **Question**: Does the router allocate localized spatial features to convolutional operators and relational/global dependencies to attention operators without direct supervision on routing targets?
- **Evaluation**: Benchmark on synthetic datasets with ground-truth structural modes (Spatial vs. Relational vs. Dense).

### RQ5: Robustness Under Distribution Shift
- **Question**: How do dynamic routing networks perform when subjected to out-of-distribution (OOD) corruption and domain shift compared to fixed architectures?
- **Hypothesis**: Neural Fabric can adaptively alter its execution graph under covariate shift, preserving core predictive features or degrading gracefully.

### RQ6: Complexity Regularization & Compute Control
- **Question**: Can explicit penalty terms $\lambda_{comp} \mathcal{L}_{comp} + \lambda_{cplx} \mathcal{L}_{cplx}$ smoothly steer the average graph depth and module count without catastrophic loss of accuracy?

### RQ7: Interpretability & Decision Signals
- **Question**: Do routing logits provide faithful decision signals that correlate with input feature saliency and module necessity?

### RQ8: Dynamic DAG Generalization
- **Question**: Does sequential / DAG composition provide measurable benefits over flat parallel 1-of-N or Top-K expert fusion?
