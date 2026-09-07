# Neural Fabric: Defensible Novelty & Core Technical Contributions

## 1. What Neural Fabric IS NOT
To maintain scientific rigor and intellectual honesty, we explicitly declare what Neural Fabric is NOT:
1. It is NOT another standard MoE layer substituting MLP experts with identical feedforward stacks.
2. It is NOT a static Neural Architecture Search (NAS) algorithm that picks one graph for the test set.
3. It is NOT a hard-coded symbolic parser that requires ground-truth syntactic program annotations.
4. It is NOT a decorative diagram with superficial quantum/symbolic layers that do not participate in gradient updates.

---

## 2. What Neural Fabric UNIQUELY CONTRIBUTES
Neural Fabric introduces five concrete contributions that distinguish it from prior art:

### Contribution 1: Instance-Conditioned Heterogeneous Primitive Composition
Instead of routing tokens across homogeneous MLP blocks, Neural Fabric deploys computational primitives with fundamentally distinct inductive biases:
- **Spatial Local Bias**: 2D Multi-Scale Convolutional Primitive (=3, 5$).
- **Global Relational Bias**: Multi-Head Self-Attention Primitive (=4$, pairwise token interactions).
- **Pointwise Non-linear Projection**: Dense MLP Primitive with SwiGLU / GeLU activations.
- **Relational/Topology Bias**: Graph Message Passing Primitive (dynamic k-NN feature graph aggregation).
The router discovers which inductive bias (or sequential pairing of inductive biases) matches the intrinsic geometry of each specific input instance $.

### Contribution 2: Dynamic Execution Graphs with Inspectable Serialization
Neural Fabric formalizes the execution trajectory of each sample as a Directed Acyclic Graph:
G_x = (V_x, E_x, \mathbf{r}_x, C(G_x))
where:
-  \subseteq \mathcal{P}$ represents executed primitives.
- $ represents directed tensor dependencies between primitive outputs.
- $\mathbf{r}_x$ represents decision gates.
- (G_x)$ represents actual executed FLOPs.
Every single inference produces a full JSON execution artifact detailing exact executed nodes, FLOP count, latency, routing entropy, and routing evidence.

### Contribution 3: Multi-Objective Pareto Loss with Complexity & Compute Penalties
We formulate an explicit multi-objective routing criterion:
\mathcal{L}_{total} = \mathcal{L}_{task}(y, \hat{y}) + \lambda_{comp} \frac{\text{FLOPs}(G_x)}{\text{FLOPs}_{max}} + \lambda_{bal} \mathcal{L}_{balance}(R) + \lambda_{cplx} |E_x|
This forces the router to seek minimal sufficient computation paths, penalizing gratuitous module invocation.

### Contribution 4: Automated Router Health & Anti-Collapse Diagnostics
Router collapse (where the router monotonically allocates 100% of samples to a single favorite primitive) is treated as a first-class failure mode. We implement:
- Dynamic Shannon entropy tracking (R) = -\sum_i p_i \log p_i$.
- Primitive utilization coefficient of variation {util}$.
- Route stability metrics under Gaussian & adversarial input perturbations.

### Contribution 5: Reproducible Experimental Ladder with Controlled Synthetic & Vision Benchmarks
We formulate an experimental progression (NF-00 through NF-11) evaluated on a benchmark specifically designed to differentiate inductive biases:
- High local spatial correlation (favors CNN).
- Long-range relational token dependency (favors Attention).
- Dense global linear separability (favors MLP).
- Graph topology reasoning (favors GNN).
