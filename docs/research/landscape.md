# Neural Fabric: Architectural Landscape & Taxonomy of Conditional Computation

## 1. Executive Taxonomy
Modern deep learning architectures overwhelmingly execute static computational graphs: given input  \in \mathcal{X}$, an invariant sequence of tensor operations \theta(x) = f_L \circ f_{L-1} \circ \dots \circ f_1(x)$ is evaluated. This imposes a rigid inductive bias across all inputs regardless of sample complexity, intrinsic dimensionality, or required perceptual structure.

Conditional computation and dynamic neural architectures break this constraint. We classify existing paradigms across four orthogonal dimensions:

`	ext
                                  TAXONOMY OF CONDITIONAL COMPUTATION
                                                  │
          ┌───────────────────────┬───────────────┴───────────────┬────────────────────────┐
          ▼                       ▼                               ▼                        ▼
  [Granularity]          [Homogeneity]                     [Topology]              [Decision Method]
  ├ Token-level (MoE)    ├ Homogeneous Experts (MLP/MLP)   ├ 1-of-N Fan-out        ├ Soft Routing (Convex Comb)
  ├ Instance-level       └ Heterogeneous Primitives        ├ Top-K Dispatch        ├ Hard Discrete (Gumbel/Top-k)
  ├ Layer-level (Early)    (CNN + Attn + Relational)       ├ Dynamic Depth/Exit    ├ Policy Gradient (RL)
  └ Graph-level (DAG)                                      └ Dynamic DAG (G_x)     └ Recurrent Controller
`

---

## 2. Granularity Dimensions

### 2.1 Token-Level Sparsity (Transformer MoE)
- **Mechanism**: Input sequence  = [x_1, \dots, x_T]$. A router maps each token vector  \in \mathbb{R}^d$ to a probability simplex over $ experts: (x_t) = \text{Softmax}(W_g x_t)$.
- **Examples**: Switch Transformer (Fedus et al., 2021), GShard (Lepikhin et al., 2020), Mixtral 8x7B (Jiang et al., 2024), DeepSeek-V2/V3 (DeepSeek-AI, 2024).
- **Limitation**: Experts are structurally identical feed-forward networks (homogeneous MLP blocks). No heterogeneous inductive bias (e.g., 2D spatial convolution vs. long-range token attention).

### 2.2 Instance-Level Routing
- **Mechanism**: A routing function acts once per problem instance or whole sample, dispatching the entire input tensor to specific modules.
- **Benefits**: Eliminates fine-grained token dispatch and permutation overhead; preserves cache locality and batch tensor execution paths.

### 2.3 Layer-Level / Dynamic Depth (Early Exit)
- **Mechanism**: Intermediate auxiliary classifiers or halting gates decide if early exit criteria are satisfied: (\text{halt} | h_l) > \tau$.
- **Examples**: BranchyNet (Teerapittayanon et al., 2016), DeeBERT (Xin et al., 2020), Adaptive Computation Time (Graves, 2016), PonderNet (Banino et al., 2021).
- **Limitation**: Path is linearly sequential; early layers cannot be skipped to access late specialized modules.

### 2.4 Graph-Level Dynamic Synthesis (Neural Fabric)
- **Mechanism**: The architecture for input $ is an instance-specific Directed Acyclic Graph  = (V_x, E_x)$, where vertices  \in V_x$ draw from a library of heterogeneous computational primitives (PROPOSED full set $\mathcal{P} = \{ \text{MLP}, \text{CNN}, \text{Self-Attention}, \text{Relational/GNN}, \text{Memory} \}$; IMPLEMENTED subset is $\{\text{MLP (ReLU)}, \text{CNN (k=3)}, \text{legacy attention projection}\}$ — see novelty.md §0).
- **Execution**: Topological traversal of $, evaluated dynamically or sequentially with state-conditioned early termination.

---

## 3. Homogeneity vs. Heterogeneity

| Attribute | Standard MoE (e.g. Mixtral, Switch) | Heterogeneous Neural Fabric |
| :--- | :--- | :--- |
| **Expert Primitives** | Homogeneous FFNs (identical MLP dimensions) | Heterogeneous operators: Spatial Conv, Relational Attention, Dense FFN |
| **Inductive Bias** | Uniform translation/dense point-wise bias | Diverse inductive biases matched to input attributes |
| **Routing Granularity** | Per-token gating | Per-instance topological composition + layer gating |
| **Topology** | Parallel fan-out / sum-aggregation | Configurable: Parallel Top-K, Sequential chain, Dynamic DAG |
| **Failure Mode** | Representation collapse (router ignores some MLPs) | Structural bias mismatch (wrong operator chosen) |
| **Compute Metric** | FLOPs per token =  \times k \times d_{ffn}$ | Dynamic FLOPs depending on active primitive sequence |

---

## 4. Optimization & Gradient Flow Dilemmas

1. **Continuous Relaxation (Soft Gating)**:
   \hat{y} = \sum_{i=1}^M r_i(x) f_i(x)
   Differentiable via standard backpropagation. **Drawback**: Requires evaluating all $ primitives, sacrificing computational efficiency at training and inference.
2. **Discrete Hard Selection**:
   i^* = \arg\max_i r_i(x)
   Non-differentiable $\frac{\partial i^*}{\partial r}$. Requires:
   - Straight-Through Gumbel-Softmax (Jang et al., 2016; Maddison et al., 2016).
   - REINFORCE / Actor-Critic policy gradient.
   - Differentiable Top-K with auxiliary load-balancing loss.
3. **Multi-Objective Loss Formulation**:
   \mathcal{L} = \mathcal{L}_{task}(y, \hat{y}) + \lambda_{comp} \mathcal{L}_{compute}(G_x) + \lambda_{bal} \mathcal{L}_{balance}(R) + \lambda_{cplx} \mathcal{L}_{complexity}(G_x)
