# Neural Fabric: Related Work & Literature Synthesis

## 1. Mixture of Experts (MoE) Foundations
- **Jacobs, Jordan, Nowlan, Hinton (1991)**: *Adaptive Mixtures of Local Experts*. Introduced competitive learning with a gating network where each expert handles a subset of the input space.
- **Shazeer et al. (2017)**: *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. First demonstrated scaling MoE in sequence models with top-$ gating and auxiliary load-balancing loss $\mathcal{L}_{balance} = w_{balance} \cdot k \sum_i m_i P_i$.
- **Fedus, Zoph, Shazeer (2021)**: *Switch Transformers*. Scaled MoE to trillions of parameters by selecting =1$ expert per token, establishing that extreme sparsity can match or exceed dense compute efficiency with simplified routing.
- **DeepSeek-AI (2024)**: *DeepSeek-V2 / DeepSeek-V3*. Fine-grained expert segmentation (dividing standard experts into smaller sub-experts) with shared expert isolation and device-limited top-$ routing.

*Distinction from Neural Fabric*: All modern MoE architectures deploy structurally homogeneous MLP experts. They route at the token level, not the instance level, and do not synthesize heterogeneous operational pipelines (e.g. chaining spatial convolutions into relational attention).

---

## 2. Dynamic Neural Networks & Conditional Execution
- **Han et al. (2021)**: *Dynamic Neural Networks: A Survey (IEEE TPAMI)*. Systematic taxonomy dividing dynamic networks into:
  1. *Sample-dependent dynamic networks* (dynamic depth, dynamic width, dynamic routing).
  2. *Spatial dynamic networks* (pixel/region-adaptive CNNs).
  3. *Temporal dynamic networks* (frame/token-adaptive RNNs/Transformers).
- **Bengio et al. (2013)**: *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation*. Explored REINFORCE and straight-through estimators for discrete binary routing choices.
- **Rosenbaum, Klinger, Riemer (2018)**: *Routing Networks: Adaptive Selection of Non-linear Functions*. Explored recursive routing over shared functional blocks for multi-task learning using reinforcement learning.

---

## 3. Adaptive Depth & Halting Formulations
- **Graves (2016)**: *Adaptive Computation Time (ACT) for Recurrent Neural Networks*. Introduced dynamic halting probabilities $ per time step, accumulating $\sum_t h_t \approx 1$ with a ponder cost penalty.
- **Banino et al. (2021)**: *PonderNet: Learning to Ponder*. Replaced ACT's deterministic halting with a probabilistic formulation (geometric distribution prior on step count), stabilizing gradient flow and improving out-of-distribution reasoning.
- **Raposo et al. (2024)**: *Mixture-of-Depths (MoD)*. Conditioned compute allocation per token by routing tokens either through a standard transformer block or through an identity residual connection, maintaining a static compute budget per sequence.

---

## 4. Modular Networks & Neural Architecture Search
- **Andreas et al. (2016)**: *Neural Module Networks (NMNs)*. Composed question-specific neural networks from linguistic parse trees for visual question answering. Requires discrete symbolic parsers at test time.
- **Liu, Simonyan, Yang (2018)**: *DARTS: Differentiable Architecture Search*. Continuous relaxation of architecture search over a fixed candidate graph of operations using bilevel optimization.
  - *Limitation*: Search produces a single static architecture ^*$ for all inputs; does not perform instance-adaptive routing at inference.

---

## 5. Comparative Matrix

| Framework | Primitive Diversity | Graph Topology | Routing Granularity | Instance Adaptive? | Compute Penalty? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Switch Transformer** | Homogeneous (MLP) | Parallel 1-of-N | Token | No (token-level) | Load-balancing only |
| **Mixture of Depths** | Homogeneous (Block/Skip) | Linear sequence | Token | No (token-level) | Fixed capacity |
| **DARTS** | Mixed (Conv/Pool/Sep) | Static DAG | Network-wide | **No** (static after search) | Indirect |
| **Neural Module Nets** | Semi-Heterogeneous | Parse-tree DAG | Sample | Yes (via text parser) | No |
| **PonderNet** | Homogeneous (Step FFN) | Linear recurrent | Sample | Yes (depth only) | KL divergence prior |
| **Neural Fabric** | **Heterogeneous** (MLP, CNN, Attn, GNN) | **Dynamic DAG / Sequential / Top-K** | **Instance** | **Yes (end-to-end learned)** | **Explicit Multi-Objective** |
