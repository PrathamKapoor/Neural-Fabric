# NEURAL FABRIC

## Trained local classifier (new workflow)

Neural Fabric now includes a small CPU classifier that learns from an offline,
synthetic dataset. It predicts one of three classes for a 64-number vector
using a trained router and three experts. This demonstrates end-to-end routing;
it is not a general problem solver or evidence of real-world performance.
The existing research scripts, legacy untrained demo and artifacts remain
available below. Their router-selection accuracy is not end-to-end expert
prediction accuracy. See [trained model design and limitations](docs/TRAINED_MODEL.md).

```text
64 numbers → router → selected experts → weighted logits → class probabilities
                       ↳ modeled router + expert FLOPs
```

Python 3.11+ is intended; this workflow was locally checked with Python 3.13
and CPU PyTorch 2.13. Install CPU PyTorch and pytest in a virtual environment:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install pytest
python -m neural_fabric train-demo --output artifacts/demo-model.pt
python -m neural_fabric evaluate --model artifacts/demo-model.pt --mode top1
python -m neural_fabric evaluate --model artifacts/demo-model.pt --mode soft
python -m neural_fabric predict --model artifacts/demo-model.pt --input examples/sample.json
python -m neural_fabric.server --model artifacts/demo-model.pt
```

Open http://127.0.0.1:8080/ for the local UI. The API provides `GET /health`,
`GET /model`, and `POST /predict`. For example, using curl:

```bash
curl -X POST http://127.0.0.1:8080/predict -H "Content-Type: application/json" --data-binary @examples/sample.json
```

The prediction JSON contains task `probabilities`, a separate `routing`
object with expert probabilities and selections, and a `compute` object
with signed savings and modeled FLOPs. Add `"mode": "soft"` or
`"mode": "topk", "k": 2` to the request to change execution.
The default is top-1. Run `python -m pytest tests/test_end_to_end.py
tests/test_trained_api.py -q` for the new focused checks, or
`python -m pytest tests/ -q` for the historical suite as well.

The synthetic test set has 150 examples. For a reproducible descriptive
comparison, evaluate both modes of the same checkpoint and its separately
trained fixed small MLP; repeat training with distinct `--seed` values.
Results depend on the chosen synthetic task and platform. FLOPs are analytic
estimates, not latency; evaluation also reports separately timed CPU calls.

## Historical research

The section below records the earlier research prototype. Its accuracy numbers
are router-selection results, not the end-to-end expert prediction results
described above. The legacy Dockerfile starts the untrained historical API;
it does not serve the new trained classifier.

### *Route less. Compute less. Prove it.*

A research system that tests whether **input-conditioned composition of heterogeneous computational primitives** beats fixed architectures under comparable budgets — with corrected FLOP accounting, 10-seed statistics, preserved negative results, and a live demo you can run in a minute.

The earlier experiments report a router-selection accuracy and modeled compute
tradeoff; they do not establish an end-to-end expert accuracy tradeoff.

---

## Why Neural Fabric?

Most efficiency work compares against weak or mismatched baselines:

```
Typical claim:    Our dynamic model  →  fewer FLOPs  →  (vs nobody in particular)
Neural Fabric:    Router             →  top-1 execution  →  vs matched fixed MLP/CNN/projection,
                                                     same data, splits, seeds, budget, cost model
```

The question was asked *before* the results, the cost model was corrected mid-program (CNN was undercounted ~7×, the "attention" primitive overcounted ~6×), and every null/negative finding is kept in the record — including a genuine-softmax-attention control that loses 10/10 under the fixed budget.

---

## The Research Loop

```
  QUESTION  ──────►  BASELINES  ──────►  ROUTE  ──────►  TRADEOFF
     ▲                                                        │
     │                                                        ▼
   LIMITS   ◄──────  CONTROL  ◄──────  SPECIALIZE  ◄──  NULL RESULTS
```

1. **Question:** can heterogeneous conditional composition win under matched budgets?
2. **Baselines:** fixed MLP / CNN / legacy truncated-Wq projection, identical protocol.
3. **Route:** soft training + top-k conditional execution (unselected primitives bypassed).
4. **Tradeoff:** compute-aware λ sweep — strong penalty saves −752 FLOPs for −4.8pp (10 seeds, paired).
5. **Nulls kept:** comparative accuracy null (p≈0.37), shift failure (0.45→0.0), fresh-seed null (p=0.054).
6. **Control:** genuine QKV softmax attention vs legacy projection — loses 10/10, mean −27pp.
7. **Limits:** synthetic-only, n=10, uncorrected contrasts — stated in the docs, not the footnotes.

---

## Live Demo (real engine, real numbers)

```bash
.venv\Scripts\python.exe scripts\demo_routing.py
```

```
NEURAL FABRIC 60-SECOND DEMO (live engine, top-1 routing)
mode=spatial    route=[0.335, 0.416, 0.250] selected=['cnn'] flops=8256  vs mlp=17152 savings=51.9%
mode=relational route=[0.382, 0.395, 0.224] selected=['cnn'] flops=8256  vs mlp=17152 savings=51.9%
mode=dense      route=[0.411, 0.373, 0.216] selected=['mlp'] flops=17152 vs mlp=17152 savings=0.0%
```

Each line is a live `FabricEngine.predict` + `bench` call: router probabilities, selected primitive, executed FLOPs, measured savings vs the fixed-MLP baseline. The demo uses an untrained router — the *measured* accuracy/FLOP tradeoffs live in `experiments/results/` (pointers printed by the script). A styled visual demo page ships at `frontend/index.html` (run the backend below, then open it).

---

## System Architecture

```mermaid
flowchart LR
    X["input x (64-dim)"] --> R["Router\nlinear 64x3 + top-k"]
    R -->|"top-1"| M["MLP\n17,152 FLOPs"]
    R -->|"top-1"| C["CNN\n8,256 FLOPs"]
    R -->|"top-1"| A["legacy truncated-Wq\n2,048 FLOPs"]
    M --> G["ComputationGraph\n(selected only)"]
    C --> G
    A --> G
    G --> P["prediction + executed FLOPs"]
    P --> B["bench(): savings vs fixed MLP"]
```

```mermaid
flowchart TD
    subgraph Serve ["Serving (stdlib only)"]
        API["backend/api/app.py\n/predict /route /bench /health"]
    end
    subgraph Research ["Research (numpy + torch CPU)"]
        EXP["experiments/nf*.py"]
        ART["experiments/results/*.json"]
    end
    subgraph Lab ["Autonomous lab"]
        LOOP["agent_loop/ manager→worker→verify→test→critic"]
        JOBS["lab/jobs.py detached runs"]
    end
    API --> EXP
    EXP --> ART
    LOOP --> JOBS
    JOBS --> ART
```

Convention: 1 FLOP = 1 mul/add; bias + CNN pooling counted; ReLU/router-softmax excluded (`fabric/primitives/cost_model.py`). "Attention" below is the legacy truncated-Wq projection unless "genuine" is stated.

---

## Core Capabilities

### Heterogeneous Routing
- **Three real primitives** (MLP 8,707 params · CNN · truncated-Wq projection), not three copies of one layer.
- **Top-k conditional execution** with verified primitive bypass; soft, hard, and Gumbel-hard modes.
- **Compute-aware training** (NLL + entropy + λ·FLOPs objective).

### Measured Interpretability
- Route specialization: P(correct primitive | input mode) = 0.93, MI = 0.87 nats (max 1.10).
- Location-bound: flat router collapses on moved signals (0.56, p=0.003) — specialization does not transfer.

### Falsification-First Evidence
- Comparative null, shift failure, genuine-attention negative, and hollow-transfer results all shipped in `docs/RESULTS.md`.
- 10-seed paired statistics with CIs, sign counts, and explicitly **uncorrected** p-values.

### Deployable Serving
- Stdlib-only HTTP API (`/predict`, `/route`, `/bench`, `/health`), `NF_HOST`/`NF_PORT` config, graceful shutdown — live-tested in `tests/test_serve.py`.
- Hardened Docker context (`.dockerignore` excludes secrets/venvs/scratch); dynamic build untested here — stated, not claimed.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Engine** | Pure Python (stdlib only) | Router, primitives, graph, cost model, serving |
| **Research numerics** | numpy 2.4.6, torch 2.14.0+cpu | Training variants, evaluation, tests |
| **Serving** | `http.server`, no framework | Zero-dependency API (`backend/api/app.py`) |
| **Lab** | `agent_loop/` + `lab/jobs.py` | Manager→worker→verify→test→critic loop, detached jobs |
| **Tests** | pytest / unittest, 342 tests | Unit, cost-model, provenance, security, serving, jobs |

---

## Quickstart (3 minutes)

### Prerequisites
- **Python:** 3.11.x
- **pip** (torch CPU only; no GPU code paths are validated)

### 1. Clone & install
```bash
git clone https://github.com/PrathamKapoor/Neural-Fabric.git
cd Neural-Fabric
pip install -r requirements.txt  # torch CPU: see header comment in requirements.txt
```

### 2. Run the 60-second demo
```bash
python scripts/demo_routing.py
```

### 3. Run the key experiment + inspect
```bash
python experiments/nf12_routing_modes.py   # seconds; writes experiments/results/nf12_routing_modes.json
python scripts/demo_routing.py
```

### 4. Run the full suite
```bash
python -m pytest tests/ -q   # 342 tests
```

### 5. Serve the API + open the visual demo
```bash
python backend/api/app.py   # :8080  (GET /health; NF_HOST/NF_PORT configurable)
# open frontend/index.html in a browser
```

> Rerunning an experiment overwrites its JSON in `experiments/results/` (values are deterministic; only timestamp/elapsed change). Back up the directory first if you need the committed artifacts byte-intact.

---

## 60-Second Judge Flow
1. **Run** `scripts/demo_routing.py` — routing decisions, FLOPs, savings, live.
2. **Read** the corrected headline: MLP 0.954 @ 17,152 vs top-1 0.942 @ ~8,805 (48.7% less compute on means, −1.2pp — descriptive, `docs/RESULTS.md`).
3. **Check** the tradeoff: NF-07 strong penalty −752 FLOPs (95% CI [−1084,−420], p≈0.0006 uncorrected) for −4.8pp.
4. **Confront** the negatives: accuracy null, shift 0.45→0.0, genuine attention −27pp.
5. **Reproduce**: `pytest tests/`, rerun any `experiments/nf*.py`, compare artifact JSONs.

---

## Verification & Testing
- **342 tests**: fabric units, v2 cost model, NF-01/NF-07/NF-12/NF-13/NF-14 validation, robustness, provenance, Docker/static security, serving (live HTTP), demo structure, lab jobs, 24h-session logic.
- Every behavior added ships with a test; no test was weakened or deleted to stay green.

## Documentation
- [Results & statistical caveats](docs/RESULTS.md) · [Experiments](docs/EXPERIMENTS.md) · [Research registry](docs/RESEARCH.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Architecture](docs/ARCHITECTURE.md) · [Evidence ledger](docs/PROJECT_COMPLETE.md) · [Prior art & novelty](docs/research/)

