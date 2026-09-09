"""NX-ATTN-01 — NEW exploratory experiment (NOT part of the NF ladder).

Question: "Does genuine attention behave differently from the legacy linear
projection under the same evaluation setup?"

Fair-comparison design (Part 10 compliant):
- Same dataset/splits/seeds/protocol/optimizer as the NF-01 10-seed arms
  (seeds 1-10, 30 epochs, SGD lr=0.05, n_train=100, n_test=50).
- Same model template as FixedAttentionModel (8-feat trunk + 8->3 head);
  only the trunk changes: SoftmaxAttention instead of legacy projection.
- Same optimization budget. No tuning. Baseline arms (mlp/cnn/legacy) are
  REUSED from experiments/results/nf01_fixed_10seed_corrected.json —
  not retrained, not modified.

Writes: experiments/results/nx_attn_genuine_vs_legacy.json
"""
import json
import math
import os
import random
import sys
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from fabric.primitives.genuine_attention import SoftmaxAttention
from fabric.primitives.cost_model import COST_MODEL_VERSION, PRIM_FLOPS
from evaluation.synthetic_dataset import make_sample


class FixedGenuineAttentionModel:
    """8-feat genuine-attention trunk + 8->3 linear head; SGD-trained."""

    def __init__(self, n_tok=8, d_tok=8, out_dim=3, seed=0):
        random.seed(seed)
        self.trunk = SoftmaxAttention(n_tok=n_tok, d_tok=d_tok, seed=seed)
        self.n_tok, self.d_tok, self.out_dim = n_tok, d_tok, out_dim
        self.num_feats = d_tok
        self.W_head = [[random.gauss(0, 0.05) for _ in range(out_dim)]
                       for _ in range(self.num_feats)]
        self.b_head = [random.gauss(0, 0.05) for _ in range(out_dim)]
        self.name = "softmax_attention"

    def forward(self, x):
        n, d = self.n_tok, self.d_tok
        if not isinstance(x, list):
            x = list(x)
        flat = (x + [0.0] * max(0, n * d - len(x)))[: n * d]
        toks = [flat[t * d:(t + 1) * d] for t in range(n)]
        P = self.trunk
        Q = [P._project(t, P.Wq) for t in toks]
        K = [P._project(t, P.Wk) for t in toks]
        V = [P._project(t, P.Wv) for t in toks]
        sc = 1.0 / math.sqrt(d)
        S = [[sum(Q[t][i] * K[s][i] for i in range(d)) * sc
              for s in range(n)] for t in range(n)]
        W = []
        for t in range(n):
            mx = max(S[t])
            ex = [math.exp(v - mx) for v in S[t]]
            ssum = sum(ex)
            W.append([v / ssum for v in ex])
        O = [[sum(W[t][s] * V[s][j] for s in range(n)) for j in range(d)]
             for t in range(n)]
        feats = [sum(O[t][j] for t in range(n)) / n for j in range(d)]
        logits = [sum(feats[f] * self.W_head[f][k] for f in range(self.num_feats))
                  + self.b_head[k] for k in range(self.out_dim)]
        cache = {"toks": toks, "Q": Q, "K": K, "V": V, "S": S, "W": W,
                 "O": O, "feats": feats}
        return logits, feats, cache

    def backward(self, cache, grad_logits, lr=0.05):
        n, d, F = self.n_tok, self.d_tok, self.num_feats
        P = self.trunk
        toks, Q, K, V, W = (cache["toks"], cache["Q"], cache["K"],
                            cache["V"], cache["W"])
        sc = 1.0 / math.sqrt(d)
        grad_feats = [sum(grad_logits[k] * self.W_head[f][k]
                          for k in range(self.out_dim)) for f in range(F)]
        grad_O = [[grad_feats[j] / n for j in range(d)] for _ in range(n)]
        # dL/dW (weights) and dL/dV
        grad_W = [[sum(grad_O[t][j] * V[s][j] for j in range(d))
                   for s in range(n)] for t in range(n)]
        grad_V = [[sum(grad_O[t][j] * W[t][s] for t in range(n))
                   for j in range(d)] for s in range(n)]
        # softmax backward -> dL/dS
        grad_S = []
        for t in range(n):
            row_dot = sum(W[t][r] * grad_W[t][r] for r in range(n))
            grad_S.append([W[t][s] * (grad_W[t][s] - row_dot)
                           for s in range(n)])
        # scores backward -> dL/dQ, dL/dK
        grad_Q = [[sc * sum(grad_S[t][s] * K[s][i] for s in range(n))
                   for i in range(d)] for t in range(n)]
        grad_K = [[sc * sum(grad_S[t][s] * Q[t][i] for t in range(n))
                   for i in range(d)] for s in range(n)]
        # projections backward -> trunk weights (SGD, same lr).
        # Q[t]/K[t]/V[t] derive from toks[t]; V[s] derives from toks[s].
        for t in range(n):
            tok = toks[t]
            for j in range(d):
                for i in range(d):
                    P.Wq[i][j] -= lr * grad_Q[t][j] * tok[i]
                    P.Wk[i][j] -= lr * grad_K[t][j] * tok[i]
        for s in range(n):
            tok = toks[s]
            for i in range(d):
                for j in range(d):
                    P.Wv[i][j] -= lr * grad_V[s][j] * tok[i]
        # classification head (identical to FixedAttentionModel)
        for k in range(self.out_dim):
            self.b_head[k] -= lr * grad_logits[k]
            for f in range(F):
                self.W_head[f][k] -= lr * cache["feats"][f] * grad_logits[k]

    def params(self):
        return self.trunk.params() + self.num_feats * self.out_dim + self.out_dim

    def flops(self):
        return self.trunk.flops() + 2 * self.num_feats * self.out_dim


def train_genuine(seed, train_data, test_data, label_map, n_epochs=30, lr=0.05):
    model = FixedGenuineAttentionModel(seed=seed)
    for _ in range(n_epochs):
        for s in train_data:
            x, y = s["x"], label_map[s["label"]]
            logits, _, cache = model.forward(x)
            mx = max(logits)
            ex = [math.exp(v - mx) for v in logits]
            probs = [v / sum(ex) for v in ex]
            grad = [probs[k] - (1.0 if k == y else 0.0) for k in range(3)]
            model.backward(cache, grad, lr=lr)
    correct, loss_sum = 0, 0.0
    for s in test_data:
        x, y = s["x"], label_map[s["label"]]
        logits, _, _ = model.forward(x)
        mx = max(logits)
        ex = [math.exp(v - mx) for v in logits]
        probs = [v / sum(ex) for v in ex]
        loss_sum += -math.log(max(probs[y], 1e-15))
        if max(range(3), key=lambda k: logits[k]) == y:
            correct += 1
    n = len(test_data)
    return {"test_accuracy": round(correct / n, 4),
            "test_loss": round(loss_sum / n, 4),
            "compute_flops": model.flops(),
            "parameters": model.params()}


def main():
    seeds = list(range(1, 11))
    out_override = None
    for a in sys.argv[1:]:
        if a.startswith("--out="):
            out_override = a.split("=", 1)[1]
    base = json.load(open(os.path.join(
        REPO, "experiments/results/nf01_fixed_10seed_corrected.json")))
    assert base["cost_model"] == COST_MODEL_VERSION
    label_map = {"dense": 0, "spatial": 1, "relational": 2}
    print("=== NX-ATTN-01 genuine vs legacy attention (seeds 1-10) ===")
    t0 = time.time()
    rows, failures = [], []
    for s in seeds:
        random.seed(s)
        train_modes = (["spatial", "relational", "dense"] * (100 // 3 + 1))[:100]
        test_modes = (["spatial", "relational", "dense"] * (50 // 3 + 1))[:50]
        random.shuffle(train_modes)
        random.shuffle(test_modes)
        train_data = [make_sample(m, seed=s * 1000 + i)
                      for i, m in enumerate(train_modes)]
        test_data = [make_sample(m, seed=s * 2000 + i)
                     for i, m in enumerate(test_modes)]
        st0 = time.time()
        try:
            g = train_genuine(s, train_data, test_data, label_map)
            g["status"] = "completed"
            g["runtime_s"] = round(time.time() - st0, 2)
            g["accuracy_per_kflop"] = round(
                g["test_accuracy"] / (g["compute_flops"] / 1000.0), 4)
        except Exception as e:
            g = {"status": f"failed: {type(e).__name__}: {e}"}
            failures.append({"seed": s, "error": str(e)})
        brow = next(r for r in base["rows"] if r["seed"] == s)
        rows.append({"seed": s, "genuine_softmax_attention": g,
                     "legacy_truncated_wq_projection": {
                         "test_accuracy": brow["architectures"]["attention"]["test_accuracy"],
                         "compute_flops": brow["architectures"]["attention"]["compute_flops"]},
                     "mlp": {"test_accuracy": brow["architectures"]["mlp"]["test_accuracy"]},
                     "cnn": {"test_accuracy": brow["architectures"]["cnn"]["test_accuracy"]}})
        print(f"seed {s}: genuine={g.get('test_accuracy')} "
              f"legacy={brow['architectures']['attention']['test_accuracy']} "
              f"mlp={brow['architectures']['mlp']['test_accuracy']} "
              f"cnn={brow['architectures']['cnn']['test_accuracy']}")
    ok = [r for r in rows if r["genuine_softmax_attention"].get("status") == "completed"]
    gacc = [r["genuine_softmax_attention"]["test_accuracy"] for r in ok]
    lacc = [r["legacy_truncated_wq_projection"]["test_accuracy"] for r in ok]
    diffs = [g - l for g, l in zip(gacc, lacc)]
    mean_d = sum(diffs) / len(diffs) if diffs else 0.0
    out = {
        "experiment": "NX-ATTN-01_genuine_vs_legacy_attention",
        "scope": ("NEW exploratory; NOT ladder; genuine trunk NEVER in "
                  "NF-01..NF-15; same data/splits/seeds/budget as NF-01 arms"),
        "hypothesis": ("genuine QKV softmax attention behaves differently from "
                       "the legacy truncated Wq projection under identical evaluation"),
        "config": {"seeds": seeds, "n_epochs": 30, "n_train": 100,
                   "n_test": 50, "lr": 0.05, "optimizer": "SGD",
                   "dataset": "evaluation.synthetic_dataset"},
        "cost_model": COST_MODEL_VERSION,
        "genuine_cost": {"flops": 5056 + 48, "params": 192 + 27,
                         "note": "trunk 5056 + 8->3 head 48"},
        "timestamp": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - t0, 2),
        "failures": failures,
        "paired_genuine_minus_legacy": {
            "per_seed": diffs,
            "mean_diff": round(mean_d, 4),
            "n_genuine_wins": sum(1 for d_ in diffs if d_ > 0),
            "n_ties": sum(1 for d_ in diffs if d_ == 0),
            "n_legacy_wins": sum(1 for d_ in diffs if d_ < 0),
        },
        "rows": rows,
        "limitations": ["single hyperparameter setting (no tuning either arm)",
                        "synthetic benchmark only", "n=10 seeds"],
    }
    path = out_override or "experiments/results/nx_attn_genuine_vs_legacy.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[DONE] wrote {path}; mean genuine-legacy diff = {round(mean_d, 4)}")


if __name__ == "__main__":
    main()
