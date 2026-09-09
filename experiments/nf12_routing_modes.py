"""NF-12 - Trained routing-mode comparison (soft / top-k execution / Gumbel-hard).

Research question: under IDENTICAL training data/protocol, how do execution
modes compare on accuracy vs executed compute, and can hard routing be
trained (Gumbel-softmax) without collapse?

Arms (seeds [1,2,3,4,5], same splits/lr/epochs as NF-02):
- soft: one identical soft training per seed. Metrics double as an
  in-session replication check against recorded NF-02 values.
- top-k execution: post-hoc evaluation of the SAME soft routers with top-k
  primitive execution (accuracy argmax-identical by construction; compute
  varies with k). Isolates the efficiency question honestly.
- gumbel-hard: Gumbel-softmax relaxation (tau annealed 1.0 -> 0.1),
  NLL on relaxed probs, evaluated with hard argmax execution.

Paired comparison vs NF-01 best-fixed accuracies uses recorded artifacts on
overlapping seeds (1-5); no NF-01 rerun needed.
"""
import os
import sys
import math
import random
import json
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from fabric.router.router import Router
from fabric.primitives.cost_model import (
    PRIM_FLOPS, PRIM_NAMES, ROUTER_FLOPS, COST_MODEL_VERSION,
)
from evaluation.synthetic_dataset import make_sample

LABEL_MAP = {"dense": 0, "spatial": 1, "relational": 2}


def softmax(h):
    mx = max(h)
    ex = [math.exp(v - mx) for v in h]
    s = sum(ex)
    return [v / s for v in ex]


def gumbel_sample(rng):
    u = max(rng.random(), 1e-12)
    return -math.log(-math.log(u))


def calc_stats(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(values), 4), "max": round(max(values), 4),
            "n": n}


def paired_t_pvalue(a, b):
    """Paired t-test two-tailed p-value (own implementation, no scipy)."""
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var <= 1e-12:
        return 0.0 if mean != 0 else 1.0
    import math as _m
    t = mean / (_m.sqrt(var) / _m.sqrt(n))
    df = n - 1
    t = abs(t)
    log_c = _m.lgamma((df + 1.0) / 2.0) - 0.5 * _m.log(df * _m.pi) - _m.lgamma(df / 2.0)
    c = _m.exp(log_c)
    steps, h = 1000, t / 1000
    s = sum((1 if i in (0, 1000) else (4 if i % 2 else 2))
            * (1.0 + ((i * h) ** 2) / df) ** (-(df + 1.0) / 2.0)
            for i in range(1001))
    return round(max(0.0, min(1.0, 2.0 * (0.5 - c * s * (h / 3.0)))), 6)


def train_soft_identical(seed, n_epochs=30, n_train=100, lr=0.05):
    """The exact NF-02 soft-training loop, returning the trained router."""
    random.seed(seed)
    train_modes = (["spatial", "relational", "dense"] * (n_train // 3 + 1))[:n_train]
    random.shuffle(train_modes)
    train_data = [make_sample(m, seed=seed * 1000 + i) for i, m in enumerate(train_modes)]
    router = Router(in_dim=64, n_prims=3, mode="soft", seed=seed)
    for _ in range(n_epochs):
        for s in train_data:
            x = s["x"]
            y = LABEL_MAP[s["label"]]
            h = router.encode(x)
            probs = softmax(h)
            for j in range(3):
                grad_h = probs[j] - (1.0 if j == y else 0.0)
                router.b[j] -= lr * grad_h
                for i in range(min(len(x), router.in_dim)):
                    router.W[i][j] -= lr * grad_h * x[i]
    return router


def train_gumbel_hard(seed, n_epochs=30, n_train=100, n_test=50, lr=0.05,
                      tau_start=1.0, tau_end=0.1):
    rng = random.Random(seed * 7919 + 13)
    random.seed(seed)
    train_modes = (["spatial", "relational", "dense"] * (n_train // 3 + 1))[:n_train]
    test_modes = (["spatial", "relational", "dense"] * (n_test // 3 + 1))[:n_test]
    random.shuffle(train_modes)
    random.shuffle(test_modes)
    train_data = [make_sample(m, seed=seed * 1000 + i) for i, m in enumerate(train_modes)]
    test_data = [make_sample(m, seed=seed * 2000 + i) for i, m in enumerate(test_modes)]
    router = Router(in_dim=64, n_prims=3, mode="hard", seed=seed)
    for epoch in range(n_epochs):
        tau = tau_start + (tau_end - tau_start) * epoch / max(1, n_epochs - 1)
        for s in train_data:
            x = s["x"]
            y = LABEL_MAP[s["label"]]
            h = router.encode(x)
            g = [gumbel_sample(rng) for _ in range(3)]
            relaxed = softmax([(h[j] + g[j]) / tau for j in range(3)])
            for k in range(3):
                gg = (relaxed[k] - (1.0 if k == y else 0.0)) / tau
                router.b[k] -= lr * gg
                for i in range(min(len(x), router.in_dim)):
                    router.W[i][k] -= lr * gg * x[i]
    correct, ents, sels, flops = 0, [], [], 0
    for s in test_data:
        x = s["x"]
        y = LABEL_MAP[s["label"]]
        h = router.encode(x)
        pred = max(range(3), key=lambda j: h[j])
        sels.append(pred)
        if pred == y:
            correct += 1
        probs = softmax(h)
        ents.append(-sum(p * math.log(max(p, 1e-15)) for p in probs))
        flops += PRIM_FLOPS[PRIM_NAMES[pred]]
    n = len(test_data)
    max_ratio = max(sels.count(j) for j in range(3)) / n
    return {
        "seed": seed, "test_accuracy": round(correct / n, 4),
        "mean_entropy": round(sum(ents) / n, 4),
        "max_primitive_ratio": round(max_ratio, 4),
        "collapse_detected": bool(max_ratio > 0.85),
        "executed_flops": int(ROUTER_FLOPS + flops / n),
    }


def main():
    seeds = [1, 2, 3, 4, 5]
    print("=== NF-12 routing-mode comparison ===")
    t0 = time.time()
    try:
        rec = json.load(open("experiments/results/nf02_soft_training.json"))
        recorded = {row["seed"]: (row["test_accuracy"], row["test_entropy"])
                    for row in rec.get("per_seed_results", [])}
        fixed = json.load(open("experiments/results/nf01_fixed_baselines.json"))
        best_fixed = {}
        for row in fixed.get("per_seed_results", []):
            best_fixed[row["seed"]] = max(row["architectures"][a]["test_accuracy"]
                                          for a in ("mlp", "cnn", "attention"))
    except OSError:
        recorded, best_fixed = {}, {}
    soft_rows, k_rows = [], {1: [], 2: [], 3: []}
    for s in seeds:
        # Mirror NF-02's exact RNG sequence (train shuffle precedes test
        # shuffle) so the test set - and the replication check - is exact.
        # Router init and samples are self-seeded; only shuffle order matters.
        random.seed(s)
        _discard = (["spatial", "relational", "dense"] * (100 // 3 + 1))[:100]
        random.shuffle(_discard)
        test_modes = (["spatial", "relational", "dense"] * (50 // 3 + 1))[:50]
        random.shuffle(test_modes)
        test_data = [make_sample(m, seed=s * 2000 + i) for i, m in enumerate(test_modes)]
        router = train_soft_identical(s)
        correct, ents, sels = 0, [], []
        k_flops = {1: 0, 2: 0, 3: 0}
        for t in test_data:
            x = t["x"]
            y = LABEL_MAP[t["label"]]
            probs = softmax(router.encode(x))
            order = sorted(range(3), key=lambda j: probs[j], reverse=True)
            sels.append(order[0])
            if order[0] == y:
                correct += 1
            ents.append(-sum(p * math.log(max(p, 1e-15)) for p in probs))
            for k in (1, 2, 3):
                k_flops[k] += sum(PRIM_FLOPS[PRIM_NAMES[j]] for j in order[:k])
        n = len(test_data)
        max_ratio = max(sels.count(j) for j in range(3)) / n
        row = {"seed": s, "test_accuracy": round(correct / n, 4),
               "test_entropy": round(sum(ents) / n, 4),
               "max_primitive_ratio": round(max_ratio, 4),
               "collapse_detected": bool(max_ratio > 0.85)}
        if s in recorded:
            row["matches_recorded_nf02"] = (
                abs(row["test_accuracy"] - recorded[s][0]) < 1e-9
                and abs(row["test_entropy"] - recorded[s][1]) < 1e-4)
        soft_rows.append(row)
        for k in (1, 2, 3):
            k_rows[k].append({"seed": s, "test_accuracy": row["test_accuracy"],
                              "executed_flops": int(ROUTER_FLOPS + k_flops[k] / n)})
    hard_rows = [train_gumbel_hard(s) for s in seeds]
    for r in hard_rows:
        print(f"gumbel-hard seed {r['seed']} acc={r['test_accuracy']} "
              f"ent={r['mean_entropy']} flops={r['executed_flops']} "
              f"collapse={r['collapse_detected']}")
    # paired comparison vs best fixed baseline on overlapping seeds
    overlap = [s for s in seeds if s in best_fixed]
    paired = None
    if overlap:
        soft_acc = [next(r["test_accuracy"] for r in soft_rows if r["seed"] == s)
                    for s in overlap]
        hard_acc = [next(r["test_accuracy"] for r in hard_rows if r["seed"] == s)
                    for s in overlap]
        base_acc = [best_fixed[s] for s in overlap]
        paired = {
            "seeds": overlap,
            "soft_vs_best_fixed_p": paired_t_pvalue(soft_acc, base_acc),
            "hard_vs_best_fixed_p": paired_t_pvalue(hard_acc, base_acc),
            "mean_soft": round(sum(soft_acc) / len(soft_acc), 4),
            "mean_hard": round(sum(hard_acc) / len(hard_acc), 4),
            "mean_best_fixed": round(sum(base_acc) / len(base_acc), 4),
        }
        print("paired:", paired)
    import datetime
    corrected = "--corrected" in sys.argv
    out = {
        "experiment": "nf12_routing_modes",
        "seeds": seeds,
        "cost_model": COST_MODEL_VERSION, "prim_flops": dict(PRIM_FLOPS),
        "corrected_run": corrected,
        "timestamp": datetime.datetime.now().isoformat(),
        "elapsed_s": round(time.time() - t0, 2),
        "soft_reference": {
            "note": "identical soft training; replication flags per seed",
            "accuracy": calc_stats([r["test_accuracy"] for r in soft_rows]),
            "entropy": calc_stats([r["test_entropy"] for r in soft_rows]),
            "rows": soft_rows},
        "topk_execution": {
            str(k): {"accuracy": calc_stats([r["test_accuracy"] for r in v]),
                     "executed_flops": calc_stats([r["executed_flops"] for r in v]),
                     "rows": v} for k, v in k_rows.items()},
        "gumbel_hard": {
            "accuracy": calc_stats([r["test_accuracy"] for r in hard_rows]),
            "entropy": calc_stats([r["mean_entropy"] for r in hard_rows]),
            "executed_flops": calc_stats([r["executed_flops"] for r in hard_rows]),
            "any_collapse": any(r["collapse_detected"] for r in hard_rows),
            "rows": hard_rows},
        "paired_vs_best_fixed": paired,
    }
    path = ("experiments/results/nf12_routing_modes_corrected.json"
            if corrected else "experiments/results/nf12_routing_modes.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[DONE] NF-12 saved to {path}")


if __name__ == "__main__":
    main()
