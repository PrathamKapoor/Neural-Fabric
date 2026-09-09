"""NF-07 - Compute-aware routing sweep (REAL experiments, replaces status stub).

Research question: can entropy/compute penalties trade accuracy for executed
compute in a controlled way, and where is the Pareto knee?

Method: analytical soft-router training (same protocol/data as NF-02) with
loss = NLL + lam_ent * H(p) + lam_comp * weighted_flops / TOTAL_FLOPS.
Gradients derived in closed form through the softmax.

Grid: lam_ent in {0.0, 0.05, 0.2} x lam_comp in {0.0, 0.5, 2.0}, seeds [1,2,3].
Metrics per cell: test accuracy, mean entropy, collapse flag, executed soft
FLOPs, post-hoc top-1 executed FLOPs (same router, hard execution).
"""
import os
import sys
import math
import random
import json
import time

sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from fabric.router.router import Router
from evaluation.synthetic_dataset import make_sample

PRIM_NAMES = ["mlp", "cnn", "attention"]
PRIM_FLOPS = {"mlp": 17152, "cnn": 1152, "attention": 12288}
TOTAL_FLOPS = float(sum(PRIM_FLOPS.values()))
LABEL_MAP = {"dense": 0, "spatial": 1, "relational": 2}


def softmax(h):
    mx = max(h)
    ex = [math.exp(v - mx) for v in h]
    s = sum(ex)
    return [v / s for v in ex]


def train_cell(seed, lam_ent, lam_comp, n_epochs=30, n_train=100, n_test=50,
               lr=0.05):
    random.seed(seed)
    train_modes = (["spatial", "relational", "dense"] * (n_train // 3 + 1))[:n_train]
    test_modes = (["spatial", "relational", "dense"] * (n_test // 3 + 1))[:n_test]
    random.shuffle(train_modes)
    random.shuffle(test_modes)
    train_data = [make_sample(m, seed=seed * 1000 + i) for i, m in enumerate(train_modes)]
    test_data = [make_sample(m, seed=seed * 2000 + i) for i, m in enumerate(test_modes)]
    fvec = [PRIM_FLOPS[p] / TOTAL_FLOPS for p in PRIM_NAMES]

    router = Router(in_dim=64, n_prims=3, mode="soft", seed=seed)
    for _ in range(n_epochs):
        for s in train_data:
            x = s["x"]
            y = LABEL_MAP[s["label"]]
            h = router.encode(x)
            probs = softmax(h)
            # dNLL/dh + lam_ent*dH/dh + lam_comp*dC/dh, all via dp/dh chain rule
            for k in range(3):
                g = probs[k] - (1.0 if k == y else 0.0)  # NLL part
                logp = math.log(max(probs[k], 1e-15))
                # dH/dh_k = -sum_j (log p_j + 1) p_j (delta_jk - p_k)
                dh = -sum((math.log(max(probs[j], 1e-15)) + 1.0) * probs[j]
                          * ((1.0 if j == k else 0.0) - probs[k])
                          for j in range(3))
                # dC/dh_k with C = sum_j f_j p_j
                dc = sum(fvec[j] * probs[j] * ((1.0 if j == k else 0.0) - probs[k])
                         for j in range(3))
                grad = g + lam_ent * dh + lam_comp * dc
                router.b[k] -= lr * grad
                for i in range(min(len(x), router.in_dim)):
                    router.W[i][k] -= lr * grad * x[i]

    # evaluate
    correct = 0
    ents = []
    top1_flops = 0
    sels = []
    for s in test_data:
        x = s["x"]
        y = LABEL_MAP[s["label"]]
        probs = softmax(router.encode(x))
        pred = max(range(3), key=lambda j: probs[j])
        sels.append(pred)
        if pred == y:
            correct += 1
        ents.append(-sum(p * math.log(max(p, 1e-15)) for p in probs))
        top1_flops += PRIM_FLOPS[PRIM_NAMES[pred]]
    n = len(test_data)
    max_ratio = max(sels.count(j) for j in range(3)) / n
    return {
        "seed": seed, "lam_ent": lam_ent, "lam_comp": lam_comp,
        "test_accuracy": round(correct / n, 4),
        "mean_entropy": round(sum(ents) / n, 4),
        "max_primitive_ratio": round(max_ratio, 4),
        "collapse_detected": bool(max_ratio > 0.85),
        "executed_soft_flops": int(TOTAL_FLOPS),
        "top1_executed_flops": int(2 * 64 * 3 + top1_flops / n),
    }


def main():
    lam_ents = [0.0, 0.05, 0.2]
    lam_comps = [0.0, 0.5, 2.0]
    seeds = [1, 2, 3]
    print(f"=== NF-07 compute-aware sweep "
          f"({len(lam_ents)}x{len(lam_comps)} grid x {len(seeds)} seeds) ===")
    t0 = time.time()
    cells = []
    for le in lam_ents:
        for lc in lam_comps:
            accs, ents, flops, collapses = [], [], [], []
            for s in seeds:
                r = train_cell(s, le, lc)
                accs.append(r["test_accuracy"])
                ents.append(r["mean_entropy"])
                flops.append(r["top1_executed_flops"])
                collapses.append(r["collapse_detected"])
            cells.append({
                "lam_ent": le, "lam_comp": lc, "seeds": seeds,
                "mean_accuracy": round(sum(accs) / len(accs), 4),
                "mean_entropy": round(sum(ents) / len(ents), 4),
                "mean_top1_flops": round(sum(flops) / len(flops), 1),
                "any_collapse": any(collapses),
                "per_seed": [{"seed": s, "acc": a, "ent": e, "flops": f}
                             for s, a, e, f in zip(seeds, accs, ents, flops)],
            })
            print(f"lam_ent={le} lam_comp={lc} acc={cells[-1]['mean_accuracy']} "
                  f"ent={cells[-1]['mean_entropy']} top1flops={cells[-1]['mean_top1_flops']} "
                  f"collapse={cells[-1]['any_collapse']}")
    out = {"experiment": "nf07_sweep", "n_seeds": len(seeds),
           "elapsed_s": round(time.time() - t0, 2), "cells": cells}
    with open("experiments/results/nf07_sweep_matrix.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[DONE] NF-07 saved to experiments/results/nf07_sweep_matrix.json")


if __name__ == "__main__":
    main()
