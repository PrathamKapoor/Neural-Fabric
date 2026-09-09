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
from fabric.primitives.cost_model import (
    PRIM_FLOPS, TOTAL_FLOPS, PRIM_NAMES, ROUTER_FLOPS, COST_MODEL_VERSION,
)
from evaluation.synthetic_dataset import make_sample
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

    # evaluate (measurement only; no weight updates below this line)
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
    sel_counts = {PRIM_NAMES[j]: sels.count(j) for j in range(3)}
    sel_freqs = {p: round(c / n, 4) for p, c in sel_counts.items()}
    # Post-hoc training-objective measurement with FINAL weights (no updates):
    # mean NLL + penalty terms, i.e. the exact loss train_cell optimizes.
    tr_nll = tr_ent = tr_comp = 0.0
    for s in train_data:
        x = s["x"]
        y = LABEL_MAP[s["label"]]
        probs = softmax(router.encode(x))
        tr_nll += -math.log(max(probs[y], 1e-15))
        tr_ent += -sum(p * math.log(max(p, 1e-15)) for p in probs)
        tr_comp += sum(fvec[j] * probs[j] for j in range(3))
    m = len(train_data)
    tr_nll, tr_ent, tr_comp = tr_nll / m, tr_ent / m, tr_comp / m
    return {
        "seed": seed, "lam_ent": lam_ent, "lam_comp": lam_comp,
        "status": "completed",
        "test_accuracy": round(correct / n, 4),
        "mean_entropy": round(sum(ents) / n, 4),
        "max_primitive_ratio": round(max_ratio, 4),
        "collapse_detected": bool(max_ratio > 0.85),
        "test_selection_counts": sel_counts,
        "test_selection_frequencies": sel_freqs,
        "train_nll_final": round(tr_nll, 4),
        "train_entropy_final": round(tr_ent, 4),
        "train_compute_final": round(tr_comp, 4),
        "train_objective_final": round(tr_nll + lam_ent * tr_ent + lam_comp * tr_comp, 4),
        "executed_soft_flops": int(TOTAL_FLOPS),
        "top1_executed_flops": int(ROUTER_FLOPS + top1_flops / n),
    }


def _parse_argv():
    seeds = [1, 2, 3]
    out_override = None
    for a in sys.argv[1:]:
        if a.startswith("--seeds="):
            seeds = [int(t) for t in a.split("=", 1)[1].split(",") if t.strip()]
        elif a.startswith("--out="):
            out_override = a.split("=", 1)[1]
    return seeds, out_override


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def main():
    lam_ents = [0.0, 0.05, 0.2]
    lam_comps = [0.0, 0.5, 2.0]
    seeds, out_override = _parse_argv()
    print(f"=== NF-07 compute-aware sweep "
          f"({len(lam_ents)}x{len(lam_comps)} grid x {len(seeds)} seeds) ===")
    t0 = time.time()
    cells = []
    failures = []
    for le in lam_ents:
        for lc in lam_comps:
            rows = []
            for s in seeds:
                try:
                    r = train_cell(s, le, lc)
                except Exception as e:  # never silently discard a seed
                    r = {"seed": s, "lam_ent": le, "lam_comp": lc,
                         "status": f"failed: {type(e).__name__}: {e}"}
                    failures.append({"seed": s, "lam_ent": le,
                                     "lam_comp": lc, "error": str(e)})
                rows.append(r)
            ok = [r for r in rows if r.get("status") == "completed"]
            cells.append({
                "lam_ent": le, "lam_comp": lc, "seeds": seeds,
                "n_completed": len(ok), "n_failed": len(rows) - len(ok),
                "mean_accuracy": _mean([r["test_accuracy"] for r in ok]),
                "mean_entropy": _mean([r["mean_entropy"] for r in ok]),
                "mean_top1_flops": (round(sum(r["top1_executed_flops"] for r in ok)
                                          / len(ok), 1) if ok else None),
                "any_collapse": any(r["collapse_detected"] for r in ok),
                "per_seed": [{k: r.get(k) for k in (
                    "seed", "status", "test_accuracy", "mean_entropy",
                    "max_primitive_ratio", "collapse_detected",
                    "test_selection_counts", "train_nll_final",
                    "train_objective_final", "top1_executed_flops")}
                    for r in rows],
            })
            print(f"lam_ent={le} lam_comp={lc} acc={cells[-1]['mean_accuracy']} "
                  f"ent={cells[-1]['mean_entropy']} top1flops={cells[-1]['mean_top1_flops']} "
                  f"collapse={cells[-1]['any_collapse']} "
                  f"completed={cells[-1]['n_completed']}/{len(rows)}")
    import datetime
    corrected = "--corrected" in sys.argv
    out = {"experiment": "nf07_sweep", "n_seeds": len(seeds), "seeds": seeds,
           "hypothesis": "compute/entropy penalties trade accuracy for executed compute",
           "config": {"lam_ents": lam_ents, "lam_comps": lam_comps,
                      "n_epochs": 30, "n_train": 100, "n_test": 50, "lr": 0.05,
                      "dataset": "evaluation.synthetic_dataset (spatial/relational/dense)",
                      "split": "seed-shuffled train/test per seed; same protocol as NF-02"},
           "cost_model": COST_MODEL_VERSION, "prim_flops": dict(PRIM_FLOPS),
           "corrected_run": corrected,
           "timestamp": datetime.datetime.now().isoformat(),
           "elapsed_s": round(time.time() - t0, 2),
           "failures": failures, "cells": cells}
    if out_override:
        path = out_override
    else:
        path = ("experiments/results/nf07_sweep_matrix_corrected.json"
                if corrected else "experiments/results/nf07_sweep_matrix.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[DONE] NF-07 saved to {path}")


if __name__ == "__main__":
    main()
