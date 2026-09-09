"""NF-13 - Routing specialization: P(selected primitive | input mode).

Research question: does the trained router exhibit interpretable
specialization (spatial->CNN, relational->attention, dense->MLP), or does
it route all modes to one primitive (collapse)?

Method: identical soft training per seed (same protocol as NF-02), then
per-mode selection rates + mutual information between input mode and
selected primitive (nats; max ln 3 ~= 1.099 for perfect specialization).
Seeds [1,2,3,4,5].
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
from evaluation.synthetic_dataset import make_sample

PRIM_NAMES = ["mlp", "cnn", "attention"]
MODES = ["dense", "spatial", "relational"]
LABEL_MAP = {"dense": 0, "spatial": 1, "relational": 2}


def softmax(h):
    mx = max(h)
    ex = [math.exp(v - mx) for v in h]
    s = sum(ex)
    return [v / s for v in ex]


def train_soft_identical(seed, n_epochs=30, n_train=100, lr=0.05):
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


def main():
    seeds = [1, 2, 3, 4, 5]
    print("=== NF-13 routing specialization ===")
    t0 = time.time()
    rows = []
    for s in seeds:
        random.seed(s)
        _discard = (["spatial", "relational", "dense"] * (100 // 3 + 1))[:100]
        random.shuffle(_discard)
        # stratified test set: 30 samples per mode for stable conditionals
        test_data = []
        for mi, m in enumerate(MODES):
            test_data += [make_sample(m, seed=s * 3000 + mi * 100 + i)
                          for i in range(30)]
        router = train_soft_identical(s)
        cond = {m: {p: 0 for p in PRIM_NAMES} for m in MODES}
        for t in test_data:
            probs = softmax(router.encode(t["x"]))
            pred = PRIM_NAMES[max(range(3), key=lambda j: probs[j])]
            cond[t["label"]][pred] += 1
        for m in MODES:
            tot = sum(cond[m].values())
            cond[m] = {p: round(c / tot, 4) for p, c in cond[m].items()}
        # mutual information I(mode; selected), nats
        mi = 0.0
        for m in MODES:
            for pi, p in enumerate(PRIM_NAMES):
                joint = cond[m][p] / 3.0
                marg_sel = sum(cond[mm][p] for mm in MODES) / 3.0
                if joint > 0 and marg_sel > 0:
                    mi += joint * math.log(joint / ((1.0 / 3.0) * marg_sel))
        # diagonal mass (correct-primitive rate) as specialization score
        expected = {"dense": "mlp", "spatial": "cnn", "relational": "attention"}
        diag = round(sum(cond[m][expected[m]] for m in MODES) / 3.0, 4)
        rows.append({"seed": s, "conditional": cond,
                     "mutual_information_nats": round(mi, 4),
                     "correct_primitive_rate": diag})
        print(f"seed {s} MI={mi:.4f} correct-prim={diag} cond={cond}")
    mis = [r["mutual_information_nats"] for r in rows]
    diags = [r["correct_primitive_rate"] for r in rows]
    out = {"experiment": "nf13_specialization", "seeds": seeds,
           "max_mi_nats": round(math.log(3), 4),
           "mean_mi_nats": round(sum(mis) / len(mis), 4),
           "mean_correct_primitive_rate": round(sum(diags) / len(diags), 4),
           "elapsed_s": round(time.time() - t0, 2), "rows": rows}
    with open("experiments/results/nf13_specialization.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[DONE] NF-13 saved; mean MI=%.4f mean correct-prim=%.4f"
          % (out["mean_mi_nats"], out["mean_correct_primitive_rate"]))


if __name__ == "__main__":
    main()
