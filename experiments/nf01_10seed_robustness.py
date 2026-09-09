"""NF-01 robustness extension: fixed baselines on seeds 1-10, corrected costs.

Same hyperparameters / data protocol / training code as NF-01
(n_epochs=30, n_train=100, n_test=50, lr=0.05; train_single_architecture):
different seeds only. No tuning.

Architectures: mlp, cnn, attention (HISTORICAL identity "attention"
preserved for provenance; mechanistically it is the legacy truncated Wq
linear projection — see the `mechanism_note` field and cost_model.py).

Writes: experiments/results/nf01_fixed_10seed_corrected.json
Supports: --out=PATH override (used by the job driver if needed).
"""
import json
import math
import os
import sys
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))

from fabric.primitives.cost_model import COST_MODEL_VERSION, PRIM_FLOPS
from evaluation.synthetic_dataset import make_sample
from nf01_fixed_baselines import train_single_architecture, calc_stats

MECHANISM_NOTE = {
    "mlp": "dense ReLU MLP (in=64, hidden=128, out=3)",
    "cnn": "2D conv (k=3) + quadrant pooling + linear head",
    "attention": ("HISTORICAL label 'attention'; mechanistically the LEGACY "
                  "TRUNCATED Wq LINEAR PROJECTION (single Wq matvec, dead "
                  "Wk/Wv, no QKT/softmax). Do not interpret as genuine attention."),
}


def run_seed(seed, n_epochs=30, n_train=100, n_test=50, lr=0.05):
    import random
    random.seed(seed)
    train_modes = (["spatial", "relational", "dense"] * (n_train // 3 + 1))[:n_train]
    test_modes = (["spatial", "relational", "dense"] * (n_test // 3 + 1))[:n_test]
    random.shuffle(train_modes)
    random.shuffle(test_modes)
    train_data = [make_sample(m, seed=seed * 1000 + i) for i, m in enumerate(train_modes)]
    test_data = [make_sample(m, seed=seed * 2000 + i) for i, m in enumerate(test_modes)]
    label_map = {"dense": 0, "spatial": 1, "relational": 2}
    row = {"seed": seed, "status": "completed", "architectures": {}}
    for arch in ("mlp", "cnn", "attention"):
        t0 = time.time()
        try:
            res = train_single_architecture(arch, seed, train_data, test_data,
                                            label_map, n_epochs=n_epochs, lr=lr)
            row["architectures"][arch] = {
                "status": "completed",
                "test_accuracy": res["test_accuracy"],
                "test_loss": res["test_loss"],
                "train_loss_final": res["train_loss_final"],
                "compute_flops": res["compute_flops"],
                "primitive_flops": res["primitive_flops"],
                "parameters": res["parameters"],
                "accuracy_per_kflop": round(res["test_accuracy"]
                                            / (res["compute_flops"] / 1000.0), 4),
                "runtime_s": round(time.time() - t0, 2),
            }
        except Exception as e:  # never silently discard
            row["architectures"][arch] = {
                "status": f"failed: {type(e).__name__}: {e}"}
    return row


def main():
    seeds = list(range(1, 11))
    out_override = None
    for a in sys.argv[1:]:
        if a.startswith("--out="):
            out_override = a.split("=", 1)[1]
    print("=== NF-01 10-seed robustness (corrected costs) ===")
    t0 = time.time()
    rows, failures = [], []
    for s in seeds:
        row = run_seed(s)
        rows.append(row)
        for arch, r in row["architectures"].items():
            if r.get("status") != "completed":
                failures.append({"seed": s, "arch": arch, "error": r["status"]})
        accs = {a: row["architectures"][a].get("test_accuracy")
                for a in ("mlp", "cnn", "attention")}
        print(f"seed {s}: " + " ".join(f"{a}={accs[a]}" for a in accs))
    summary = {}
    for arch in ("mlp", "cnn", "attention"):
        accs = [r["architectures"][arch]["test_accuracy"] for r in rows
                if r["architectures"][arch].get("status") == "completed"]
        apk = [r["architectures"][arch]["accuracy_per_kflop"] for r in rows
               if r["architectures"][arch].get("status") == "completed"]
        fl = list({r["architectures"][arch]["compute_flops"] for r in rows
                   if r["architectures"][arch].get("status") == "completed"})
        summary[arch] = {
            "test_accuracy": calc_stats(accs),
            "accuracy_per_kflop": calc_stats(apk),
            "compute_flops_values": fl,
            "parameters": rows[0]["architectures"][arch].get("parameters"),
            "mechanism_note": MECHANISM_NOTE[arch],
        }
    # Pareto membership on (mean accuracy, compute): higher acc, lower flops.
    cands = [(a, summary[a]["test_accuracy"]["mean"],
              summary[a]["compute_flops_values"][0]) for a in summary]
    frontier = []
    for a, acc, fl in cands:
        dominated = any((oa >= acc and ofl < fl) or (oa > acc and ofl <= fl)
                        for oa_, oa, ofl in cands if oa_ != a)
        if not dominated:
            frontier.append(a)
    out = {
        "experiment": "nf01_fixed_10seed_robustness",
        "hypothesis": ("corrected efficiency ordering (legacy-projection cheapest) "
                       "is stable across seeds 1-10"),
        "config": {"seeds": seeds, "n_epochs": 30, "n_train": 100,
                   "n_test": 50, "lr": 0.05,
                   "dataset": "evaluation.synthetic_dataset",
                   "trainer": "train_single_architecture (identical to NF-01)"},
        "cost_model": COST_MODEL_VERSION, "prim_flops": dict(PRIM_FLOPS),
        "timestamp": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - t0, 2),
        "failures": failures,
        "pareto_frontier_mean": sorted(frontier),
        "efficiency_winner_mean": max(summary, key=lambda a: summary[a][
            "accuracy_per_kflop"]["mean"]),
        "summary": summary,
        "rows": rows,
        "limitations": ["synthetic benchmark only", "n=10 seeds",
                        "mean-frontier; per-seed frontiers may vary"],
    }
    path = out_override or "experiments/results/nf01_fixed_10seed_corrected.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[DONE] wrote {path}; frontier={sorted(frontier)} "
          f"winner={out['efficiency_winner_mean']}")


if __name__ == "__main__":
    main()
