"""Recompute NF-01 compute-derived fields under the v2-corrected cost model.

Scientific basis (REPORTING-ONLY correction, no retraining needed):
- train_single_architecture() in experiments/nf01_fixed_baselines.py never
  references FLOPs: forward/backward/loss/accuracy dynamics are identical
  under v1 and v2. Only DERIVED reporting fields (compute_flops,
  primitive_flops, pareto_analysis, accuracy_per_kflop) change, and they are
  pure functions of the corrected primitive flops() values.
- Fixed-model corrected costs: MLP 17152 (unchanged), CNN 8256 + 192 head =
  8448, Attention-legacy 2048 + 48 head = 2096.
- NF-02 soft-routing executed/weighted compute comes from the genuinely
  rerun experiments/results/nf02_soft_training_corrected.json (5 seeds;
  accuracies verified bit-identical to the historical run).

Reads : experiments/results/nf01_fixed_baselines.json (HISTORICAL, untouched)
        experiments/results/nf02_soft_training_corrected.json
Writes: experiments/results/nf01_fixed_baselines_corrected_compute.json
"""
import copy
import json
import math
import os
import sys
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from fabric.primitives.cost_model import (
    COST_MODEL_VERSION, PRIM_FLOPS, mlp_flops, cnn_flops,
    legacy_attention_flops,
)

CORRECTED_MODEL_FLOPS = {
    "mlp": mlp_flops(),                       # 17152
    "cnn": cnn_flops() + 2 * 32 * 3,          # 8256 + 192 head = 8448
    "attention": legacy_attention_flops() + 2 * 8 * 3,  # 2048 + 48 head = 2096
}


def calc_stats(values):
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return {"mean": round(mean, 4), "std": round(std, 4),
            "min": round(min(values), 4), "max": round(max(values), 4), "n": n}


def main():
    with open(os.path.join(REPO, "experiments/results/nf01_fixed_baselines.json")) as f:
        hist = json.load(f)
    with open(os.path.join(REPO, "experiments/results/nf02_soft_training_corrected.json")) as f:
        nf02c = json.load(f)
    assert nf02c.get("cost_model") == COST_MODEL_VERSION

    out = copy.deepcopy(hist)
    changed_best = []
    for run in out["per_seed_results"]:
        for arch in ("mlp", "cnn", "attention"):
            a = run["architectures"][arch]
            a["compute_flops"] = CORRECTED_MODEL_FLOPS[arch]
            a["primitive_flops"] = PRIM_FLOPS[arch]
            a["compute"]["primitive_flops"] = PRIM_FLOPS[arch]
            a["compute"]["total_model_flops"] = CORRECTED_MODEL_FLOPS[arch]
            a["compute"]["executed_flops"] = CORRECTED_MODEL_FLOPS[arch]
        # best-fixed reselect under corrected tiebreak (acc, then -flops)
        archs = run["architectures"]
        new_best = max(archs.keys(),
                       key=lambda a: (archs[a]["test_accuracy"],
                                      -archs[a]["compute_flops"]))
        if new_best != run["best_baseline"]:
            changed_best.append((run["seed"], run["best_baseline"], new_best))
        run["best_baseline"] = new_best
        run["compute_flops"] = archs[new_best]["compute_flops"]
        run["compute"] = archs[new_best]["compute"]

    for arch in ("mlp", "cnn", "attention"):
        flops = [r["architectures"][arch]["compute_flops"] for r in out["per_seed_results"]]
        params = [r["architectures"][arch]["parameters"] for r in out["per_seed_results"]]
        out["summary"][arch]["compute_flops"] = calc_stats(flops)
        out["summary"][arch]["parameters"] = calc_stats(params)

    # Corrected NF-02 soft-routing compute (from genuine corrected rerun).
    nf02_runs = {r["seed"]: r for r in nf02c["per_seed_results"]}
    seeds = out["metadata"]["seeds"]
    nf02_exec = [r["compute"]["executed_flops"] for r in nf02c["per_seed_results"]]
    nf02_w = [r["compute"]["weighted_expected_flops"] for r in nf02c["per_seed_results"]]
    out["summary"]["nf02_soft_routing"]["executed_compute_flops"] = calc_stats(nf02_exec)
    out["summary"]["nf02_soft_routing"]["weighted_compute_flops"] = calc_stats(nf02_w)

    # Rebuild pareto candidates under corrected compute (accuracies untouched).
    cands = []
    for name, arch in (("Fixed CNN", "cnn"), ("Fixed Attention", "attention"),
                       ("Fixed MLP", "mlp")):
        acc = out["summary"][arch]["test_accuracy"]["mean"]
        fl = out["summary"][arch]["compute_flops"]["mean"]
        cands.append({"name": name, "architecture": arch, "mean_accuracy": acc,
                      "compute_flops": fl,
                      "parameters": out["summary"][arch]["parameters"]["mean"],
                      "accuracy_per_kflop": round(acc / (fl / 1000.0), 4)})
    nf02_acc = out["summary"]["nf02_soft_routing"]["test_accuracy"]["mean"]
    nf02_fl = out["summary"]["nf02_soft_routing"]["executed_compute_flops"]["mean"]
    cands.append({"name": "NF-02 Soft Routing (Executed)", "architecture": "soft_routing",
                  "mean_accuracy": nf02_acc, "compute_flops": nf02_fl,
                  "parameters": 384 + 8707 + 72 + 3072,
                  "accuracy_per_kflop": round(nf02_acc / (nf02_fl / 1000.0), 4)})
    for cand in cands:
        dominated = any(
            (o["mean_accuracy"] >= cand["mean_accuracy"] and o["compute_flops"] < cand["compute_flops"])
            or (o["mean_accuracy"] > cand["mean_accuracy"] and o["compute_flops"] <= cand["compute_flops"])
            for o in cands if o["name"] != cand["name"])
        cand["is_pareto_optimal"] = not dominated
    out["pareto_analysis"] = {
        "candidates": cands,
        "pareto_frontier": [c["name"] for c in cands if c["is_pareto_optimal"]],
        "pareto_optimal_architectures": [c for c in cands if c["is_pareto_optimal"]],
        "efficiency_winner": max(cands, key=lambda c: c["accuracy_per_kflop"])["name"],
        "accuracy_winner": max(cands, key=lambda c: c["mean_accuracy"])["name"],
    }
    out["provenance"] = {
        "derivation": "reporting-only recomputation; no retraining (training is cost-independent)",
        "historical_source": "experiments/results/nf01_fixed_baselines.json @ 0a4cbea",
        "nf02_corrected_source": "experiments/results/nf02_soft_training_corrected.json",
        "cost_model": COST_MODEL_VERSION,
        "prim_flops": dict(PRIM_FLOPS),
        "corrected_model_flops": dict(CORRECTED_MODEL_FLOPS),
        "best_baseline_changed_seeds": changed_best,
        "timestamp": datetime.now().isoformat(),
    }
    path = os.path.join(REPO, "experiments/results/nf01_fixed_baselines_corrected_compute.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[DONE] wrote {path}")
    print("OLD pareto:", hist["pareto_analysis"]["pareto_frontier"],
          "efficiency:", hist["pareto_analysis"]["efficiency_winner"])
    print("NEW pareto:", out["pareto_analysis"]["pareto_frontier"],
          "efficiency:", out["pareto_analysis"]["efficiency_winner"])
    print("best-baseline changes:", changed_best or "none")


if __name__ == "__main__":
    main()
