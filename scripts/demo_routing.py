"""Canonical 60-second Neural Fabric demo (real behavior, real numbers).

Shows, for one input per synthetic mode: router probabilities, selected
primitive, executed FLOPs, prediction, and measured savings vs the fixed
MLP baseline. No mocks: every number comes from fabric/ or the recorded
v2-corrected artifacts named in the output.

Run: .venv\\Scripts\\python.exe scripts\\demo_routing.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fabric.engine import FabricEngine
from evaluation.synthetic_dataset import make_sample

NAMES = ["mlp", "cnn", "attention"]
MODES = ("spatial", "relational", "dense")
EVIDENCE = {
    "top1": "experiments/results/nf12_routing_modes_corrected.json",
    "tradeoff": "experiments/results/nf07_10seed_statistics.json",
    "baselines": "experiments/results/nf01_fixed_10seed_corrected.json",
}


def demo_case(mode, seed=0):
    """Run one routed prediction; return a JSON-serializable record."""
    s = make_sample(mode, seed=seed)
    x = s["x"]
    eng = FabricEngine(mode="topk", k=1)
    res = eng.predict(list(x), graph_mode=True)
    bench = eng.bench(list(x), k=1)
    routing = res["routing"]
    pred = res["prediction"]
    return {
        "mode": mode,
        "expected_label": s["label"],
        "route_probs": [round(float(p), 4) for p in routing["probs"]],
        "selected": [NAMES[s] for s in routing["selected"]],
        "executed_flops": res["compute"],
        "prediction": pred,
        # Only a 3-dim head output is a class vector; primitive feature
        # maps (e.g. 32-dim CNN) have no class index — report None there
        # rather than an invented "predicted class".
        "predicted_class": (int(max(range(len(pred)), key=pred.__getitem__))
                            if len(pred) == 3 else None),
        "baseline_mlp_flops": bench["baseline_compute"],
        "savings_pct_vs_mlp": bench["savings_pct"],
        "accuracy_note": bench["accuracy_status"],
    }


def main():
    cases = [demo_case(m, seed=i) for i, m in enumerate(MODES)]
    print("NEURAL FABRIC 60-SECOND DEMO (live engine, top-1 routing)")
    for c in cases:
        print("mode=%s route=%s selected=%s flops=%d vs mlp=%d savings=%s%% pred_class=%s" % (
            c["mode"], c["route_probs"], c["selected"], c["executed_flops"],
            c["baseline_mlp_flops"], c["savings_pct_vs_mlp"],
            c["predicted_class"]))
    print("evidence: " + json.dumps(EVIDENCE))
    print("caveat: demo uses an untrained router; measured accuracy/FLOP "
          "tradeoffs live in the evidence artifacts above.")
    return cases


if __name__ == "__main__":
    main()
