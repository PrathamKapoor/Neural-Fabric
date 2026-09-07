"""NF-03 — Hard routing vs fixed baselines. Real comparison using executed FLOPs."""
import sys, os, json
sys.path.insert(0,".")
from fabric.engine import FabricEngine
from fabric.primitives.mlp import MLP
from evaluation.synthetic_dataset import dataset

results = {"hard_vs_fixed": []}
engine = FabricEngine(mode="hard")
base = MLP()
for s in dataset(30):
    x = s["x"]
    r = engine.predict(x, graph_mode=True)
    results["hard_vs_fixed"].append({
        "label": s["label"], "fabric_flops": r["compute"],
        "base_flops": base.flops(),
        "selected": r["graph"]["nodes"][0]["primitive"] if r["graph"]["nodes"] else None,
        "path": r["graph"]["nodes"]
    })
with open("experiments/results/nf03_hard_comparison.json","w") as f:
    json.dump(results, f, indent=2)
print("NF-03 saved; samples:", len(results["hard_vs_fixed"]))
# Summary stats
from collections import Counter
sel = Counter([r["selected"] for r in results["hard_vs_fixed"]])
print("Selection frequencies:", dict(sel))
