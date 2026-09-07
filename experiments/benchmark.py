"""Quick benchmark harness — runs fabric vs fixed baselines, records to experiments/results/"""
import sys, os, json, time
sys.path.insert(0, ".")
from fabric.engine import FabricEngine
from fabric.primitives.mlp import MLP
from evaluation.synthetic_dataset import dataset

engine_soft = FabricEngine(mode="soft")
engine_hard = FabricEngine(mode="hard")
engine_top = FabricEngine(mode="topk")
results = {"runs": []}

# Fixed baseline
base = MLP()

for i, s in enumerate(dataset(30)):
    x = s["x"]
    # Fixed
    t0 = time.time(); base_res = base.forward(x); t_base = time.time()-t0
    # Fabric hard
    t0 = time.time(); res = engine_hard.predict(x, graph_mode=True); t_f = time.time()-t0
    results["runs"].append({
        "id": s["id"], "label": s["label"], "mode": "hard",
        "base_flops": base.flops(), "fabric_flops": res["compute"],
        "routing": res["routing"], "graph_nodes": res["graph"]["nodes"],
        "latency_ms": round(t_f*1000,2)
    })

os.makedirs("experiments/results", exist_ok=True)
with open("experiments/results/nf01_benchmark.json","w") as f:
    json.dump(results, f, indent=2)
print("Benchmark saved to experiments/results/nf01_benchmark.json — samples:", len(results["runs"]))
# Print summary
from collections import Counter
c = Counter([r["label"] for r in results["runs"]])
print("Label distribution:", dict(c))
# Routing specialization check
for mode in ["spatial","relational","dense"]:
    nodes = [r["graph_nodes"] for r in results["runs"] if r["label"]==mode]
    prims = [n[0]["primitive"] if n else "none" for n in nodes]
    print(mode, "-> first primitive:", Counter(prims).most_common(3))
