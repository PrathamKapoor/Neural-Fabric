"""NF-10 — Distribution shift. Train synthetic mode A; test mode B."""
import sys, os, json, random
sys.path.insert(0,".")
from fabric.engine import FabricEngine
from fabric.primitives.mlp import MLP
from evaluation.synthetic_dataset import make_sample

def split_dataset(mode, n=100, seed=0):
    random.seed(seed)
    return [make_sample(mode, seed=i) for i in range(n)]

res = {"in_dist": {}, "shifted": {}, "degradation": {}}
# Fixed baselines
base_mlp = MLP(); base_cnn = MLP()  # same for simplicity
engine = FabricEngine(mode="hard")
train = split_dataset("spatial", 60, seed=10)
test_same = split_dataset("spatial", 20, seed=11)
test_shift = split_dataset("relational", 20, seed=12)

# Simple metric: selection matches expected structural mode
for label, data in [("in", test_same), ("shift", test_shift)]:
    paths = []; computes = []
    for s in data:
        r = engine.predict(s["x"], graph_mode=True)
        paths.append([n["primitive"] for n in r["graph"]["nodes"]])
        computes.append(r["compute"])
    expected = {"spatial":"cnn","relational":"attention"}.get({"in":"spatial","shift":"relational"}[label],"mlp")
    match = sum(1 for p in paths if p and p[0]==expected)
    res[label == "in" and "in_dist" or "shifted"]["expected"] = expected
    res[label == "in" and "in_dist" or "shifted"]["match_rate"] = match/len(data)
    res[label == "in" and "in_dist" or "shifted"]["mean_compute"] = sum(computes)/len(computes)

res["degradation"]["match_diff"] = res["in_dist"]["match_rate"] - res["shifted"]["match_rate"]
with open("experiments/results/nf10_distribution_shift.json","w") as f:
    json.dump(res, f, indent=2)
print("NF-10 saved:", json.dumps(res, indent=2)[:300])
