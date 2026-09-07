"""NF-09 — Robustness under noise / perturbation."""
import sys, os, json, random
sys.path.insert(0,".")
from fabric.engine import FabricEngine
res = {"noise_perturbation": []}
engine = FabricEngine(mode="hard")
for seed in range(5):
    random.seed(seed); x = [random.random() for _ in range(64)]
    clean = engine.predict(x, graph_mode=True)
    noisy = engine.predict([v+0.05 for v in x], graph_mode=True)
    res["noise_perturbation"].append({
        "seed": seed, "clean_path": clean["graph"]["nodes"],
        "noisy_path": noisy["graph"]["nodes"],
        "path_changed": clean["graph"]["nodes"]!=noisy["graph"]["nodes"]
    })
with open("experiments/results/nf09_robustness.json","w") as f:
    json.dump(res, f, indent=2)
print("NF-09 saved; path changes:", sum(1 for r in res["noise_perturbation"] if r["path_changed"]))
