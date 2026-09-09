"""NF-04 — Top-k ablation across k=1,2,3."""
import sys, os, json
from datetime import datetime
sys.path.insert(0, ".")
from fabric.engine import FabricEngine
from fabric.primitives.cost_model import COST_MODEL_VERSION, PRIM_FLOPS
from evaluation.synthetic_dataset import dataset
res = {}
for k in [1, 2, 3]:
    eng = FabricEngine(mode="topk")
    # Manually override selected count to simulate k via graph construction
    # Here we just use topk mode directly (selects top-2 by default from router)
    compute_vals = []
    for s in dataset(10):
        r = eng.predict(s["x"], graph_mode=True, k=k)
        compute_vals.append(r["compute"])
    mean_val = sum(compute_vals) / len(compute_vals)
    res[f"topk_k{k}"] = {
        "mean_compute": mean_val,
        "flops": mean_val,
        "samples": len(compute_vals)
    }
corrected = "--corrected" in sys.argv
res["cost_model"] = COST_MODEL_VERSION
res["corrected_run"] = corrected
res["prim_flops"] = dict(PRIM_FLOPS)
res["timestamp"] = datetime.now().isoformat()
path = ("experiments/results/nf04_topk_ablation_corrected.json"
        if corrected else "experiments/results/nf04_topk_ablation.json")
with open(path, "w") as f:
    json.dump(res, f, indent=2)
print(f"NF-04 ablation saved to {path}", res)
