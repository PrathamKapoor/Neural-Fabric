"""Scalable benchmark — 100 samples, real primitives."""
import json, os, sys, time
from datetime import datetime
sys.path.insert(0, ".")
from fabric.primitives.mlp import MLP
from fabric.primitives.cnn import CNN
from fabric.primitives.attention import Attention
from fabric.primitives.cost_model import COST_MODEL_VERSION
from evaluation.synthetic_dataset import dataset
res = {"scalable_100": []}
for s in dataset(100)[:100]:
    x = s["x"]
    m = MLP(); c = CNN(); a = Attention()
    m.forward(x); c.forward(x); a.forward(x)
    res["scalable_100"].append({
        "label": s["label"],
        "mlp_flops": m.flops(),
        "cnn_flops": c.flops(),
        "attn_flops": a.flops()
    })
res["cost_model"] = COST_MODEL_VERSION
res["timestamp"] = datetime.now().isoformat()
corrected = "--corrected" in sys.argv
path = ("experiments/results/nf11_scalable_100_corrected.json"
        if corrected else "experiments/results/nf11_scalable_100.json")
with open(path, "w") as f:
    json.dump(res, f, indent=2)
print(f"Scalable 100 saved to {path}; samples:", len(res["scalable_100"]))
