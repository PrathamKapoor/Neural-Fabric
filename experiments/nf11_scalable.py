"""Scalable benchmark — 100 samples, real primitives."""
import json, os, sys, time
sys.path.insert(0, ".")
from fabric.primitives.mlp import MLP
from fabric.primitives.cnn import CNN
from fabric.primitives.attention import Attention
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
with open("experiments/results/nf11_scalable_100.json", "w") as f:
    json.dump(res, f, indent=2)
print("Scalable 100 saved; samples:", len(res["scalable_100"]))
