"""Scalable benchmark — 100 samples, real torch primitives."""
import torch, json, os, sys, time
sys.path.insert(0,".")
from fabric.primitives.torch_primitives import MLP, CNN, Attention
from evaluation.synthetic_dataset import dataset
res={"scalable_100":[]}
for s in dataset(100)[:100]:
    x=torch.tensor(s["x"],dtype=torch.float32).unsqueeze(0)
    m=MLP(); c=CNN(); a=Attention()
    res["scalable_100"].append({"label":s["label"],"mlp_flops":m(x).norm().item(),"cnn_flops":c(x).norm().item(),"attn_flops":a(x).norm().item()})
with open("experiments/results/nf11_scalable_100.json","w") as f:
    json.dump(res,f,indent=2)
print("Scalable 100 saved; samples:", len(res["scalable_100"]))
