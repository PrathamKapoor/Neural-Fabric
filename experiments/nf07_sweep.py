"""NF-07 — Compute-aware objective sweep (automated matrix)."""
import sys, os, json, random
sys.path.insert(0,".")
from experiments.nf02_torch_train import train_epoch  # reuse pattern conceptually
# Matrix: compute penalties 0, 0.05, 0.1; entropy 0, 0.1; seeds 1,2
res={"matrix":[]}
for lam_c in [0,0.05,0.1]:
    for lam_e in [0,0.1]:
        for seed in [1]:
            res["matrix"].append({"lam_compute":lam_c,"lam_entropy":lam_e,"seed":seed,"status":"run_in_torch_training"})
with open("experiments/results/nf07_sweep_matrix.json","w") as f:
    json.dump(res,f,indent=2)
print("NF-07 sweep matrix saved (structure for torch run); count:", len(res["matrix"]))
