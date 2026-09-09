"""NF-09 - Robustness under noise / perturbation (noise-level curve).

Research question: at what noise magnitude does hard routing destabilize?

Method: hard-mode FabricEngine, graph execution; per noise level and seed,
compare clean vs noisy executed paths AND selected primitives. Noise grid
covers the original 0.05 level (backward compatible) plus weaker/stronger
perturbations. 5 seeds per level.
"""
import sys
import os
import json
import random
import time

sys.path.insert(0, ".")
from fabric.engine import FabricEngine

NOISE_LEVELS = [0.0, 0.02, 0.05, 0.1, 0.2]
SEEDS = [0, 1, 2, 3, 4]


def main():
    print("=== NF-09 robustness curve ===")
    t0 = time.time()
    engine = FabricEngine(mode="hard")
    levels = []
    for noise in NOISE_LEVELS:
        rows = []
        for seed in SEEDS:
            random.seed(seed)
            x = [random.random() for _ in range(64)]
            clean = engine.predict(x, graph_mode=True)
            noisy_x = [v + random.gauss(0, noise) for v in x] if noise > 0 else list(x)
            noisy = engine.predict(noisy_x, graph_mode=True)
            clean_nodes = clean["graph"]["nodes"]
            noisy_nodes = noisy["graph"]["nodes"]
            rows.append({
                "seed": seed,
                "clean_path": clean_nodes,
                "noisy_path": noisy_nodes,
                "path_changed": clean_nodes != noisy_nodes,
                "clean_compute": clean["compute"],
                "noisy_compute": noisy["compute"],
            })
        rate = sum(1 for r in rows if r["path_changed"]) / len(rows)
        levels.append({"noise_std": noise, "seeds": SEEDS,
                       "path_change_rate": round(rate, 4), "rows": rows})
        print(f"noise={noise} path_change_rate={rate:.2f}")
    from fabric.primitives.cost_model import COST_MODEL_VERSION, PRIM_FLOPS
    corrected = "--corrected" in sys.argv
    out = {"experiment": "nf09_robustness", "noise_levels": NOISE_LEVELS,
           "cost_model": COST_MODEL_VERSION, "prim_flops": dict(PRIM_FLOPS),
           "corrected_run": corrected,
           "elapsed_s": round(time.time() - t0, 2), "levels": levels}
    path = ("experiments/results/nf09_robustness_corrected.json"
            if corrected else "experiments/results/nf09_robustness.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[DONE] NF-09 saved to {path}")


if __name__ == "__main__":
    main()
