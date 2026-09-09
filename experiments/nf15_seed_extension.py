"""NF-15 - Seed extension (seeds 6-10) for the headline null result.

Motivation (registry RQ-NF): the soft-vs-fixed null rests on seeds 1-5 with
4/5 identical pairs (leave-one-out p in [0.39, 1.0]). Fresh seeds test
whether the null persists prospectively.

Method: identical protocols (NF-01 fixed x3 archs, NF-02 soft) on seeds
6-10. Per-seed calls only - existing artifacts are NEVER rewritten; this
script writes only nf15_seed_extension.json (+ new per-seed runs files from
the shared trainers). Heartbeats emitted per seed for detached execution.
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))

from nf01_fixed_baselines import train_fixed_baselines_single_seed
from nf02_soft_training import train_soft_single_seed

SEEDS = [6, 7, 8, 9, 10]


def heartbeat(msg):
    try:
        from lab.jobs import JobManager
        hb = os.environ.get("NF_HB", "")
        if hb:
            JobManager.heartbeat(hb, step=msg)
    except Exception:
        pass
    print(msg, flush=True)


def main():
    print("=== NF-15 seed extension (6-10) ===", flush=True)
    t0 = time.time()
    rows = []
    for s in SEEDS:
        fb = train_fixed_baselines_single_seed(seed=s)
        best = max(fb["architectures"][a]["test_accuracy"]
                   for a in ("mlp", "cnn", "attention"))
        soft = train_soft_single_seed(seed=s)
        rows.append({"seed": s,
                     "soft_acc": soft["test_accuracy"],
                     "best_fixed_acc": best,
                     "diff": round(soft["test_accuracy"] - best, 4)})
        heartbeat("seed %d soft=%.4f best_fixed=%.4f" % (s, rows[-1]["soft_acc"], best))
    diffs = [r["diff"] for r in rows]
    out = {"experiment": "nf15_seed_extension", "seeds": SEEDS,
           "elapsed_s": round(time.time() - t0, 2),
           "mean_diff_new_seeds": round(sum(diffs) / len(diffs), 4),
           "all_zero": all(d == 0 for d in diffs),
           "rows": rows,
           "interpretation": ("Fresh seeds replicate the null pattern exactly. "
                              if all(d == 0 for d in diffs) else
                              "Fresh-seed differences: see rows.")}
    with open("experiments/results/nf15_seed_extension.json", "w") as f:
        import json
        json.dump(out, f, indent=2)
    print("[DONE] NF-15", out["interpretation"], flush=True)


if __name__ == "__main__":
    main()
