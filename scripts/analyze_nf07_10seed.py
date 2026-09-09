"""NF-07 10-seed statistical analysis (committed scientific record).

Reads : experiments/results/nf07_sweep_10seed_corrected.json
Writes: experiments/results/nf07_10seed_statistics.json

Method (no model fitting; pure descriptive + paired inference):
- Per condition: mean/SD/median/min/max of accuracy and top-1 FLOPs.
- Primary contrast: each penalized cell vs the unpenalized baseline
  (lam_ent=0, lam_comp=0), PAIRED BY SEED (same seed = same data splits and
  router init, so pairing removes seed-level variability — this is why a
  paired test, not a two-sample test, is appropriate).
- 95% CI for the mean paired difference via t_{n-1} critical values, plus a
  paired t two-tailed p-value computed from first principles (no scipy).
- Sign-consistency counts (how many seeds move in the claimed direction).
- Caveats recorded in-artifact: n=10; test sets are 50 samples so accuracy
  granularity is 0.02; differences are discrete/bounded so t-normality is
  approximate — CIs and sign counts carry equal weight to p-values here.
"""
import json
import math
import os
import sys
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from fabric.primitives.cost_model import COST_MODEL_VERSION, PRIM_FLOPS

T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
             29: 2.045}


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)


def desc(xs):
    n = len(xs)
    mean = sum(xs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": round(mean, 4), "sd": round(sd, 4),
            "median": round(_median(xs), 4), "min": round(min(xs), 4),
            "max": round(max(xs), 4)}


def paired_t_pvalue(a, b):
    """Paired t-test two-tailed p-value (own implementation, no scipy)."""
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var <= 1e-12:
        return (1.0 if mean == 0 else 0.0), round(mean, 4)
    t = abs(mean / (math.sqrt(var) / math.sqrt(n)))
    df = n - 1
    log_c = math.lgamma((df + 1.0) / 2.0) - 0.5 * math.log(df * math.pi) \
        - math.lgamma(df / 2.0)
    c = math.exp(log_c)
    steps, h = 2000, t / 2000
    s = sum((1 if i in (0, 2000) else (4 if i % 2 else 2))
            * (1.0 + ((i * h) ** 2) / df) ** (-(df + 1.0) / 2.0)
            for i in range(2001))
    p = max(0.0, min(1.0, 2.0 * (0.5 - c * s * (h / 3.0))))
    return round(p, 6), round(mean, 4)


def mean_diff_ci(a, b, conf=0.95):
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0
    t = T_CRIT_95.get(n - 1, 1.96)
    margin = t * sd / math.sqrt(n) if n > 0 else 0.0
    return {"mean_diff": round(mean, 4),
            "ci95": [round(mean - margin, 4), round(mean + margin, 4)]}


def main():
    src = os.path.join(REPO, "experiments/results/nf07_sweep_10seed_corrected.json")
    d = json.load(open(src))
    assert d["cost_model"] == COST_MODEL_VERSION
    assert d["failures"] == [], d["failures"]
    cells = {(c["lam_ent"], c["lam_comp"]): c for c in d["cells"]}
    base = cells[(0.0, 0.0)]
    base_acc = [r["test_accuracy"] for r in base["per_seed"]]
    base_fl = [r["top1_executed_flops"] for r in base["per_seed"]]

    summary, contrasts = {}, {}
    for key, c in sorted(cells.items()):
        accs = [r["test_accuracy"] for r in c["per_seed"]]
        fl = [r["top1_executed_flops"] for r in c["per_seed"]]
        summary[f"le={key[0]}_lc={key[1]}"] = {
            "accuracy": desc(accs), "top1_flops": desc(fl),
            "any_collapse": c["any_collapse"],
            "n_completed": c["n_completed"],
        }
        if key != (0.0, 0.0):
            p_acc, md_acc = paired_t_pvalue(accs, base_acc)
            p_fl, md_fl = paired_t_pvalue(fl, base_fl)
            ci_acc = mean_diff_ci(accs, base_acc)
            ci_fl = mean_diff_ci(fl, base_fl)
            n_acc_down = sum(1 for x, y in zip(accs, base_acc) if x < y)
            n_acc_same = sum(1 for x, y in zip(accs, base_acc) if x == y)
            n_fl_down = sum(1 for x, y in zip(fl, base_fl) if x < y)
            contrasts[f"le={key[0]}_lc={key[1]}_vs_baseline"] = {
                "accuracy_paired": {
                    "mean_diff": md_acc, "ci95": ci_acc["ci95"],
                    "p_two_tailed": p_acc,
                    "n_seeds_lower": n_acc_down, "n_seeds_equal": n_acc_same,
                    "n_seeds_total": len(accs)},
                "flops_paired": {
                    "mean_diff": md_fl, "ci95": ci_fl["ci95"],
                    "p_two_tailed": p_fl,
                    "n_seeds_lower": n_fl_down,
                    "n_seeds_total": len(fl)},
            }

    out = {
        "analysis": "nf07_10seed_statistics",
        "source_artifact": "experiments/results/nf07_sweep_10seed_corrected.json",
        "cost_model": COST_MODEL_VERSION, "prim_flops": dict(PRIM_FLOPS),
        "method": ("descriptive stats + paired-by-seed contrasts vs unpenalized "
                   "baseline; paired t justified by shared seeds/splits; "
                   "t-normality approximate on discrete 0.02-granularity "
                   "accuracies — CIs and sign counts weigh equally with p-values"),
        "timestamp": datetime.now().isoformat(),
        "per_condition": summary,
        "contrasts_vs_unpenalized_baseline": contrasts,
        "limitations": [
            "n=10 seeds; modest power, wide CIs",
            "test accuracy granularity 0.02 (50-sample test sets)",
            "synthetic benchmark only; no generalization claim",
            "paired-t normality approximate on bounded discrete differences",
        ],
    }
    path = os.path.join(REPO, "experiments/results/nf07_10seed_statistics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[DONE] wrote {path}")
    for k, v in contrasts.items():
        print(k, "| acc:", v["accuracy_paired"], "| flops:", v["flops_paired"])


if __name__ == "__main__":
    main()
