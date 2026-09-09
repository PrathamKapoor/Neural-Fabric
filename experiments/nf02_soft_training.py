"""NF-02 — Soft routing training with multi-seed validation.

Trains input-conditioned router using analytical gradient descent on synthetic dataset.
Evaluates accuracy, routing entropy, routing collapse metrics, compute budget, and statistical significance.
"""
import os
import sys
import math
import random
import json
import time
from datetime import datetime

# Path setup
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from fabric.engine import FabricEngine
from fabric.router.router import Router
from evaluation.synthetic_dataset import make_sample


def student_t_pvalue(t, df):
    """Compute two-tailed p-value for Student's t-distribution using numerical integration."""
    t = abs(t)
    if t == 0.0 or df <= 0:
        return 1.0
    log_c = math.lgamma((df + 1.0) / 2.0) - 0.5 * math.log(df * math.pi) - math.lgamma(df / 2.0)
    c = math.exp(log_c)
    n_steps = 1000
    h = t / n_steps
    s = 0.0
    for i in range(n_steps + 1):
        x = i * h
        y = (1.0 + (x * x) / df) ** (-(df + 1.0) / 2.0)
        weight = 1 if (i == 0 or i == n_steps) else (4 if i % 2 == 1 else 2)
        s += weight * y
    area_0_to_t = c * s * (h / 3.0)
    p_val = max(0.0, min(1.0, 2.0 * (0.5 - area_0_to_t)))
    return round(p_val, 6)


def paired_t_test(sample_a, sample_b):
    """Compute paired t-statistic and two-tailed p-value."""
    n = len(sample_a)
    if n != len(sample_b) or n < 2:
        return {"t_stat": 0.0, "df": n - 1 if n > 1 else 0, "p_value": 1.0, "mean_diff": 0.0}
    diffs = [a - b for a, b in zip(sample_a, sample_b)]
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    if var_diff <= 1e-12:
        return {"t_stat": 0.0 if mean_diff == 0 else float("inf"), "df": n - 1, "p_value": 1.0 if mean_diff == 0 else 0.0, "mean_diff": mean_diff}
    std_diff = math.sqrt(var_diff)
    t_stat = mean_diff / (std_diff / math.sqrt(n))
    df = n - 1
    p_val = student_t_pvalue(t_stat, df)
    return {"t_stat": round(t_stat, 4), "df": df, "p_value": p_val, "mean_diff": round(mean_diff, 4)}


def one_sample_t_test(sample, null_value):
    """Compute one-sample t-statistic and two-tailed p-value against a null hypothesis value."""
    n = len(sample)
    if n < 2:
        return {"t_stat": 0.0, "df": 0, "p_value": 1.0, "mean": sample[0] if n == 1 else 0.0}
    mean = sum(sample) / n
    var = sum((x - mean) ** 2 for x in sample) / (n - 1)
    if var <= 1e-12:
        return {"t_stat": 0.0 if mean == null_value else float("inf"), "df": n - 1, "p_value": 1.0 if mean == null_value else 0.0, "mean": round(mean, 4)}
    std = math.sqrt(var)
    t_stat = (mean - null_value) / (std / math.sqrt(n))
    df = n - 1
    p_val = student_t_pvalue(t_stat, df)
    return {"t_stat": round(t_stat, 4), "df": df, "p_value": p_val, "mean": round(mean, 4), "null_value": null_value}


def calc_stats(values):
    """Calculate mean, std, min, max, and 95% confidence interval for a list of values."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "ci_95": [0.0, 0.0]}
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(var)
        t_crit_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
        t_crit = t_crit_table.get(n - 1, 1.96)
        margin = t_crit * (std / math.sqrt(n))
        ci_95 = [round(mean - margin, 4), round(mean + margin, 4)]
    else:
        std = 0.0
        ci_95 = [round(mean, 4), round(mean, 4)]
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "ci_95": ci_95,
        "n": n
    }


def train_soft_single_seed(seed=1, n_epochs=30, n_train=100, n_test=50, lr=0.05):
    """Train soft router for one seed, logging loss, entropy, collapse metrics, and accuracy."""
    random.seed(seed)
    train_modes = (["spatial", "relational", "dense"] * (n_train // 3 + 1))[:n_train]
    test_modes = (["spatial", "relational", "dense"] * (n_test // 3 + 1))[:n_test]
    random.shuffle(train_modes)
    random.shuffle(test_modes)

    train_data = [make_sample(m, seed=seed * 1000 + i) for i, m in enumerate(train_modes)]
    test_data = [make_sample(m, seed=seed * 2000 + i) for i, m in enumerate(test_modes)]

    label_map = {"dense": 0, "spatial": 1, "relational": 2}
    prim_names = ["mlp", "cnn", "attention"]
    prim_flops = {"mlp": 17152, "cnn": 1152, "attention": 12288}

    router = Router(in_dim=64, n_prims=3, mode="soft", seed=seed)
    
    loss_trend = []
    entropy_trend = []
    train_acc_trend = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_entropy = 0.0
        epoch_correct = 0

        for s in train_data:
            x = s["x"]
            y = label_map[s["label"]]

            h = router.encode(x)
            mx = max(h)
            ex = [math.exp(v - mx) for v in h]
            sm = sum(ex)
            probs = [v / sm for v in ex]

            p_target = max(probs[y], 1e-15)
            loss = -math.log(p_target)
            epoch_loss += loss

            ent = -sum(p * math.log(max(p, 1e-15)) for p in probs)
            epoch_entropy += ent

            pred = max(range(3), key=lambda j: probs[j])
            if pred == y:
                epoch_correct += 1

            for j in range(3):
                grad_h = probs[j] - (1.0 if j == y else 0.0)
                router.b[j] -= lr * grad_h
                for i in range(min(len(x), router.in_dim)):
                    router.W[i][j] -= lr * grad_h * x[i]

        loss_trend.append(round(epoch_loss / len(train_data), 4))
        entropy_trend.append(round(epoch_entropy / len(train_data), 4))
        train_acc_trend.append(round(epoch_correct / len(train_data), 4))

    test_loss = 0.0
    test_correct = 0
    test_entropies = []
    selections = []
    test_probs_all = []

    t0 = time.time()
    for s in test_data:
        x = s["x"]
        y = label_map[s["label"]]

        h = router.encode(x)
        mx = max(h)
        ex = [math.exp(v - mx) for v in h]
        sm = sum(ex)
        probs = [v / sm for v in ex]
        test_probs_all.append(probs)

        loss = -math.log(max(probs[y], 1e-15))
        test_loss += loss

        ent = -sum(p * math.log(max(p, 1e-15)) for p in probs)
        test_entropies.append(ent)

        pred = max(range(3), key=lambda j: probs[j])
        selections.append(prim_names[pred])
        if pred == y:
            test_correct += 1

    inference_time_ms = round((time.time() - t0) * 1000, 2)
    test_acc = round(test_correct / len(test_data), 4)
    test_mean_entropy = round(sum(test_entropies) / len(test_entropies), 4)
    test_loss_mean = round(test_loss / len(test_data), 4)

    total_test = len(test_data)
    sel_counts = {p: selections.count(p) for p in prim_names}
    sel_freqs = {p: round(sel_counts[p] / total_test, 4) for p in prim_names}
    max_primitive_ratio = round(max(sel_freqs.values()), 4)
    min_primitive_ratio = round(min(sel_freqs.values()), 4)

    avg_probs = [sum(test_probs_all[k][j] for k in range(total_test)) / total_test for j in range(3)]
    global_entropy = -sum(ap * math.log(max(ap, 1e-15)) for ap in avg_probs)
    normalized_global_entropy = round(global_entropy / math.log(3.0), 4)

    collapse_detected = (max_primitive_ratio > 0.85) or (normalized_global_entropy < 0.50)

    router_flops = 2 * router.in_dim * router.n
    executed_soft_flops = router_flops + sum(prim_flops.values())
    weighted_compute = router_flops + sum(avg_probs[j] * prim_flops[prim_names[j]] for j in range(3))

    seed_result = {
        "seed": seed,
        "n_epochs": n_epochs,
        "train_loss_final": loss_trend[-1],
        "train_accuracy_final": train_acc_trend[-1],
        "test_loss": test_loss_mean,
        "test_accuracy": test_acc,
        "test_entropy": test_mean_entropy,
        "loss_trend": loss_trend,
        "entropy_trend": entropy_trend,
        "train_accuracy_trend": train_acc_trend,
        "collapse_metrics": {
            "selection_counts": sel_counts,
            "selection_frequencies": sel_freqs,
            "max_primitive_ratio": max_primitive_ratio,
            "min_primitive_ratio": min_primitive_ratio,
            "global_routing_entropy": round(global_entropy, 4),
            "normalized_global_entropy": normalized_global_entropy,
            "collapse_detected": collapse_detected
        },
        "compute": {
            "router_flops": router_flops,
            "executed_flops": executed_soft_flops,
            "weighted_expected_flops": round(weighted_compute, 2),
            "inference_time_ms": inference_time_ms
        }
    }

    os.makedirs("experiments/runs", exist_ok=True)
    with open(f"experiments/runs/nf02_soft_seed{seed}.json", "w") as f:
        json.dump(seed_result, f, indent=2)

    return seed_result


def run_nf02_soft_training(seeds=None, n_epochs=30, n_train=100, n_test=50, lr=0.05):
    """Execute NF-02 multi-seed soft training and save structured results."""
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]

    print(f"=== Running NF-02 Soft Routing Training ({len(seeds)} seeds) ===")
    per_seed_results = []
    for s in seeds:
        res = train_soft_single_seed(seed=s, n_epochs=n_epochs, n_train=n_train, n_test=n_test, lr=lr)
        per_seed_results.append(res)
        print(f"Seed {s:2d} | Train Loss: {res['train_loss_final']:.4f} | "
              f"Test Acc: {res['test_accuracy']:.4f} | Entropy: {res['test_entropy']:.4f} | "
              f"Collapse: {res['collapse_metrics']['collapse_detected']}")

    test_accs = [r["test_accuracy"] for r in per_seed_results]
    train_losses = [r["train_loss_final"] for r in per_seed_results]
    test_entropies = [r["test_entropy"] for r in per_seed_results]
    max_ratios = [r["collapse_metrics"]["max_primitive_ratio"] for r in per_seed_results]
    global_entropies = [r["collapse_metrics"]["global_routing_entropy"] for r in per_seed_results]
    weighted_computes = [r["compute"]["weighted_expected_flops"] for r in per_seed_results]
    executed_computes = [r["compute"]["executed_flops"] for r in per_seed_results]

    summary = {
        "test_accuracy": calc_stats(test_accs),
        "train_loss": calc_stats(train_losses),
        "routing_entropy": calc_stats(test_entropies),
        "max_primitive_ratio": calc_stats(max_ratios),
        "global_routing_entropy": calc_stats(global_entropies),
        "weighted_compute_flops": calc_stats(weighted_computes),
        "executed_compute_flops": calc_stats(executed_computes)
    }

    t_test_acc = one_sample_t_test(test_accs, 1.0 / 3.0)
    t_test_ent = one_sample_t_test(test_entropies, math.log(3.0))
    any_collapse = any(r["collapse_metrics"]["collapse_detected"] for r in per_seed_results)

    final_payload = {
        "metadata": {
            "experiment": "NF-02 Soft Routing Training",
            "timestamp": datetime.now().isoformat(),
            "n_seeds": len(seeds),
            "seeds": seeds,
            "hyperparameters": {
                "n_epochs": n_epochs,
                "n_train": n_train,
                "n_test": n_test,
                "learning_rate": lr,
                "in_dim": 64,
                "n_prims": 3
            },
            "primitives": ["mlp", "cnn", "attention"]
        },
        "summary": summary,
        "statistical_tests": {
            "test_accuracy_vs_random_choice": {
                "null_hypothesis_accuracy": 0.3333,
                "observed_mean": summary["test_accuracy"]["mean"],
                "t_statistic": t_test_acc["t_stat"],
                "degrees_of_freedom": t_test_acc["df"],
                "p_value": t_test_acc["p_value"],
                "statistically_significant_p001": t_test_acc["p_value"] < 0.01
            },
            "routing_entropy_vs_uniform_router": {
                "null_hypothesis_entropy": round(math.log(3.0), 4),
                "observed_mean": summary["routing_entropy"]["mean"],
                "t_statistic": t_test_ent["t_stat"],
                "degrees_of_freedom": t_test_ent["df"],
                "p_value": t_test_ent["p_value"],
                "specialization_significant_p001": t_test_ent["p_value"] < 0.01
            }
        },
        "collapse_analysis": {
            "any_seed_collapsed": any_collapse,
            "mean_max_primitive_ratio": summary["max_primitive_ratio"]["mean"],
            "mean_global_entropy": summary["global_routing_entropy"]["mean"],
            "verdict": "No collapse observed; healthy multi-primitive specialization." if not any_collapse else "Warning: routing collapse detected on one or more seeds."
        },
        "per_seed_results": per_seed_results
    }

    os.makedirs("experiments/results", exist_ok=True)
    out_file = "experiments/results/nf02_soft_training.json"
    with open(out_file, "w") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\n[DONE] NF-02 results saved to {out_file}")
    print(f"Test Accuracy: {summary['test_accuracy']['mean']} +/- {summary['test_accuracy']['std']}")
    print(f"Routing Entropy: {summary['routing_entropy']['mean']} +/- {summary['routing_entropy']['std']}")
    print(f"p-value vs random baseline: {t_test_acc['p_value']}")
    return final_payload


if __name__ == "__main__":
    run_nf02_soft_training(seeds=[1, 2, 3, 4, 5])
