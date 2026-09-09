"""NF-01 — Fixed Homogeneous Baseline Experiments.

Trains MLP-only, CNN-only, and Attention-only models on the synthetic dataset
using the same training protocol (epochs=30, optimizer=SGD, lr=0.05, seeds=[42, 123, 456, 1, 2])
as NF-02 soft routing.

Enables direct statistical comparison and Pareto frontier evaluation:
heterogeneous soft routing vs. fixed homogeneous architectures under comparable compute budgets.
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

from fabric.primitives.mlp import MLP
from fabric.primitives.cnn import CNN
from fabric.primitives.attention import Attention
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
        return {
            "t_stat": 0.0 if mean_diff == 0 else float("inf"),
            "df": n - 1,
            "p_value": 1.0 if mean_diff == 0 else 0.0,
            "mean_diff": round(mean_diff, 4),
        }
    std_diff = math.sqrt(var_diff)
    t_stat = mean_diff / (std_diff / math.sqrt(n))
    df = n - 1
    p_val = student_t_pvalue(t_stat, df)
    return {
        "t_stat": round(t_stat, 4),
        "df": df,
        "p_value": p_val,
        "mean_diff": round(mean_diff, 4),
    }


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
        t_crit_table = {
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
        }
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
        "n": n,
    }


class FixedMLPModel:
    """Fixed homogeneous MLP architecture with 2 layers and ReLU activation."""

    def __init__(self, in_dim=64, hidden=128, out_dim=3, seed=0):
        self.primitive = MLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim, seed=seed)
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.name = "mlp"

    def forward(self, x):
        # Forward pass returning (logits, hidden_activations, pre_activations)
        x_pad = (x + [0.0] * (self.in_dim - len(x)))[: self.in_dim] if len(x) != self.in_dim else x
        z1 = [
            sum(x_pad[i] * self.primitive.W1[i][j] for i in range(self.in_dim))
            + self.primitive.b1[j]
            for j in range(self.hidden)
        ]
        h = [max(0.0, v) for v in z1]
        logits = [
            sum(h[j] * self.primitive.W2[j][k] for j in range(self.hidden))
            + self.primitive.b2[k]
            for k in range(self.out_dim)
        ]
        return logits, h, z1, x_pad

    def backward(self, x_pad, z1, h, grad_logits, lr=0.05):
        # Backprop through layer 2
        grad_h = [
            sum(self.primitive.W2[j][k] * grad_logits[k] for k in range(self.out_dim))
            for j in range(self.hidden)
        ]
        grad_z1 = [grad_h[j] if z1[j] > 0.0 else 0.0 for j in range(self.hidden)]

        # Update layer 2 weights and bias
        for k in range(self.out_dim):
            self.primitive.b2[k] -= lr * grad_logits[k]
            for j in range(self.hidden):
                self.primitive.W2[j][k] -= lr * h[j] * grad_logits[k]

        # Update layer 1 weights and bias
        for j in range(self.hidden):
            self.primitive.b1[j] -= lr * grad_z1[j]
            for i in range(self.in_dim):
                self.primitive.W1[i][j] -= lr * x_pad[i] * grad_z1[j]

    def params(self):
        return self.primitive.params()

    def flops(self):
        return self.primitive.flops()


class FixedCNNModel:
    """Fixed homogeneous CNN architecture with 8 3x3 filters, 2x2 quadrant pooling, and linear head."""

    def __init__(self, in_ch=1, out_ch=8, k=3, out_dim=3, seed=0):
        random.seed(seed)
        self.primitive = CNN(in_ch=in_ch, out_ch=out_ch, k=k, seed=seed)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = k
        self.out_dim = out_dim
        self.name = "cnn"
        self.num_pooled = out_ch * 4  # 32 features
        # Classification head: 32 -> 3
        self.W_head = [
            [random.gauss(0, 0.05) for _ in range(out_dim)] for _ in range(self.num_pooled)
        ]
        self.b_head = [random.gauss(0, 0.05) for _ in range(out_dim)]

    def forward(self, x):
        H, W = 8, 8
        expected_len = self.in_ch * H * W
        flat = (x + [0.0] * max(0, expected_len - len(x)))[:expected_len]
        img = [flat[idx * W : (idx + 1) * W] for idx in range(H)]
        pad = self.k // 2

        conv_maps = []
        for oc in range(self.out_ch):
            cmap = []
            for r in range(H):
                row = []
                for c in range(W):
                    val = 0.0
                    for kr in range(self.k):
                        ir = r + kr - pad
                        if 0 <= ir < H:
                            for kc in range(self.k):
                                ic_col = c + kc - pad
                                if 0 <= ic_col < W:
                                    val += img[ir][ic_col] * self.primitive.filters[oc][0][kr][kc]
                    row.append(val)
                cmap.append(row)
            conv_maps.append(cmap)

        pooled_feats = []
        for oc in range(self.out_ch):
            for qr in range(2):
                r_start, r_end = qr * 4, (qr + 1) * 4
                for qc in range(2):
                    c_start, c_end = qc * 4, (qc + 1) * 4
                    pool_sum = sum(
                        max(0.0, conv_maps[oc][r][c])
                        for r in range(r_start, r_end)
                        for c in range(c_start, c_end)
                    )
                    pooled_feats.append(pool_sum / 16.0)

        logits = [
            sum(pooled_feats[f] * self.W_head[f][k] for f in range(self.num_pooled))
            + self.b_head[k]
            for k in range(self.out_dim)
        ]
        return logits, pooled_feats, conv_maps, img

    def backward(self, img, conv_maps, pooled_feats, grad_logits, lr=0.05):
        H, W = 8, 8
        pad = self.k // 2

        # Gradients w.r.t pooled features
        grad_pooled = [
            sum(grad_logits[k] * self.W_head[f][k] for k in range(self.out_dim))
            for f in range(self.num_pooled)
        ]

        # Backprop through pooling & ReLU into conv filters
        for oc in range(self.out_ch):
            for qr in range(2):
                r_start, r_end = qr * 4, (qr + 1) * 4
                for qc in range(2):
                    c_start, c_end = qc * 4, (qc + 1) * 4
                    f_idx = oc * 4 + qr * 2 + qc
                    grad_pool_cell = grad_pooled[f_idx] / 16.0
                    for r in range(r_start, r_end):
                        for c in range(c_start, c_end):
                            if conv_maps[oc][r][c] > 0.0:
                                for kr in range(self.k):
                                    ir = r + kr - pad
                                    if 0 <= ir < H:
                                        for kc in range(self.k):
                                            ic_col = c + kc - pad
                                            if 0 <= ic_col < W:
                                                self.primitive.filters[oc][0][kr][kc] -= (
                                                    lr * grad_pool_cell * img[ir][ic_col]
                                                )

        # Update classification head
        for k in range(self.out_dim):
            self.b_head[k] -= lr * grad_logits[k]
            for f in range(self.num_pooled):
                self.W_head[f][k] -= lr * pooled_feats[f] * grad_logits[k]

    def params(self):
        head_params = self.num_pooled * self.out_dim + self.out_dim
        return self.primitive.params() + head_params

    def flops(self):
        head_flops = 2 * self.num_pooled * self.out_dim
        return self.primitive.flops() + head_flops


class FixedAttentionModel:
    """Fixed homogeneous Attention architecture with projection and linear head."""

    def __init__(self, d=32, n_heads=1, out_dim=3, seed=0):
        random.seed(seed)
        self.primitive = Attention(d=d, n_heads=n_heads, seed=seed)
        self.d = d
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.name = "attention"
        self.num_feats = 8  # primitive returns proj[:8]
        # Classification head: 8 -> 3
        self.W_head = [
            [random.gauss(0, 0.05) for _ in range(out_dim)] for _ in range(self.num_feats)
        ]
        self.b_head = [random.gauss(0, 0.05) for _ in range(out_dim)]

    def forward(self, x):
        if not isinstance(x, list):
            x = list(x)
        x_d = (x + [0.0] * max(0, self.d - len(x)))[: self.d]
        proj = [sum(x_d[i] * self.primitive.Wq[i][j] for i in range(self.d)) for j in range(self.d)]
        feats = proj[: self.num_feats]
        logits = [
            sum(feats[f] * self.W_head[f][k] for f in range(self.num_feats)) + self.b_head[k]
            for k in range(self.out_dim)
        ]
        return logits, feats, x_d

    def backward(self, x_d, feats, grad_logits, lr=0.05):
        # Gradients w.r.t attention projected features
        grad_feats = [
            sum(grad_logits[k] * self.W_head[f][k] for k in range(self.out_dim))
            for f in range(self.num_feats)
        ]

        # Backprop into attention projection matrix Wq
        for j in range(self.num_feats):
            for i in range(self.d):
                self.primitive.Wq[i][j] -= lr * grad_feats[j] * x_d[i]

        # Update classification head
        for k in range(self.out_dim):
            self.b_head[k] -= lr * grad_logits[k]
            for f in range(self.num_feats):
                self.W_head[f][k] -= lr * feats[f] * grad_logits[k]

    def params(self):
        head_params = self.num_feats * self.out_dim + self.out_dim
        return self.primitive.params() + head_params

    def flops(self):
        head_flops = 2 * self.num_feats * self.out_dim
        return self.primitive.flops() + head_flops


def train_single_architecture(arch_name, seed, train_data, test_data, label_map, n_epochs=30, lr=0.05):
    """Train a single fixed homogeneous architecture for one seed with SGD and cross-entropy loss."""
    if arch_name == "mlp":
        model = FixedMLPModel(in_dim=64, hidden=128, out_dim=3, seed=seed)
    elif arch_name == "cnn":
        model = FixedCNNModel(in_ch=1, out_ch=8, k=3, out_dim=3, seed=seed)
    elif arch_name == "attention":
        model = FixedAttentionModel(d=32, n_heads=1, out_dim=3, seed=seed)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

    loss_trend = []
    train_acc_trend = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_correct = 0

        for sample in train_data:
            x = sample["x"]
            y = label_map[sample["label"]]

            if arch_name == "mlp":
                logits, h, z1, x_pad = model.forward(x)
            elif arch_name == "cnn":
                logits, feats, conv_maps, img = model.forward(x)
            elif arch_name == "attention":
                logits, feats, x_d = model.forward(x)

            mx = max(logits)
            ex = [math.exp(v - mx) for v in logits]
            sm = sum(ex)
            probs = [v / sm for v in ex]

            p_target = max(probs[y], 1e-15)
            loss = -math.log(p_target)
            epoch_loss += loss

            pred = max(range(3), key=lambda k: logits[k])
            if pred == y:
                epoch_correct += 1

            grad_logits = [probs[k] - (1.0 if k == y else 0.0) for k in range(3)]

            if arch_name == "mlp":
                model.backward(x_pad, z1, h, grad_logits, lr=lr)
            elif arch_name == "cnn":
                model.backward(img, conv_maps, feats, grad_logits, lr=lr)
            elif arch_name == "attention":
                model.backward(x_d, feats, grad_logits, lr=lr)

        loss_trend.append(round(epoch_loss / len(train_data), 4))
        train_acc_trend.append(round(epoch_correct / len(train_data), 4))

    # Evaluation on test set
    test_loss_sum = 0.0
    test_correct = 0
    t0 = time.time()
    for sample in test_data:
        x = sample["x"]
        y = label_map[sample["label"]]

        if arch_name == "mlp":
            logits, _, _, _ = model.forward(x)
        elif arch_name == "cnn":
            logits, _, _, _ = model.forward(x)
        elif arch_name == "attention":
            logits, _, _ = model.forward(x)

        mx = max(logits)
        ex = [math.exp(v - mx) for v in logits]
        sm = sum(ex)
        probs = [v / sm for v in ex]

        test_loss_sum += -math.log(max(probs[y], 1e-15))
        pred = max(range(3), key=lambda k: logits[k])
        if pred == y:
            test_correct += 1

    inference_time_ms = round((time.time() - t0) * 1000, 2)
    test_acc = round(test_correct / len(test_data), 4)
    test_loss = round(test_loss_sum / len(test_data), 4)

    prim_usage = {"mlp": 0.0, "cnn": 0.0, "attention": 0.0}
    prim_usage[arch_name] = 1.0
    selection_counts = {"mlp": 0, "cnn": 0, "attention": 0}
    selection_counts[arch_name] = len(test_data)

    primitive_flops_table = {"mlp": 17152, "cnn": 1152, "attention": 12288}
    model_flops = model.flops()
    primitive_flops = primitive_flops_table[arch_name]

    collapse_metrics = {
        "selection_counts": selection_counts,
        "selection_frequencies": prim_usage,
        "max_primitive_ratio": 1.0,
        "min_primitive_ratio": 0.0,
        "global_routing_entropy": 0.0,
        "normalized_global_entropy": 0.0,
        "collapse_detected": True,
    }

    return {
        "architecture": arch_name,
        "seed": seed,
        "n_epochs": n_epochs,
        "train_loss_final": loss_trend[-1],
        "train_accuracy_final": train_acc_trend[-1],
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_entropy": 0.0,
        "accuracy": test_acc,
        "loss_trend": loss_trend,
        "loss_curve": loss_trend,
        "entropy_trend": [0.0] * n_epochs,
        "train_accuracy_trend": train_acc_trend,
        "compute_flops": model_flops,
        "primitive_flops": primitive_flops,
        "parameters": model.params(),
        "primitive_usage": prim_usage,
        "selection_counts": selection_counts,
        "selection_frequencies": prim_usage,
        "collapse_metrics": collapse_metrics,
        "compute": {
            "primitive_flops": primitive_flops,
            "total_model_flops": model_flops,
            "executed_flops": model_flops,
            "inference_time_ms": inference_time_ms,
        },
    }


def train_fixed_baselines_single_seed(seed=42, n_epochs=30, n_train=100, n_test=50, lr=0.05):
    """Train all 3 fixed homogeneous baselines (MLP, CNN, Attention) for a given seed."""
    random.seed(seed)
    train_modes = (["spatial", "relational", "dense"] * (n_train // 3 + 1))[:n_train]
    test_modes = (["spatial", "relational", "dense"] * (n_test // 3 + 1))[:n_test]
    random.shuffle(train_modes)
    random.shuffle(test_modes)

    train_data = [make_sample(m, seed=seed * 1000 + i) for i, m in enumerate(train_modes)]
    test_data = [make_sample(m, seed=seed * 2000 + i) for i, m in enumerate(test_modes)]
    label_map = {"dense": 0, "spatial": 1, "relational": 2}

    arch_results = {}
    for arch in ["mlp", "cnn", "attention"]:
        res = train_single_architecture(
            arch, seed, train_data, test_data, label_map, n_epochs=n_epochs, lr=lr
        )
        arch_results[arch] = res

        # Also save individual architecture run file for maximum compatibility
        os.makedirs("experiments/runs", exist_ok=True)
        with open(f"experiments/runs/nf01_fixed_{arch}_seed{seed}.json", "w") as f:
            json.dump(res, f, indent=2)

    # Determine best fixed baseline on test accuracy (or by compute if tied)
    best_arch = max(arch_results.keys(), key=lambda a: (arch_results[a]["test_accuracy"], -arch_results[a]["compute_flops"]))

    seed_result = {
        "seed": seed,
        "n_epochs": n_epochs,
        "train_loss_final": arch_results[best_arch]["train_loss_final"],
        "train_accuracy_final": arch_results[best_arch]["train_accuracy_final"],
        "test_loss": arch_results[best_arch]["test_loss"],
        "test_accuracy": arch_results[best_arch]["test_accuracy"],
        "test_entropy": 0.0,
        "accuracy": arch_results[best_arch]["accuracy"],
        "loss_curve": arch_results[best_arch]["loss_curve"],
        "loss_trend": arch_results[best_arch]["loss_trend"],
        "train_accuracy_trend": arch_results[best_arch]["train_accuracy_trend"],
        "compute_flops": arch_results[best_arch]["compute_flops"],
        "primitive_usage": arch_results[best_arch]["primitive_usage"],
        "best_baseline": best_arch,
        "collapse_metrics": arch_results[best_arch]["collapse_metrics"],
        "compute": arch_results[best_arch]["compute"],
        "architectures": arch_results,
    }

    # Save joint seed run file
    with open(f"experiments/runs/nf01_fixed_seed{seed}.json", "w") as f:
        json.dump(seed_result, f, indent=2)

    # Save per-seed result in experiments/results/ for direct visibility and comparability
    os.makedirs("experiments/results", exist_ok=True)
    seed_result_payload = {
        "experiment": "nf01_fixed_baselines",
        "seed": seed,
        "n_epochs": n_epochs,
        "best_baseline": best_arch,
        "best_test_accuracy": arch_results[best_arch]["test_accuracy"],
        "compute_flops": arch_results[best_arch]["compute_flops"],
        "architectures": {
            arch: {
                "test_accuracy": arch_results[arch]["test_accuracy"],
                "train_loss_final": arch_results[arch]["train_loss_final"],
                "train_accuracy_final": arch_results[arch]["train_accuracy_final"],
                "parameters": arch_results[arch]["parameters"],
                "compute_flops": arch_results[arch]["compute_flops"],
                "primitive_flops": arch_results[arch]["primitive_flops"],
                "loss_curve": arch_results[arch]["loss_curve"],
                "train_accuracy_trend": arch_results[arch]["train_accuracy_trend"],
            }
            for arch in ["mlp", "cnn", "attention"]
        }
    }
    with open(f"experiments/results/nf01_fixed_baselines_{seed}.json", "w") as f:
        json.dump(seed_result_payload, f, indent=2)

    return seed_result


def run_nf01_fixed_baselines(seeds=None, n_epochs=30, n_train=100, n_test=50, lr=0.05):
    """Execute fixed homogeneous baseline experiments across 8 seeds matching NF-02."""
    if seeds is None:
        seeds = [42, 123, 456, 1, 2, 3, 4, 5]

    print(f"=== Running NF-01 Fixed Homogeneous Baselines ({len(seeds)} seeds: {seeds}) ===")
    per_seed_runs = []

    for s in seeds:
        t_start = time.time()
        run_res = train_fixed_baselines_single_seed(
            seed=s, n_epochs=n_epochs, n_train=n_train, n_test=n_test, lr=lr
        )
        per_seed_runs.append(run_res)
        dur = round(time.time() - t_start, 2)
        archs = run_res["architectures"]
        print(
            f"Seed {s:3d} ({dur}s) | MLP Acc: {archs['mlp']['test_accuracy']:.4f} "
            f"(Loss: {archs['mlp']['train_loss_final']:.4f}) | "
            f"CNN Acc: {archs['cnn']['test_accuracy']:.4f} "
            f"(Loss: {archs['cnn']['train_loss_final']:.4f}) | "
            f"Attn Acc: {archs['attention']['test_accuracy']:.4f} "
            f"(Loss: {archs['attention']['train_loss_final']:.4f})"
        )

    # Compute summaries across seeds for each architecture
    summary = {}
    for arch in ["mlp", "cnn", "attention"]:
        accs = [r["architectures"][arch]["test_accuracy"] for r in per_seed_runs]
        train_losses = [r["architectures"][arch]["train_loss_final"] for r in per_seed_runs]
        train_accs = [r["architectures"][arch]["train_accuracy_final"] for r in per_seed_runs]
        flops = [r["architectures"][arch]["compute_flops"] for r in per_seed_runs]
        params = [r["architectures"][arch]["parameters"] for r in per_seed_runs]

        summary[arch] = {
            "test_accuracy": calc_stats(accs),
            "train_loss": calc_stats(train_losses),
            "train_accuracy": calc_stats(train_accs),
            "compute_flops": calc_stats(flops),
            "parameters": calc_stats(params),
        }

    # Load NF-02 soft routing results for the matching seeds
    nf02_accs = []
    nf02_executed_flops = []
    nf02_weighted_flops = []
    for s in seeds:
        nf02_run_path = f"experiments/runs/nf02_soft_seed{s}.json"
        if os.path.exists(nf02_run_path):
            with open(nf02_run_path) as f:
                nf02_data = json.load(f)
                nf02_accs.append(nf02_data.get("test_accuracy", 0.0))
                nf02_executed_flops.append(nf02_data.get("compute", {}).get("executed_flops", 30976))
                nf02_weighted_flops.append(nf02_data.get("compute", {}).get("weighted_expected_flops", 10000))
        else:
            print(f"Warning: NF-02 run file for seed {s} not found at {nf02_run_path}")

    # Statistical comparisons vs NF-02 soft routing
    statistical_tests = {}
    if len(nf02_accs) == len(seeds):
        summary["nf02_soft_routing"] = {
            "test_accuracy": calc_stats(nf02_accs),
            "executed_compute_flops": calc_stats(nf02_executed_flops),
            "weighted_compute_flops": calc_stats(nf02_weighted_flops),
        }

        for arch in ["mlp", "cnn", "attention"]:
            arch_accs = [r["architectures"][arch]["test_accuracy"] for r in per_seed_runs]
            tt = paired_t_test(nf02_accs, arch_accs)
            statistical_tests[f"nf02_soft_vs_{arch}"] = {
                "nf02_mean": round(sum(nf02_accs) / len(nf02_accs), 4),
                f"{arch}_mean": round(sum(arch_accs) / len(arch_accs), 4),
                "mean_accuracy_diff": tt["mean_diff"],
                "t_statistic": tt["t_stat"],
                "degrees_of_freedom": tt["df"],
                "p_value": tt["p_value"],
                "statistically_significant_p05": tt["p_value"] < 0.05,
                "statistically_significant_p01": tt["p_value"] < 0.01,
            }

        # Best fixed baseline comparison
        best_fixed_accs = [
            max(r["architectures"][a]["test_accuracy"] for a in ["mlp", "cnn", "attention"])
            for r in per_seed_runs
        ]
        tt_best = paired_t_test(nf02_accs, best_fixed_accs)
        statistical_tests["nf02_soft_vs_best_fixed"] = {
            "nf02_mean": round(sum(nf02_accs) / len(nf02_accs), 4),
            "best_fixed_mean": round(sum(best_fixed_accs) / len(best_fixed_accs), 4),
            "mean_accuracy_diff": tt_best["mean_diff"],
            "t_statistic": tt_best["t_stat"],
            "degrees_of_freedom": tt_best["df"],
            "p_value": tt_best["p_value"],
            "statistically_significant_p05": tt_best["p_value"] < 0.05,
        }

    # Accuracy / Compute Pareto Frontier Analysis
    # Candidates: Fixed CNN, Fixed Attention, Fixed MLP, NF-02 Soft Routing
    pareto_candidates = [
        {
            "name": "Fixed CNN",
            "architecture": "cnn",
            "mean_accuracy": summary["cnn"]["test_accuracy"]["mean"],
            "compute_flops": summary["cnn"]["compute_flops"]["mean"],
            "parameters": summary["cnn"]["parameters"]["mean"],
            "accuracy_per_kflop": round(
                summary["cnn"]["test_accuracy"]["mean"] / (summary["cnn"]["compute_flops"]["mean"] / 1000.0), 4
            ),
        },
        {
            "name": "Fixed Attention",
            "architecture": "attention",
            "mean_accuracy": summary["attention"]["test_accuracy"]["mean"],
            "compute_flops": summary["attention"]["compute_flops"]["mean"],
            "parameters": summary["attention"]["parameters"]["mean"],
            "accuracy_per_kflop": round(
                summary["attention"]["test_accuracy"]["mean"]
                / (summary["attention"]["compute_flops"]["mean"] / 1000.0),
                4,
            ),
        },
        {
            "name": "Fixed MLP",
            "architecture": "mlp",
            "mean_accuracy": summary["mlp"]["test_accuracy"]["mean"],
            "compute_flops": summary["mlp"]["compute_flops"]["mean"],
            "parameters": summary["mlp"]["parameters"]["mean"],
            "accuracy_per_kflop": round(
                summary["mlp"]["test_accuracy"]["mean"] / (summary["mlp"]["compute_flops"]["mean"] / 1000.0), 4
            ),
        },
    ]

    if "nf02_soft_routing" in summary:
        pareto_candidates.append(
            {
                "name": "NF-02 Soft Routing (Executed)",
                "architecture": "soft_routing",
                "mean_accuracy": summary["nf02_soft_routing"]["test_accuracy"]["mean"],
                "compute_flops": summary["nf02_soft_routing"]["executed_compute_flops"]["mean"],
                "parameters": 384 + 8707 + 72 + 3072,  # router + all primitives
                "accuracy_per_kflop": round(
                    summary["nf02_soft_routing"]["test_accuracy"]["mean"]
                    / (summary["nf02_soft_routing"]["executed_compute_flops"]["mean"] / 1000.0),
                    4,
                ),
            }
        )

    # Determine Pareto frontier (higher accuracy, lower compute FLOPs)
    # A candidate A dominates B if accuracy(A) >= accuracy(B) and compute(A) <= compute(B) with at least one strict
    for cand in pareto_candidates:
        dominated = False
        for other in pareto_candidates:
            if other["name"] == cand["name"]:
                continue
            if (other["mean_accuracy"] >= cand["mean_accuracy"] and other["compute_flops"] < cand["compute_flops"]) or \
               (other["mean_accuracy"] > cand["mean_accuracy"] and other["compute_flops"] <= cand["compute_flops"]):
                dominated = True
                break
        cand["is_pareto_optimal"] = not dominated

    pareto_frontier = [c["name"] for c in pareto_candidates if c["is_pareto_optimal"]]

    final_payload = {
        "metadata": {
            "experiment": "NF-01 Fixed Homogeneous Baselines",
            "timestamp": datetime.now().isoformat(),
            "n_seeds": len(seeds),
            "seeds": seeds,
            "hyperparameters": {
                "n_epochs": n_epochs,
                "n_train": n_train,
                "n_test": n_test,
                "learning_rate": lr,
                "in_dim": 64,
                "n_classes": 3,
            },
            "architectures": ["mlp", "cnn", "attention"],
        },
        "summary": summary,
        "statistical_tests": statistical_tests,
        "pareto_analysis": {
            "candidates": pareto_candidates,
            "pareto_frontier": pareto_frontier,
            "pareto_optimal_architectures": [c for c in pareto_candidates if c["is_pareto_optimal"]],
            "efficiency_winner": max(pareto_candidates, key=lambda c: c["accuracy_per_kflop"])["name"],
            "accuracy_winner": max(pareto_candidates, key=lambda c: c["mean_accuracy"])["name"],
        },
        "per_seed_results": per_seed_runs,
    }

    os.makedirs("experiments/results", exist_ok=True)
    out_file = "experiments/results/nf01_fixed_baselines.json"
    with open(out_file, "w") as f:
        json.dump(final_payload, f, indent=2)

    # Save per-architecture multi-seed summary files
    for arch in ["mlp", "cnn", "attention"]:
        arch_summary = {
            "experiment": "nf01_fixed_baselines",
            "architecture": arch,
            "seeds": seeds,
            "n_seeds": len(seeds),
            "summary": summary[arch],
            "per_seed_results": [
                {
                    "seed": r["seed"],
                    "test_accuracy": r["architectures"][arch]["test_accuracy"],
                    "train_loss_final": r["architectures"][arch]["train_loss_final"],
                    "train_accuracy_final": r["architectures"][arch]["train_accuracy_final"],
                    "compute_flops": r["architectures"][arch]["compute_flops"],
                    "parameters": r["architectures"][arch]["parameters"],
                }
                for r in per_seed_runs
            ]
        }
        with open(f"experiments/results/nf01_fixed_baselines_{arch}.json", "w") as f:
            json.dump(arch_summary, f, indent=2)

    print(f"\n[DONE] NF-01 fixed baseline results saved to {out_file}")
    for arch in ["mlp", "cnn", "attention"]:
        s_acc = summary[arch]["test_accuracy"]
        s_flops = summary[arch]["compute_flops"]
        print(f"  {arch.upper()}-only: Acc = {s_acc['mean']} +/- {s_acc['std']} | FLOPs = {s_flops['mean']}")

    if "nf02_soft_vs_best_fixed" in statistical_tests:
        st = statistical_tests["nf02_soft_vs_best_fixed"]
        print(f"  NF-02 vs Best Fixed: Diff = {st['mean_accuracy_diff']} (p = {st['p_value']})")
    print(f"  Pareto Frontier: {pareto_frontier}")

    return final_payload


if __name__ == "__main__":
    run_nf01_fixed_baselines(seeds=[42, 123, 456, 1, 2, 3, 4, 5])
