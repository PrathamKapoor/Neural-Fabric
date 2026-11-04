"""Training, evaluation and validated inference."""
import copy
import math
import platform
import statistics
import subprocess
import time
import torch
from torch.nn import functional as F
from .model import RoutedModel, NAMES, compute, EXPERT_FLOPS, CONVENTION
from .data import split
from .checkpoint import save


def validate_input(values):
    if not isinstance(values, list) or len(values) != 64:
        raise ValueError("input must be a list of exactly 64 finite numbers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("input must contain only finite numbers")
    return values


def train(path, seed=42, epochs=20):
    if not 1 <= epochs <= 500:
        raise ValueError("epochs must be between 1 and 500")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = RoutedModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x, y = split("train", seed)
    vx, vy = split("validation", seed)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    generator = torch.Generator().manual_seed(seed + 99)
    for epoch in range(epochs):
        model.train()
        for batch in torch.randperm(len(x), generator=generator).split(64):
            optimizer.zero_grad()
            logits, _, _ = model(x[batch], "soft")
            F.cross_entropy(logits, y[batch]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(model(vx, "soft")[0], vy).item()
        if val_loss < best_loss:
            best_loss, best_state, best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch + 1
    model.load_state_dict(best_state)
    # A separately trained fixed expert sees the same splits, epochs and optimizer.
    torch.manual_seed(seed + 1)
    baseline = RoutedModel().experts[1]
    base_optimizer = torch.optim.Adam(baseline.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(seed + 99)
    base_loss, base_state, base_epoch = float("inf"), None, 0
    for epoch in range(epochs):
        for batch in torch.randperm(len(x), generator=generator).split(64):
            base_optimizer.zero_grad()
            F.cross_entropy(baseline(x[batch]), y[batch]).backward()
            base_optimizer.step()
        with torch.no_grad():
            val = F.cross_entropy(baseline(vx), vy).item()
        if val < base_loss:
            base_loss, base_state, base_epoch = val, copy.deepcopy(baseline.state_dict()), epoch + 1
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    metadata = {"seed": seed, "epochs": epochs, "selected_epoch": best_epoch,
                "validation_loss": best_loss, "optimizer": "Adam", "learning_rate": 0.01,
                "batch_size": 64, "code_commit": commit, "torch_version": str(torch.__version__),
                "baseline_validation_loss": base_loss, "baseline_selected_epoch": base_epoch}
    save(path, model, metadata, base_state)
    return metadata


def predict(model, values, mode="top1", k=1):
    validate_input(values)
    x = torch.tensor([values], dtype=torch.float32)
    if not torch.isfinite(x).all():
        raise ValueError("input values exceed float32 range")
    with torch.no_grad():
        logits, route, mask = model(x, mode, k)
        probs = logits.softmax(-1)[0]
    return {"model_version": "linear-smallmlp-widemlp-v1", "mode": mode,
            "prediction": int(probs.argmax()), "probabilities": probs.tolist(),
            "routing": {"probabilities": route[0].tolist(),
                        "selected_experts": [name for i, name in enumerate(NAMES) if mask[0, i]]},
            "compute": compute(mask[0])}


def evaluate(model, seed, mode="top1", k=1, repeats=30):
    torch.set_num_threads(1)
    x, y = split("test", seed)
    with torch.no_grad():
        logits, _, mask = model(x, mode, k)
        baseline_logits = model.baseline(x)
        loss = F.cross_entropy(logits, y).item()
        accuracy = (logits.argmax(1) == y).float().mean().item()
        costs = [compute(row)["total_flops_estimate"] for row in mask]
        for _ in range(5):
            model(x[:1], mode, k)
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(x[:1], mode, k)
            timings.append((time.perf_counter() - start) * 1000)
        for _ in range(5):
            model.baseline(x[:1])
        baseline_timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            model.baseline(x[:1])
            baseline_timings.append((time.perf_counter() - start) * 1000)
    timings.sort()
    baseline_timings.sort()
    return {"dataset": "synthetic-linear-v1", "split": "test", "seed": seed,
            "examples": len(y), "mode": mode, "accuracy": accuracy, "loss": loss,
            "fixed_baseline_accuracy": (baseline_logits.argmax(1) == y).float().mean().item(),
            "fixed_baseline_loss": F.cross_entropy(baseline_logits, y).item(),
            "fixed_baseline_parameters": sum(p.numel() for p in model.baseline.parameters()),
            "router_flops_per_sample": compute(mask[0])["router_flops_estimate"],
            "expert_flops_mean": statistics.mean(costs) - compute(mask[0])["router_flops_estimate"],
            "total_flops_mean": statistics.mean(costs),
            "baseline": "fixed small_mlp", "baseline_flops": EXPERT_FLOPS[1],
            "savings_percent": 100 * (1 - statistics.mean(costs) / EXPERT_FLOPS[1]),
            "convention": CONVENTION, "parameters": sum(p.numel() for p in model.router.parameters()) + sum(p.numel() for p in model.experts.parameters()),
            "latency": {"median_ms": statistics.median(timings),
                        "p95_ms": timings[math.ceil(.95 * len(timings)) - 1],
                        "baseline_median_ms": statistics.median(baseline_timings),
                        "baseline_p95_ms": baseline_timings[math.ceil(.95 * len(baseline_timings)) - 1],
                        "repeats": repeats, "warmup": 5, "batch_size": 1,
                        "device": "cpu", "python": platform.python_version(),
                        "torch": str(torch.__version__), "threads": 1}}
