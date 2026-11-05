"""CPU classification of numeric CSV files, with train-only preprocessing."""
import copy
import hashlib
import statistics
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .csv_data import read_csv, partition, fit_preprocessing, transform


class CsvModel(nn.Module):
    def __init__(self, inputs, classes):
        super().__init__()
        self.router = nn.Linear(inputs, 3)
        self.experts = nn.ModuleList([nn.Linear(inputs, classes),
            nn.Sequential(nn.Linear(inputs, 16), nn.ReLU(), nn.Linear(16, classes)),
            nn.Sequential(nn.Linear(inputs, 32), nn.ReLU(), nn.Linear(32, classes))])

    def forward(self, x, mode="soft"):
        weights = self.router(x).softmax(-1)
        if mode == "soft":
            return sum(self.experts[i](x) * weights[:, i:i+1] for i in range(3))
        if mode != "top1":
            raise ValueError("mode must be soft or top1")
        selected = weights.argmax(-1)
        output = x.new_zeros((len(x), self.experts[0].out_features))
        for i, expert in enumerate(self.experts):
            mask = selected == i
            if mask.any():
                output[mask] = expert(x[mask])
        return output


def train_csv(path, target, output, seed=42, epochs=20):
    if not 1 <= epochs <= 500:
        raise ValueError("epochs must be between 1 and 500")
    names, rows, labels = read_csv(path, target=target)
    tr, val, test = partition(labels, seed)
    preprocessing = fit_preprocessing([rows[i] for i in tr])
    x = transform(rows, preprocessing)
    classes = sorted(set(labels))
    y = torch.tensor([classes.index(label) for label in labels], dtype=torch.long)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = CsvModel(len(names), len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        for indices in torch.randperm(len(tr), generator=generator).split(64):
            batch = torch.tensor(tr)[indices]
            optimizer.zero_grad()
            F.cross_entropy(model(x[batch]), y[batch]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = F.cross_entropy(model(x[val]), y[val]).item()
        if loss < best_loss:
            best_loss, best_state, best_epoch = loss, copy.deepcopy(model.state_dict()), epoch + 1
    model.load_state_dict(best_state)
    baselines = {}
    for expert_index, name in enumerate(("linear", "small_mlp", "wide_mlp")):
        torch.manual_seed(seed + expert_index + 1)
        baseline = CsvModel(len(names), len(classes)).experts[expert_index]
        base_optimizer = torch.optim.Adam(baseline.parameters(), lr=.01)
        generator = torch.Generator().manual_seed(seed)
        best_base_loss, best_base_state = float("inf"), None
        for epoch in range(epochs):
            for indices in torch.randperm(len(tr), generator=generator).split(64):
                batch = torch.tensor(tr)[indices]
                base_optimizer.zero_grad()
                F.cross_entropy(baseline(x[batch]), y[batch]).backward()
                base_optimizer.step()
            with torch.no_grad():
                loss = F.cross_entropy(baseline(x[val]), y[val]).item()
            if loss < best_base_loss:
                best_base_loss, best_base_state = loss, copy.deepcopy(baseline.state_dict())
        baselines[name] = best_base_state
    checkpoint = {"format": "neural-fabric-csv-v1", "features": names, "target": target,
                  "labels": classes, "preprocessing": preprocessing, "seed": seed,
                  "epochs": epochs, "selected_epoch": best_epoch, "validation_loss": best_loss,
                  "dataset_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                  "split_indices": {"train": tr, "validation": val, "test": test},
                  "state_dict": model.state_dict(), "baselines": baselines}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {"training_rows": len(tr), "validation_rows": len(val), "test_rows": len(test),
            "validation_loss": best_loss, "selected_epoch": best_epoch}


def load_csv_model(path):
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["format"] != "neural-fabric-csv-v1":
            raise ValueError("unsupported format")
        model = CsvModel(len(data["features"]), len(data["labels"]))
        model.load_state_dict(data["state_dict"])
        model.eval()
        return model, data
    except Exception as exc:
        raise ValueError(f"invalid CSV model checkpoint: {exc}") from exc


def predict_csv(model_path, csv_path, mode="top1"):
    model, data = load_csv_model(model_path)
    _, rows, _ = read_csv(csv_path, features=data["features"])
    with torch.no_grad():
        probs = model(transform(rows, data["preprocessing"]), mode).softmax(-1)
    return [{"prediction": data["labels"][int(row.argmax())], "probabilities": row.tolist()} for row in probs]


def evaluate_csv(model_path, csv_path, mode="top1"):
    model, data = load_csv_model(model_path)
    _, rows, labels = read_csv(csv_path, target=data["target"], features=data["features"])
    if hashlib.sha256(Path(csv_path).read_bytes()).hexdigest() != data["dataset_sha256"]:
        raise ValueError("evaluation CSV must be the same training dataset in the same order")
    indices = data["split_indices"]["test"]
    try:
        y = torch.tensor([data["labels"].index(labels[i]) for i in indices])
    except ValueError as exc:
        raise ValueError("dataset contains unknown labels") from exc
    with torch.no_grad():
        x = transform([rows[i] for i in indices], data["preprocessing"])
        logits = model(x, mode)
        predictions = {"routed_" + mode: logits}
        for i, name in enumerate(("linear", "small_mlp", "wide_mlp")):
            expert = model.experts[i]
            expert.load_state_dict(data["baselines"][name])
            predictions[name] = expert(x)
        # Restore routed experts after evaluating separately trained baselines.
        model.load_state_dict(data["state_dict"])
        costs = [2*len(data["features"])*len(data["labels"]),
                 2*(len(data["features"])*16 + 16*len(data["labels"])),
                 2*(len(data["features"])*32 + 32*len(data["labels"]))]
        selected = model.router(x).argmax(-1)
        router_cost = 2*len(data["features"])*3
        model_cost = router_cost + (sum(costs) if mode == "soft" else sum(costs[int(i)] for i in selected)/len(selected))
        results = {}
        for name, values in predictions.items():
            if name.startswith("routed_"):
                call = lambda: model(x[:1], mode)
                flops = model_cost
            else:
                index = ("linear", "small_mlp", "wide_mlp").index(name)
                expert = CsvModel(len(data["features"]), len(data["labels"])).experts[index]
                expert.load_state_dict(data["baselines"][name])
                expert.eval()
                call = lambda expert=expert: expert(x[:1])
                flops = costs[index]
            for _ in range(5):
                call()
            timings = []
            for _ in range(30):
                start = time.perf_counter()
                call()
                timings.append((time.perf_counter()-start)*1000)
            results[name] = {"accuracy": (values.argmax(-1) == y).float().mean().item(),
                             "loss": F.cross_entropy(values, y).item(),
                             "flops_per_sample": flops, "median_latency_ms": statistics.median(timings)}
    return {"examples": len(indices), "accuracy": (logits.argmax(-1) == y).float().mean().item(),
            "loss": F.cross_entropy(logits, y).item(), "mode": mode, "seed": data["seed"],
            "results": results, "latency_method": "CPU batch=1, 5 warmups, median of 30; no HTTP",
            "flops_convention": "2 per multiply-add; bias, activations, softmax and dispatch excluded"}
