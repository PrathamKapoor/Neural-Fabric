"""Versioned tensor-only checkpoint; never use unsafe pickle loading."""
from pathlib import Path
import torch
from .model import RoutedModel
from .data import DATASET


def save(path, model, metadata, baseline_state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "dataset": DATASET, "input_dim": 64,
                "output_dim": 3, "architecture": "linear-smallmlp-widemlp-v1",
                "labels": ["class_0", "class_1", "class_2"],
                "metadata": metadata, "state_dict": model.state_dict(),
                "baseline_state_dict": baseline_state}, path)


def load(path):
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if (data["format_version"], data["dataset"], data["input_dim"],
            data["output_dim"], data["architecture"]) != (1, DATASET, 64, 3, "linear-smallmlp-widemlp-v1"):
            raise ValueError("incompatible checkpoint")
        model = RoutedModel()
        model.load_state_dict(data["state_dict"], strict=True)
        baseline = RoutedModel().experts[1]
        baseline.load_state_dict(data["baseline_state_dict"], strict=True)
        model.baseline = baseline
        model.eval()
        return model, data
    except Exception as exc:
        raise ValueError("Cannot load compatible model checkpoint: " + str(exc)) from exc
