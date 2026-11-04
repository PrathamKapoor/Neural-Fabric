"""Offline generated classification dataset, with independently seeded splits."""
import torch

DATASET = "synthetic-linear-v1"


def split(name, seed=42):
    sizes = {"train": 600, "validation": 150, "test": 150}
    offsets = {"train": 0, "validation": 100000, "test": 200000}
    generator = torch.Generator().manual_seed(seed + offsets[name])
    x = torch.randn(sizes[name], 64, generator=generator)
    # Task labels are determined by input features, independently of expert identity.
    scores = torch.stack((x[:, :8].sum(1), x[:, 8:16].sum(1), x[:, 16:24].sum(1)), dim=1)
    return x, scores.argmax(dim=1)
