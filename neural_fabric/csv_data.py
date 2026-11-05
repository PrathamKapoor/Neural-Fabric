"""Validated numeric CSV input and train-only standardization."""
import csv
import math
import random
from pathlib import Path
import torch


def read_csv(path, target=None, features=None):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames
        if not names or any(not name or name.strip() != name for name in names) or len(set(names)) != len(names):
            raise ValueError("CSV must have unique, nonempty headers without surrounding whitespace")
        if target is not None and target not in names:
            raise ValueError("target column not found")
        expected = features or [name for name in names if name != target]
        if not expected or any(name not in names for name in expected):
            raise ValueError("required feature columns missing")
        rows, labels = [], []
        for number, row in enumerate(reader, start=2):
            if None in row or any(row.get(name) in (None, "") for name in expected):
                raise ValueError(f"row {number} has missing or extra fields")
            try:
                values = [float(row[name]) for name in expected]
            except ValueError as exc:
                raise ValueError(f"row {number} has non-numeric features") from exc
            if not all(math.isfinite(v) and abs(v) <= 1e30 for v in values):
                raise ValueError(f"row {number} contains non-finite or out-of-range values")
            if target is not None:
                label = row.get(target)
                if label is None or not label.strip():
                    raise ValueError(f"row {number} has a missing target")
                labels.append(label)
            rows.append(values)
        if not rows:
            raise ValueError("CSV contains no data rows")
    return expected, rows, labels


def partition(labels, seed):
    groups = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, []).append(i)
    if len(groups) < 2 or any(len(indices) < 5 for indices in groups.values()):
        raise ValueError("at least two classes with five rows each are required")
    rng = random.Random(seed)
    train, validation, test = [], [], []
    for indices in groups.values():
        rng.shuffle(indices)
        ntest = max(1, round(len(indices) * 0.2))
        nval = max(1, round(len(indices) * 0.2))
        test.extend(indices[:ntest])
        validation.extend(indices[ntest:ntest+nval])
        train.extend(indices[ntest+nval:])
    for indices in (train, validation, test):
        rng.shuffle(indices)
    return train, validation, test


def fit_preprocessing(rows):
    values = torch.tensor(rows, dtype=torch.float32)
    means = values.mean(dim=0)
    scales = values.std(dim=0, unbiased=False)
    scales = torch.where(scales < 1e-8, torch.ones_like(scales), scales)
    return {"mean": means.tolist(), "scale": scales.tolist()}


def transform(rows, preprocessing):
    values = torch.tensor(rows, dtype=torch.float32)
    transformed = (values - torch.tensor(preprocessing["mean"])) / torch.tensor(preprocessing["scale"])
    if not torch.isfinite(transformed).all():
        raise ValueError("transformed input contains non-finite values")
    return transformed
