"""Hierarchical composition benchmark - 2-level input structure.

Coarse tag: WHICH half carries the signal (A: dims 0-31, B: dims 32-63).
Fine tag: WHAT the signal is (dense/spatial/relational, same structure as
make_sample but placed in the coarse half; other half is uniform noise).
Label = fine type. The router must generalize specialization beyond the
fixed signal locations of the flat benchmark - a transfer test for RQ-HIER.
"""
import random


def make_hierarchical(coarse, fine, dim=64, seed=0):
    random.seed(seed)
    half = dim // 2
    x = [random.random() for _ in range(dim)]
    lo, hi = (0, half) if coarse == "A" else (half, dim)
    width = hi - lo
    if fine == "spatial":
        for i in range(lo, lo + 8):
            x[i] = 0.8 + 0.2 * random.random()
    elif fine == "relational":
        for i in range(lo, hi):
            x[i] = 0.5 + 0.4 * (((i - lo) % 7) / 6)
    elif fine == "dense":
        pass
    _ = width
    return {"x": x, "label": fine, "coarse": coarse, "id": seed}


def hierarchical_dataset(n_per_cell=25, seed=0):
    rng = random.Random(seed)
    cells = [(c, f) for c in ("A", "B") for f in ("dense", "spatial", "relational")]
    data = []
    k = 0
    for _ in range(n_per_cell):
        for c, f in cells:
            data.append(make_hierarchical(c, f, seed=seed * 1000 + k))
            k += 1
    rng.shuffle(data)
    return data
