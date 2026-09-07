"""Synthetic benchmark — controlled structural modes.
Mode A = spatial (CNN preferred), B = relational (Attention preferred), C = dense (MLP preferred).
Used to verify router learns correct computation (RQ4)."""
import random
def make_sample(mode, dim=64, seed=0):
    random.seed(seed)
    x = [random.random() for _ in range(dim)]
    # Inject structural signal: local correlation -> spatial; global correlation -> relational; uniform -> dense
    if mode == "spatial":
        for i in range(8): x[i] = 0.8 + 0.2*random.random()
    elif mode == "relational":
        for i in range(dim): x[i] = 0.5 + 0.4*((i%7)/6)
    elif mode == "dense":
        pass
    return {"x": x, "label": mode, "id": seed}

def dataset(n=150):
    return [make_sample(random.choice(["spatial","relational","dense"]), seed=i) for i in range(n)]
