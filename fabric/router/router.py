"""Adaptive Router — outputs routing scores r(x); supports soft, hard, top-k."""
import math, random

class Router:
    def __init__(self, in_dim=64, n_prims=3, mode="soft", seed=0, k=2):
        random.seed(seed); self.in_dim=in_dim; self.n=n_prims; self.mode=mode; self.k=k
        # Simple linear projection + softmax params
        self.W = [[random.gauss(0,0.05) for _ in range(n_prims)] for _ in range(in_dim)]
        self.b = [random.gauss(0,0.05) for _ in range(n_prims)]
    def encode(self, x):
        # x can be feature vector or image stats
        h = [sum(x[i]*self.W[i][j] for i in range(min(len(x), self.in_dim))) + self.b[j] for j in range(self.n)]
        return h
    def topk(self, x_or_k=None, k=None):
        if isinstance(x_or_k, int) and k is None:
            self.k = x_or_k
            return self
        if k is not None:
            self.k = k
        if x_or_k is not None and not isinstance(x_or_k, int):
            eff_k = k if k is not None else getattr(self, "k", 2)
            return self.route(x_or_k, k=eff_k)
        return self
    def __call__(self, x, k=None):
        return self.route(x, k=k)
    def route(self, x, k=None):
        h = self.encode(x)
        # softmax
        mx = max(h); ex = [math.exp(v-mx) for v in h]; s = sum(ex)
        probs = [v/s for v in ex]
        if k is not None:
            k_val = max(1, min(k, self.n))
            idxs = sorted(range(self.n), key=lambda i: probs[i], reverse=True)[:k_val]
            return {"probs": probs, "mode":"topk", "selected": idxs}
        if self.mode == "soft":
            return {"probs": probs, "mode":"soft", "selected": list(range(self.n))}
        elif self.mode == "hard":
            idx = max(range(self.n), key=lambda i: probs[i])
            return {"probs": probs, "mode":"hard", "selected": [idx]}
        elif self.mode == "topk":
            eff_k = getattr(self, "k", 2)
            k_val = max(1, min(eff_k, self.n))
            idxs = sorted(range(self.n), key=lambda i: probs[i], reverse=True)[:k_val]
            return {"probs": probs, "mode":"topk", "selected": idxs}
        else:
            return {"probs": probs, "mode":"soft", "selected": list(range(self.n))}

