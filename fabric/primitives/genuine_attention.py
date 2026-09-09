"""Genuine single-head softmax attention primitive (NEW, post-v1).

Distinct identity: name "softmax_attention" (never "attention"), so it cannot
be confused with the legacy truncated projection in
fabric.primitives.attention.Attention. Introduced during the v2-corrected
cost-model pass; NO historical experiment used this primitive.

Computes honest Q/K/V attention over a token sequence reshaped from the flat
input: x (length n_tok*d_tok, default 8*8=64) -> n_tok tokens of dim d_tok;
per-token Q/K/V projections; full n_tok×n_tok QK^T score matrix with
1/sqrt(d) scaling; row-wise softmax; attention-weighted sum over V; mean over
query positions as the output vector (length d_tok).

Cost convention (same as cost_model.py): only mul/add counted; softmax
exp/divide/normalization excluded as documented second-order terms.
flops() = qkv_projs + scores(scale incl.) + weighted_sum, derived in code.
"""
import math
import random


class SoftmaxAttention:
    def __init__(self, n_tok=8, d_tok=8, seed=0):
        random.seed(seed)
        self.n_tok = n_tok
        self.d_tok = d_tok
        self.name = "softmax_attention"
        d = d_tok
        self.Wq = [[random.gauss(0, 0.05) for _ in range(d)] for _ in range(d)]
        self.Wk = [[random.gauss(0, 0.05) for _ in range(d)] for _ in range(d)]
        self.Wv = [[random.gauss(0, 0.05) for _ in range(d)] for _ in range(d)]

    def _project(self, tok, W):
        d = self.d_tok
        return [sum(tok[i] * W[i][j] for i in range(d)) for j in range(d)]

    def forward(self, x, state=None):
        n, d = self.n_tok, self.d_tok
        if not isinstance(x, list):
            x = list(x)
        flat = (x + [0.0] * max(0, n * d - len(x)))[: n * d]
        toks = [flat[t * d:(t + 1) * d] for t in range(n)]
        Q = [self._project(t, self.Wq) for t in toks]
        K = [self._project(t, self.Wk) for t in toks]
        V = [self._project(t, self.Wv) for t in toks]
        scale = 1.0 / math.sqrt(d)
        # Full QK^T score matrix (all K/V genuinely used).
        scores = [[sum(Q[t][i] * K[s][i] for i in range(d)) * scale
                   for s in range(n)] for t in range(n)]
        weights = []
        for t in range(n):
            row = scores[t]
            mx = max(row)
            ex = [math.exp(v - mx) for v in row]
            s = sum(ex)
            weights.append([v / s for v in ex])
        # Attention-weighted sum over values.
        out = [[sum(weights[t][s] * V[s][j] for s in range(n))
                for j in range(d)] for t in range(n)]
        # Mean over query positions.
        return [sum(out[t][j] for t in range(n)) / n for j in range(d)]

    def params(self):
        return 3 * self.d_tok * self.d_tok

    def flops(self):
        n, d = self.n_tok, self.d_tok
        qkv = 3 * n * (2 * d * d)
        # scores: n*n dots of dim d (d mult + (d-1) add) + n*n scale mults
        scores = n * n * (2 * d - 1) + n * n
        # weighted sums: n*n*d mults + n*d*(n-1) adds
        vsum = n * n * d + n * d * (n - 1)
        return qkv + scores + vsum
