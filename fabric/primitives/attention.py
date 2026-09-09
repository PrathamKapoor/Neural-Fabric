"""Legacy "attention" primitive — LEGACY/TRUNCATED linear projection.

HISTORICAL PROVENANCE (path B, do not rewrite history):
every experiment up to and including cost-model v1 used THIS implementation
under the name "attention". It does NOT compute conventional attention:
forward() executes exactly ONE d×d matrix-vector product with Wq and returns
the first 8 entries. Wk/Wv are allocated but DEAD weights (never read in
forward); there is no QK^T score matrix, no softmax/normalization, and no
attention-weighted sum over values. The name is retained so historical
artifacts stay reproducible; it must be read as "legacy truncated
projection", not as self-attention. For genuine QKV attention use
fabric.primitives.genuine_attention.SoftmaxAttention (name
"softmax_attention"), a SEPARATE primitive never substituted into historical
experiments.
"""
import math, random
class Attention:
    LEGACY = True
    def __init__(self, d=32, n_heads=1, seed=0):
        random.seed(seed); self.d=d; self.n_heads=n_heads; self.name="attention"
        self.Wq = [[random.gauss(0,0.05) for _ in range(d)] for _ in range(d)]
        self.Wk = [[random.gauss(0,0.05) for _ in range(d)] for _ in range(d)]
        self.Wv = [[random.gauss(0,0.05) for _ in range(d)] for _ in range(d)]
    def forward(self, x, state=None):
        # x: sequence of vectors length d; return aggregated vector
        # Legacy behavior preserved exactly: single Wq projection, truncate.
        if not isinstance(x, list): x = list(x)
        x = (x + [0.0]*max(0, self.d - len(x)))[:self.d]
        proj = [sum(x[i]*self.Wq[i][j] for i in range(self.d)) for j in range(self.d)]
        return proj[:8] if len(proj)>8 else proj
    def params(self): return 3*self.d*self.d
    def flops(self):
        # Corrected: only the executed Wq matvec counts (2*d*d = 2048 for
        # d=32). v1 charged 3*d*d*4 (=12288) for Q/K/V + score math that
        # never executes. Wk/Wv storage is still reported by params().
        from .cost_model import legacy_attention_flops
        return legacy_attention_flops(self.d)
