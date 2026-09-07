"""Self-Attention primitive — single-head scaled dot-product on small sequences."""
import math, random
class Attention:
    def __init__(self, d=32, n_heads=1, seed=0):
        random.seed(seed); self.d=d; self.n_heads=n_heads; self.name="attention"
        self.Wq = [[random.gauss(0,0.05) for _ in range(d)] for _ in range(d)]
        self.Wk = [[random.gauss(0,0.05) for _ in range(d)] for _ in range(d)]
        self.Wv = [[random.gauss(0,0.05) for _ in range(d)] for _ in range(d)]
    def forward(self, x, state=None):
        # x: sequence of vectors length d; return aggregated vector
        # Simplified: mean of projected vectors
        proj = [sum(x[i]*self.Wq[i][j] for i in range(self.d)) for j in range(self.d)]
        return proj[:8] if len(proj)>8 else proj
    def params(self): return 3*self.d*self.d
    def flops(self): return 3*self.d*self.d*4
