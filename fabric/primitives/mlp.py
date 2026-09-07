"""MLP primitive — dense feed-forward with configurable hidden dims."""
import math, random

class MLP:
    def __init__(self, in_dim=64, hidden=128, out_dim=3, seed=0):
        random.seed(seed); self.in_dim=in_dim; self.hidden=hidden; self.out_dim=out_dim
        # Small random weights for demo computation
        self.W1 = [[random.gauss(0,0.05) for _ in range(hidden)] for _ in range(in_dim)]
        self.b1 = [random.gauss(0,0.05) for _ in range(hidden)]
        self.W2 = [[random.gauss(0,0.05) for _ in range(out_dim)] for _ in range(hidden)]
        self.b2 = [random.gauss(0,0.05) for _ in range(out_dim)]
        self.name="mlp"
    def forward(self, x, state=None):
        # x: list of floats length in_dim
        h = [max(0, sum(x[i]*self.W1[i][j] for i in range(self.in_dim)) + self.b1[j]) for j in range(self.hidden)]
        y = [sum(h[j]*self.W2[j][k] for j in range(self.hidden)) + self.b2[k] for k in range(self.out_dim)]
        return y
    def params(self): return self.in_dim*self.hidden + self.hidden + self.hidden*self.out_dim + self.out_dim
    def flops(self): return 2*self.in_dim*self.hidden + 2*self.hidden*self.out_dim
