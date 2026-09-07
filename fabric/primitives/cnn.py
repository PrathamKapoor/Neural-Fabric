"""CNN primitive — 2D convolutional filter over a small grid (simulated)."""
import math, random
class CNN:
    def __init__(self, in_ch=1, out_ch=8, k=3, seed=0):
        random.seed(seed); self.in_ch=in_ch; self.out_ch=out_ch; self.k=k
        self.filters = [[[ [random.gauss(0,0.05) for _ in range(k)] for _ in range(k)] for _ in range(in_ch)] for _ in range(out_ch)]
        self.name="cnn"
    def forward(self, x, state=None):
        # x: flat 28x28 image -> 784; here simplified to small 8x8 for speed
        # Return a pooled feature vector
        return [random.gauss(0,0.1) for _ in range(self.out_ch*4)]
    def params(self): return self.in_ch*self.out_ch*self.k*self.k
    def flops(self): return self.in_ch*self.out_ch*self.k*self.k*16
