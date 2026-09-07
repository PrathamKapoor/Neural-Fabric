import torch, torch.nn as nn
class TorchRouter(nn.Module):
    def __init__(self, d=64, n=3):
        super().__init__(); self.lin=nn.Linear(d,n)
    def forward(self, x):
        return torch.softmax(self.lin(x), dim=-1)
