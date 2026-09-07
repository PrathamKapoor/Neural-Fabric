import torch, torch.nn as nn
class Primitive(nn.Module):
    def __init__(self, primitive_type, d=64):
        super().__init__(); self.name=primitive_type; self.d=d
class MLP(Primitive):
    def __init__(self, d=64):
        super().__init__("mlp", d); self.net=nn.Sequential(nn.Linear(d,128), nn.ReLU(), nn.Linear(128,3))
    def forward(self, x): return self.net(x)
class CNN(Primitive):
    def __init__(self, d=64):
        super().__init__("cnn", d); self.conv=nn.Conv2d(1,8,3,padding=1); self.pool=nn.AdaptiveAvgPool2d(1); self.fc=nn.Linear(8,3)
    def forward(self, x):
        # x: [batch, d] -> reshape to 1x8x8 for demo
        b=x.shape[0]; x=x.view(b,1,8,8); x=self.conv(x); x=self.pool(x); return self.fc(x.view(b,-1))
class Attention(Primitive):
    def __init__(self, d=64):
        super().__init__("attention", d); self.Wq=nn.Linear(d,d); self.proj=nn.Linear(d,3)
    def forward(self, x):
        q=self.Wq(x); attn=torch.softmax(q@q.transpose(-2,-1), dim=-1); return self.proj(x*(attn.mean(dim=1)))
