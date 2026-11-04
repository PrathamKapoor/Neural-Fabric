"""End-to-end mixture of three different classifiers."""
import torch
from torch import nn

NAMES = ("linear", "small_mlp", "wide_mlp")
INPUT_DIM = 64
CLASSES = 3
CONVENTION = "Per sample: one multiply and one add count as two FLOPs; bias, activations and softmax excluded."
ROUTER_FLOPS = 2 * INPUT_DIM * len(NAMES)
EXPERT_FLOPS = (2 * 64 * 3, 2 * (64 * 16 + 16 * 3), 2 * (64 * 32 + 32 * 3))


class RoutedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(INPUT_DIM, len(NAMES))
        self.experts = nn.ModuleList([
            nn.Linear(64, 3),
            nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 3)),
            nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3)),
        ])

    def forward(self, x, mode="soft", k=1):
        if mode not in ("soft", "top1", "topk"):
            raise ValueError("mode must be soft, top1, or topk")
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= len(NAMES):
            raise ValueError("k must be an integer from 1 to 3")
        weights = self.router(x).softmax(dim=-1)
        count = len(NAMES) if mode == "soft" else (1 if mode == "top1" else k)
        indices = weights.topk(count, dim=-1).indices
        mask = torch.zeros_like(weights).scatter(1, indices, 1.0)
        selected_weights = weights * mask
        selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True)
        output = x.new_zeros((x.shape[0], CLASSES))
        # Each expert receives only rows selecting it; unselected experts are bypassed.
        for i, expert in enumerate(self.experts):
            rows = mask[:, i].bool()
            if rows.any():
                output[rows] += expert(x[rows]) * selected_weights[rows, i:i+1]
        return output, weights, mask


def compute(mask):
    expert = sum(float(mask[i]) * EXPERT_FLOPS[i] for i in range(len(NAMES)))
    total = ROUTER_FLOPS + expert
    return {"router_flops_estimate": ROUTER_FLOPS, "expert_flops_estimate": int(expert),
            "total_flops_estimate": int(total), "baseline": "fixed small_mlp",
            "baseline_flops_estimate": EXPERT_FLOPS[1],
            "savings_percent": round(100 * (1 - total / EXPERT_FLOPS[1]), 2),
            "convention": CONVENTION}
