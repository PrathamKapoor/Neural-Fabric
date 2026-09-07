"""Real torch training — NF-02 with actual gradients."""
import torch, torch.nn as nn, torch.optim as optim, random, json, os, time
sys_path = "."
import sys; sys.path.insert(0, sys_path)
from fabric.primitives.torch_primitives import MLP, CNN, Attention
from fabric.router.torch_router import TorchRouter
from evaluation.synthetic_dataset import dataset

def train_epoch(engine, router, opt, data, lambdas=(1,0.1,0.05)):
    total_loss=0; router_optimizer=opt
    for s in data:
        x = torch.tensor(s["x"], dtype=torch.float32).unsqueeze(0)
        # router
        r = router(x)
        # primitives
        m = MLP(); c = CNN(); a = Attention()
        y_m = m(x); y_c = c(x); y_a = a(x)
        # weighted output (soft routing)
        y = r[0,0]*y_m + r[0,1]*y_c + r[0,2]*y_a
        # dummy target = argmax of expected mode (simplified)
        target = torch.tensor([[1,0,0]] if s["label"]=="dense" else ([0,1,0] if s["label"]=="spatial" else [0,0,1]), dtype=torch.float32)
        task = nn.functional.mse_loss(y, target)
        entropy = -torch.sum(r*torch.log(r+1e-9))
        compute = (r[0,0]*m.forward(x).norm() + r[0,1]*c.forward(x).norm() + r[0,2]*a.forward(x).norm()).mean()
        loss = task + lambdas[1]*entropy + lambdas[2]*compute
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    return total_loss/len(data)

if __name__ == "__main__":
    seeds = [1,2,3]; results={"seeds":[],"best_loss":float('inf')}
    for seed in seeds:
        random.seed(seed); torch.manual_seed(seed)
        router = TorchRouter()
        optimizer = optim.Adam(router.parameters(), lr=0.01)
        data = dataset(60)[:30]
        losses=[]
        for epoch in range(20):
            l = train_epoch(None, router, optimizer, data)
            losses.append(round(l,4))
        results["seeds"].append({"seed":seed,"losses":losses,"final_loss":losses[-1]})
        if losses[-1] < results["best_loss"]: results["best_loss"]=losses[-1]
    os.makedirs("experiments/runs", exist_ok=True)
    with open("experiments/runs/nf02_torch_real.json","w") as f:
        json.dump(results, f, indent=2)
    print("Torch training done; best loss:", results["best_loss"], "seeds:", len(results["seeds"]))
