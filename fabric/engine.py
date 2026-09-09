from .primitives.mlp import MLP
from .primitives.cnn import CNN
from .primitives.attention import Attention
from .router.router import Router
from .graph.graph import ComputationGraph

class FabricEngine:
    def __init__(self, mode="soft", k=2):
        self.prims = {"mlp": MLP(), "cnn": CNN(), "attention": Attention()}
        self.router = Router(in_dim=64, n_prims=3, mode=mode, k=k)

    def route(self, x, k=None):
        """Route input through the adaptive router with optional k override."""
        return self.router.route(x, k=k)

    def bench(self, x, k=None):
        """Benchmark fabric graph execution against fixed MLP baseline.

        Compute figures are measured from real primitive flops(). Accuracy
        is NOT measured by this method: ``fabric_accuracy_est`` is None and
        ``accuracy_status`` says so explicitly. Never invent it.
        """
        res_fabric = self.predict(x, graph_mode=True, k=k)
        base = self.prims["mlp"]
        compute = res_fabric["compute"]
        base_compute = base.flops()
        savings_pct = max(0.0, round((1.0 - compute / max(base_compute, 1)) * 100.0, 1))
        return {
            "fabric_accuracy_est": None,
            "accuracy_status": "unmeasured: no accuracy evaluation exists",
            "fabric_compute": compute,
            "fabric_path": res_fabric["graph"]["nodes"],
            "baseline_compute": base_compute,
            "savings_pct": savings_pct,
        }

    def predict(self, x, graph_mode=False, k=None):
        # x: feature vector (simplified)
        routing = self.router.route(x, k=k)
        if graph_mode:
            g = ComputationGraph()
            for s in routing["selected"]:
                name = ["mlp","cnn","attention"][s]
                g.add(name, step=len(g.nodes))
            result = g.execute(x, self.prims)
            compute_flops = sum(self.prims[n["primitive"]].flops() for n in g.nodes)
            return {"prediction": result, "routing": routing, "graph": g.to_json(), "compute": compute_flops, "flops": compute_flops}
        else:
            # Soft fusion (weighted average of outputs for demo)
            outputs = {}
            for s in routing["selected"]:
                name = ["mlp","cnn","attention"][s]
                outputs[name] = self.prims[name].forward(x)
            # Simple average over selected
            if outputs:
                num_selected = len(outputs)
                min_len = min(len(v) for v in outputs.values())
                pred = [sum(outputs[n][i] for n in outputs)/num_selected for i in range(min_len)]
            else:
                pred = [0]*3
            compute_flops = sum(self.prims[name].flops() for name in outputs)
            return {"prediction": pred, "routing": routing, "graph": None, "compute": compute_flops, "flops": compute_flops}
