from .primitives.mlp import MLP
from .primitives.cnn import CNN
from .primitives.attention import Attention
from .router.router import Router
from .graph.graph import ComputationGraph

class FabricEngine:
    def __init__(self, mode="soft"):
        self.prims = {"mlp": MLP(), "cnn": CNN(), "attention": Attention()}
        self.router = Router(in_dim=64, n_prims=3, mode=mode)
    def predict(self, x, graph_mode=False):
        # x: feature vector (simplified)
        routing = self.router.route(x)
        if graph_mode:
            g = ComputationGraph()
            for s in routing["selected"]:
                name = ["mlp","cnn","attention"][s]
                g.add(name, step=len(g.nodes))
            result = g.execute(x, self.prims)
            return {"prediction": result, "routing": routing, "graph": g.to_json(), "compute": sum(self.prims[n["primitive"]].flops() for n in g.nodes)}
        else:
            # Soft fusion (weighted average of outputs for demo)
            outputs = {}
            for s in routing["selected"]:
                name = ["mlp","cnn","attention"][s]
                outputs[name] = self.prims[name].forward(x)
            # Simple average over selected
            if outputs:
                k = len(outputs); pred = [sum(outputs[n][i] for n in outputs)/k for i in range(len(list(outputs.values())[0]))]
            else:
                pred = [0]*3
            return {"prediction": pred, "routing": routing, "graph": None, "compute": sum(self.prims[name].flops() for name in outputs)}
