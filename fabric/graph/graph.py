"""ComputationGraph — explicit DAG of primitives with topological execution."""
class ComputationGraph:
    def __init__(self, nodes=None, edges=None):
        self.nodes = nodes or []
        self.edges = edges or []
    def add(self, name, step=0):
        self.nodes.append({"primitive": name, "step": step})
    def connect(self, a, b):
        self.edges.append([a,b])
    def to_json(self):
        return {"nodes": self.nodes, "edges": self.edges, "halt_step": len(self.nodes)}
    def execute(self, x, primitives, state=None):
        # Sequential execution for demo: execute nodes in order
        result = x
        for n in self.nodes:
            prim = primitives.get(n["primitive"])
            if prim is None: continue
            result = prim.forward(result, state)
        return result
