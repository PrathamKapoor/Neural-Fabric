"""Dedicated unit test suite for Neural Fabric.

Covers:
- fabric/primitives (MLP, CNN, Attention): deterministic forward, params/flops, input shape edge cases
- fabric/router (Router): soft/hard/top-k modes, k parameter (k=1, k=max, invalid k), numerical stability
- fabric/graph (ComputationGraph): DAG execution, empty graph edge case, JSON serialization
- fabric/engine (FabricEngine): predict, routing, bench/compute accounting, k-parameter plumbing

Pure-Python implementation with zero PyTorch dependency in tests.
Run with:
    python -m pytest tests/test_fabric.py -v
or:
    python -m unittest tests/test_fabric.py -v
"""
import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fabric.primitives.mlp import MLP
from fabric.primitives.cnn import CNN
from fabric.primitives.attention import Attention
from fabric.router.router import Router
from fabric.graph.graph import ComputationGraph
from fabric.engine import FabricEngine


class TestPrimitives(unittest.TestCase):
    """Unit tests for heterogeneous computational primitives (MLP, CNN, Attention)."""

    def test_mlp_forward_deterministic(self):
        """MLP produces identical outputs across calls and across identical seeds."""
        x = [0.05 * (i % 7) for i in range(64)]
        mlp1 = MLP(in_dim=64, hidden=128, out_dim=3, seed=42)
        mlp2 = MLP(in_dim=64, hidden=128, out_dim=3, seed=42)
        out1 = mlp1.forward(x)
        out2 = mlp1.forward(x)
        out_seed = mlp2.forward(x)

        self.assertEqual(out1, out2)
        self.assertEqual(out1, out_seed)
        self.assertIsInstance(out1, list)
        self.assertEqual(len(out1), 3)
        self.assertTrue(all(isinstance(v, float) for v in out1))

    def test_mlp_seed_differentiation(self):
        """Different seeds produce distinct MLP weight initializations."""
        x = [0.1] * 64
        mlp1 = MLP(seed=1)
        mlp2 = MLP(seed=2)
        self.assertNotEqual(mlp1.forward(x), mlp2.forward(x))

    def test_mlp_flops_and_params(self):
        """MLP correctly calculates parameter count and FLOP requirements."""
        # Default: in_dim=64, hidden=128, out_dim=3
        mlp_default = MLP()
        expected_params = 64 * 128 + 128 + 128 * 3 + 3
        expected_flops = 2 * 64 * 128 + 2 * 128 * 3
        self.assertEqual(mlp_default.params(), expected_params)
        self.assertEqual(mlp_default.flops(), expected_flops)
        self.assertEqual(mlp_default.params(), 8707)
        self.assertEqual(mlp_default.flops(), 17152)

        # Custom dimensions
        mlp_custom = MLP(in_dim=16, hidden=32, out_dim=2)
        self.assertEqual(mlp_custom.params(), 16 * 32 + 32 + 32 * 2 + 2)
        self.assertEqual(mlp_custom.flops(), 2 * 16 * 32 + 2 * 32 * 2)

    def test_mlp_input_padding_and_truncation(self):
        """MLP handles varying input lengths via padding or truncation."""
        mlp = MLP(in_dim=64, out_dim=3, seed=0)
        # Exact length
        out_exact = mlp.forward([0.1] * 64)
        self.assertEqual(len(out_exact), 3)

        # Shorter input (padded with 0.0)
        out_short = mlp.forward([0.1] * 10)
        self.assertEqual(len(out_short), 3)

        # Longer input (truncated)
        out_long = mlp.forward([0.1] * 128)
        self.assertEqual(len(out_long), 3)

        # Empty input
        out_empty = mlp.forward([])
        self.assertEqual(len(out_empty), 3)

    def test_cnn_forward_deterministic(self):
        """CNN primitive produces deterministic output given the same input."""
        x = [0.02 * (i % 9) for i in range(64)]
        cnn1 = CNN(in_ch=1, out_ch=8, k=3, seed=42)
        cnn2 = CNN(in_ch=1, out_ch=8, k=3, seed=42)
        out1 = cnn1.forward(x)
        out2 = cnn1.forward(x)
        out_seed = cnn2.forward(x)

        self.assertEqual(out1, out2)
        self.assertEqual(out1, out_seed)
        self.assertIsInstance(out1, list)
        self.assertEqual(len(out1), 32)  # out_ch * 4 pooled values (8 * 4)
        self.assertTrue(all(isinstance(v, (int, float)) for v in out1))

    def test_cnn_flops_and_params(self):
        """CNN correctly reports parameter count and FLOPs."""
        # Default: in_ch=1, out_ch=8, k=3
        cnn_default = CNN()
        expected_params = 1 * 8 * 3 * 3
        expected_flops = 1 * 8 * 3 * 3 * 16
        self.assertEqual(cnn_default.params(), expected_params)
        self.assertEqual(cnn_default.flops(), expected_flops)
        self.assertEqual(cnn_default.params(), 72)
        self.assertEqual(cnn_default.flops(), 1152)

        # Custom dimensions
        cnn_custom = CNN(in_ch=2, out_ch=4, k=5)
        self.assertEqual(cnn_custom.params(), 2 * 4 * 5 * 5)
        self.assertEqual(cnn_custom.flops(), 2 * 4 * 5 * 5 * 16)

    def test_cnn_input_formats_and_edge_cases(self):
        """CNN handles flat lists, 2D/3D grids, and non-standard input sizes."""
        cnn = CNN(in_ch=1, out_ch=8, k=3, seed=0)

        # 1D flat list
        out_flat = cnn.forward([0.5] * 64)
        self.assertEqual(len(out_flat), 32)

        # Short input (padded)
        out_short = cnn.forward([0.5] * 16)
        self.assertEqual(len(out_short), 32)

        # Empty input
        out_empty = cnn.forward([])
        self.assertEqual(len(out_empty), 32)

        # Tuple input
        out_tuple = cnn.forward(tuple([0.5] * 64))
        self.assertEqual(len(out_tuple), 32)

        # 2D grid input (1 channel of 8x8)
        grid_2d = [[0.1 * c for c in range(8)] for _ in range(8)]
        out_2d = cnn.forward(grid_2d)
        self.assertEqual(len(out_2d), 32)

    def test_attention_forward_deterministic(self):
        """Self-attention primitive is deterministic across forward passes."""
        x = [0.03 * (i % 5) for i in range(32)]
        att1 = Attention(d=32, n_heads=1, seed=42)
        att2 = Attention(d=32, n_heads=1, seed=42)
        out1 = att1.forward(x)
        out2 = att1.forward(x)
        out_seed = att2.forward(x)

        self.assertEqual(out1, out2)
        self.assertEqual(out1, out_seed)
        self.assertIsInstance(out1, list)
        self.assertEqual(len(out1), 8)
        self.assertTrue(all(isinstance(v, float) for v in out1))

    def test_attention_flops_and_params(self):
        """Attention reports correct FLOP count and parameters."""
        att_default = Attention()
        expected_params = 3 * 32 * 32
        expected_flops = 3 * 32 * 32 * 4
        self.assertEqual(att_default.params(), expected_params)
        self.assertEqual(att_default.flops(), expected_flops)
        self.assertEqual(att_default.params(), 3072)
        self.assertEqual(att_default.flops(), 12288)

        # Custom dimension
        att_custom = Attention(d=16)
        self.assertEqual(att_custom.params(), 3 * 16 * 16)
        self.assertEqual(att_custom.flops(), 3 * 16 * 16 * 4)

    def test_attention_input_padding_and_truncation(self):
        """Attention pads short sequences and truncates long sequences."""
        att = Attention(d=32, seed=0)
        # Short input
        out_short = att.forward([0.2] * 8)
        self.assertEqual(len(out_short), 8)

        # Long input
        out_long = att.forward([0.2] * 64)
        self.assertEqual(len(out_long), 8)

        # Empty input
        out_empty = att.forward([])
        self.assertEqual(len(out_empty), 8)

    def test_primitives_common_interface(self):
        """All primitives expose forward, flops, params, and name attributes."""
        primitives = [MLP(), CNN(), Attention()]
        for p in primitives:
            self.assertTrue(hasattr(p, "forward"))
            self.assertTrue(hasattr(p, "flops"))
            self.assertTrue(hasattr(p, "params"))
            self.assertTrue(hasattr(p, "name"))
            self.assertIsInstance(p.name, str)
            self.assertGreater(p.params(), 0)
            self.assertGreater(p.flops(), 0)


class TestRouter(unittest.TestCase):
    """Unit tests for adaptive Router: soft, hard, top-k modes, and k-parameter."""

    def setUp(self):
        self.x = [0.1 * (i % 6) for i in range(64)]

    def test_router_probabilities_validity(self):
        """Routing scores form a valid probability distribution."""
        router = Router(in_dim=64, n_prims=3, mode="soft", seed=42)
        routing = router.route(self.x)
        probs = routing["probs"]

        self.assertEqual(len(probs), 3)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
        self.assertTrue(math.isclose(sum(probs), 1.0, rel_tol=1e-6))

    def test_router_soft_mode(self):
        """Soft mode selects all available computational primitives."""
        router = Router(in_dim=64, n_prims=3, mode="soft", seed=0)
        res = router.route(self.x)

        self.assertEqual(res["mode"], "soft")
        self.assertEqual(res["selected"], [0, 1, 2])
        self.assertEqual(len(res["selected"]), 3)

    def test_router_hard_mode(self):
        """Hard mode selects exactly the single argmax primitive."""
        router = Router(in_dim=64, n_prims=3, mode="hard", seed=0)
        res = router.route(self.x)

        self.assertEqual(res["mode"], "hard")
        self.assertEqual(len(res["selected"]), 1)
        expected_idx = max(range(3), key=lambda i: res["probs"][i])
        self.assertEqual(res["selected"], [expected_idx])

    def test_router_topk_mode_default_k(self):
        """Top-k mode selects k highest-probability primitives in descending order."""
        router = Router(in_dim=64, n_prims=3, mode="topk", seed=0, k=2)
        res = router.route(self.x)

        self.assertEqual(res["mode"], "topk")
        self.assertEqual(len(res["selected"]), 2)
        p0 = res["probs"][res["selected"][0]]
        p1 = res["probs"][res["selected"][1]]
        self.assertGreaterEqual(p0, p1)

    def test_router_k_parameter_k1(self):
        """Top-k mode with k=1 selects exactly one best primitive."""
        router = Router(in_dim=64, n_prims=3, mode="topk", seed=0, k=1)
        res = router.route(self.x)

        self.assertEqual(len(res["selected"]), 1)
        expected_idx = max(range(3), key=lambda i: res["probs"][i])
        self.assertEqual(res["selected"], [expected_idx])

        # Also via explicit route(x, k=1)
        router_def = Router(in_dim=64, n_prims=3, mode="soft", seed=0)
        res_override = router_def.route(self.x, k=1)
        self.assertEqual(res_override["mode"], "topk")
        self.assertEqual(res_override["selected"], [expected_idx])

    def test_router_k_parameter_k_max(self):
        """Top-k with k=n_prims or k exceeding n_prims clamps to all primitives."""
        router = Router(in_dim=64, n_prims=3, mode="topk", seed=0, k=3)
        res_k3 = router.route(self.x)
        self.assertEqual(len(res_k3["selected"]), 3)
        self.assertEqual(set(res_k3["selected"]), {0, 1, 2})

        res_k_excess = router.route(self.x, k=99)
        self.assertEqual(len(res_k_excess["selected"]), 3)
        self.assertEqual(set(res_k_excess["selected"]), {0, 1, 2})

    def test_router_invalid_k_clamping(self):
        """Invalid k values (k=0, negative k) are clamped to minimum 1."""
        router = Router(in_dim=64, n_prims=3, mode="topk", seed=0)
        res_zero = router.route(self.x, k=0)
        self.assertEqual(len(res_zero["selected"]), 1)

        res_neg = router.route(self.x, k=-5)
        self.assertEqual(len(res_neg["selected"]), 1)

    def test_router_deterministic(self):
        """Router outputs are deterministic given identical seed and input."""
        r1 = Router(seed=99)
        r2 = Router(seed=99)
        res1 = r1.route(self.x)
        res2 = r2.route(self.x)
        self.assertEqual(res1, res2)

    def test_router_topk_method_and_call(self):
        """Router supports .topk() mutating/routing and __call__ interface."""
        router = Router(in_dim=64, n_prims=3, mode="topk", seed=0, k=2)

        ret = router.topk(1)
        self.assertEqual(ret, router)
        self.assertEqual(router.k, 1)

        res_topk_call = router.topk(self.x, k=2)
        self.assertEqual(len(res_topk_call["selected"]), 2)

        res_call = router(self.x, k=1)
        self.assertEqual(len(res_call["selected"]), 1)


class TestComputationGraph(unittest.TestCase):
    """Unit tests for ComputationGraph DAG execution and JSON serialization."""

    def setUp(self):
        self.prims = {"mlp": MLP(seed=0), "cnn": CNN(seed=0), "attention": Attention(seed=0)}
        self.x = [0.1] * 64

    def test_empty_graph(self):
        """Empty graph executes as identity pass-through and serializes cleanly."""
        g = ComputationGraph()
        self.assertEqual(g.nodes, [])
        self.assertEqual(g.edges, [])

        res = g.execute(self.x, self.prims)
        self.assertEqual(res, self.x)

        d = g.to_json()
        self.assertEqual(d, {"nodes": [], "edges": [], "halt_step": 0})
        serialized = json.dumps(d)
        self.assertIn('"halt_step": 0', serialized)

    def test_graph_add_and_connect(self):
        """Adding nodes and edges reflects accurately in graph structure."""
        g = ComputationGraph()
        g.add("cnn", step=0)
        g.add("mlp", step=1)
        g.connect(0, 1)

        self.assertEqual(len(g.nodes), 2)
        self.assertEqual(g.nodes[0], {"primitive": "cnn", "step": 0})
        self.assertEqual(g.nodes[1], {"primitive": "mlp", "step": 1})
        self.assertEqual(g.edges, [[0, 1]])

    def test_graph_json_serialization(self):
        """ComputationGraph to_json produces valid JSON-serializable dictionary."""
        g = ComputationGraph()
        g.add("cnn", step=0)
        g.add("attention", step=1)
        g.add("mlp", step=2)
        g.connect(0, 1)
        g.connect(1, 2)

        payload = g.to_json()
        self.assertEqual(payload["halt_step"], 3)
        self.assertEqual(len(payload["nodes"]), 3)
        self.assertEqual(len(payload["edges"]), 2)

        encoded = json.dumps(payload)
        decoded = json.loads(encoded)
        self.assertEqual(decoded, payload)

    def test_graph_topological_execution(self):
        """Sequential DAG execution correctly composes heterogeneous primitives."""
        g = ComputationGraph()
        g.add("cnn", step=0)
        g.add("attention", step=1)
        g.add("mlp", step=2)
        g.connect(0, 1)
        g.connect(1, 2)

        graph_result = g.execute(self.x, self.prims)

        step1 = self.prims["cnn"].forward(self.x)
        step2 = self.prims["attention"].forward(step1)
        step3 = self.prims["mlp"].forward(step2)

        self.assertEqual(graph_result, step3)

    def test_graph_unknown_primitive_skipped(self):
        """Nodes referencing missing primitive names are safely skipped."""
        g = ComputationGraph()
        g.add("nonexistent_primitive", step=0)
        g.add("mlp", step=1)

        res = g.execute(self.x, self.prims)
        expected = self.prims["mlp"].forward(self.x)
        self.assertEqual(res, expected)

    def test_graph_deterministic_execution(self):
        """Repeated graph execution yields deterministic output."""
        g = ComputationGraph()
        g.add("attention", step=0)
        g.add("mlp", step=1)

        res1 = g.execute(self.x, self.prims)
        res2 = g.execute(self.x, self.prims)
        self.assertEqual(res1, res2)


class TestFabricEngine(unittest.TestCase):
    """Unit tests for FabricEngine predict, routing, benching, and k-parameter plumbing."""

    def setUp(self):
        self.x = [0.1 * (i % 5) for i in range(64)]

    def test_engine_init(self):
        """FabricEngine initializes with standard primitives and adaptive router."""
        engine = FabricEngine(mode="soft", k=2)
        self.assertIn("mlp", engine.prims)
        self.assertIn("cnn", engine.prims)
        self.assertIn("attention", engine.prims)
        self.assertIsInstance(engine.router, Router)
        self.assertEqual(engine.router.mode, "soft")
        self.assertEqual(engine.router.k, 2)

    def test_engine_predict_soft_mode(self):
        """Soft mode predict performs fusion and reports compute FLOPs."""
        engine = FabricEngine(mode="soft")
        res = engine.predict(self.x, graph_mode=False)

        self.assertIn("prediction", res)
        self.assertIn("routing", res)
        self.assertIn("graph", res)
        self.assertIn("compute", res)
        self.assertIn("flops", res)

        self.assertIsNone(res["graph"])
        self.assertEqual(res["compute"], res["flops"])
        self.assertEqual(res["routing"]["mode"], "soft")
        self.assertEqual(len(res["routing"]["selected"]), 3)

        total_flops = sum(p.flops() for p in engine.prims.values())
        self.assertEqual(res["compute"], total_flops)
        self.assertEqual(len(res["prediction"]), 3)

    def test_engine_predict_graph_mode(self):
        """Graph mode predict constructs ComputationGraph and reports node FLOPs."""
        engine = FabricEngine(mode="topk", k=2)
        res = engine.predict(self.x, graph_mode=True)

        self.assertIsNotNone(res["graph"])
        self.assertIn("nodes", res["graph"])
        self.assertIn("edges", res["graph"])
        self.assertIn("halt_step", res["graph"])
        self.assertEqual(len(res["graph"]["nodes"]), 2)

        expected_flops = sum(engine.prims[n["primitive"]].flops() for n in res["graph"]["nodes"])
        self.assertEqual(res["compute"], expected_flops)
        self.assertEqual(res["compute"], res["flops"])

    def test_engine_k_parameter_plumbing(self):
        """k-parameter is properly plumbed through predict to router and graph."""
        engine = FabricEngine(mode="topk", k=2)

        # Override k=1
        res_k1 = engine.predict(self.x, graph_mode=True, k=1)
        self.assertEqual(len(res_k1["routing"]["selected"]), 1)
        self.assertEqual(len(res_k1["graph"]["nodes"]), 1)
        active_prim = res_k1["graph"]["nodes"][0]["primitive"]
        self.assertEqual(res_k1["compute"], engine.prims[active_prim].flops())

        # Override k=3
        res_k3 = engine.predict(self.x, graph_mode=True, k=3)
        self.assertEqual(len(res_k3["routing"]["selected"]), 3)
        self.assertEqual(len(res_k3["graph"]["nodes"]), 3)
        total_flops = sum(p.flops() for p in engine.prims.values())
        self.assertEqual(res_k3["compute"], total_flops)

        # Check edge case k=max (e.g. k=10)
        res_k10 = engine.predict(self.x, graph_mode=True, k=10)
        self.assertEqual(len(res_k10["routing"]["selected"]), 3)

        # Check edge case invalid k (e.g. k=0 or k=-1)
        res_k0 = engine.predict(self.x, graph_mode=True, k=0)
        self.assertEqual(len(res_k0["routing"]["selected"]), 1)

    def test_engine_init_k_respected(self):
        """k specified at engine construction is respected during default predict."""
        engine_k1 = FabricEngine(mode="topk", k=1)
        res1 = engine_k1.predict(self.x, graph_mode=True)
        self.assertEqual(len(res1["routing"]["selected"]), 1)

        engine_k3 = FabricEngine(mode="topk", k=3)
        res3 = engine_k3.predict(self.x, graph_mode=True)
        self.assertEqual(len(res3["routing"]["selected"]), 3)

    def test_engine_deterministic_predict(self):
        """FabricEngine predict is strictly deterministic for identical inputs."""
        engine = FabricEngine(mode="topk", k=2)
        res1 = engine.predict(self.x, graph_mode=True)
        res2 = engine.predict(self.x, graph_mode=True)

        self.assertEqual(res1["prediction"], res2["prediction"])
        self.assertEqual(res1["routing"], res2["routing"])
        self.assertEqual(res1["graph"], res2["graph"])
        self.assertEqual(res1["compute"], res2["compute"])

    def test_engine_route_and_bench_operations(self):
        """Routing and benching/compute accounting operations through engine with k parameter."""
        engine = FabricEngine(mode="topk", k=2)

        # Route directly through engine
        routing_ref = engine.router.route(self.x, k=2)
        res_route = engine.route(self.x, k=2)
        self.assertEqual(res_route["selected"], routing_ref["selected"])
        self.assertEqual(len(res_route["selected"]), 2)
        self.assertIn("probs", res_route)

        # Route with k overrides: k=1, k=max, invalid k
        res_route_k1 = engine.route(self.x, k=1)
        self.assertEqual(len(res_route_k1["selected"]), 1)

        res_route_kmax = engine.route(self.x, k=10)
        self.assertEqual(len(res_route_kmax["selected"]), 3)

        res_route_kinvalid = engine.route(self.x, k=0)
        self.assertEqual(len(res_route_kinvalid["selected"]), 1)

        # Bench directly through engine
        res_bench_k1 = engine.bench(self.x, k=1)
        self.assertIn("fabric_compute", res_bench_k1)
        self.assertIn("baseline_compute", res_bench_k1)
        self.assertIn("savings_pct", res_bench_k1)
        self.assertIn("fabric_path", res_bench_k1)
        self.assertEqual(len(res_bench_k1["fabric_path"]), 1)

        # Verify compute matches predict() in graph mode
        predict_k1 = engine.predict(self.x, graph_mode=True, k=1)
        self.assertEqual(res_bench_k1["fabric_compute"], predict_k1["compute"])
        self.assertEqual(res_bench_k1["baseline_compute"], engine.prims["mlp"].flops())
        self.assertGreater(res_bench_k1["savings_pct"], 0.0)

        # Bench with k=3
        res_bench_k3 = engine.bench(self.x, k=3)
        self.assertEqual(len(res_bench_k3["fabric_path"]), 3)
        predict_k3 = engine.predict(self.x, graph_mode=True, k=3)
        self.assertEqual(res_bench_k3["fabric_compute"], predict_k3["compute"])


if __name__ == "__main__":
    unittest.main()
