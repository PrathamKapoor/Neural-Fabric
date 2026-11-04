import json
import tempfile
import unittest
from pathlib import Path
import torch
from neural_fabric.checkpoint import load
from neural_fabric.model import RoutedModel, compute, EXPERT_FLOPS
from neural_fabric.workflow import train, predict, evaluate


class RoutedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = RoutedModel().eval()
        self.x = torch.randn(4, 64)

    def test_sparse_execution_and_weighted_fusion(self):
        calls = [0, 0, 0]
        handles = []
        for i, expert in enumerate(self.model.experts):
            def record(_module, _args, _output, index=i):
                calls[index] += 1
            handles.append(expert.register_forward_hook(record))
        with torch.no_grad():
            logits, weights, mask = self.model(self.x[:1], "soft")
            expected = sum(weights[:, i:i+1] * expert(self.x[:1])
                           for i, expert in enumerate(self.model.experts))
            self.assertTrue(torch.allclose(logits, expected))
            calls[:] = [0, 0, 0]
            logits, weights, mask = self.model(self.x[:1], "top1")
            selected = int(mask[0].argmax())
            self.assertEqual(calls, [int(i == selected) for i in range(3)])
            self.assertEqual(tuple(logits.shape), (1, 3))
            self.assertAlmostEqual(weights.sum().item(), 1.0, places=6)
            calls[:] = [0, 0, 0]
            _, _, mask = self.model(self.x[:1], "topk", 2)
            self.assertEqual(calls, [int(mask[0, i]) for i in range(3)])
        for handle in handles:
            handle.remove()

    def test_cost_keeps_negative_savings(self):
        result = compute([1, 1, 1])
        self.assertLess(result["savings_percent"], 0)
        self.assertEqual(result["expert_flops_estimate"], sum(EXPERT_FLOPS))
        self.assertEqual(result["total_flops_estimate"],
                         result["expert_flops_estimate"] + result["router_flops_estimate"])

    def test_train_reload_predict_evaluate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.pt")
            train(path, seed=42, epochs=2)
            model, data = load(path)
            values = self.x[0].tolist()
            before = predict(model, values)
            model2, _ = load(path)
            after = predict(model2, values)
            self.assertEqual(before, after)
            self.assertEqual(len(before["probabilities"]), 3)
            self.assertEqual(len(before["routing"]["selected_experts"]), 1)
            self.assertEqual(evaluate(model, data["metadata"]["seed"], repeats=2)["examples"], 150)
            with self.assertRaises(ValueError):
                load(str(Path(directory) / "missing.pt"))
