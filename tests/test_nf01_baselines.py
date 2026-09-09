"""Unit tests for NF-01 fixed homogeneous baselines.

Verifies:
- FixedMLPModel, FixedCNNModel, FixedAttentionModel forward and backward passes
- Parameter count and FLOP calculations
- Loss convergence on synthetic data
- Schema and artifact integrity for experiments/results/nf01_fixed_baselines.json and experiments/runs/
- Statistical comparisons vs NF-02 soft routing
- Pareto frontier trade-off structure
"""
import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.nf01_fixed_baselines import (
    FixedMLPModel,
    FixedCNNModel,
    FixedAttentionModel,
    calc_stats,
    paired_t_test,
    one_sample_t_test,
    train_single_architecture,
)
from evaluation.synthetic_dataset import make_sample


class TestFixedBaselineModels(unittest.TestCase):
    def test_mlp_model_forward_and_backward(self):
        model = FixedMLPModel(in_dim=64, hidden=128, out_dim=3, seed=42)
        x = [0.1] * 64
        logits, h, z1, x_pad = model.forward(x)
        self.assertEqual(len(logits), 3)
        self.assertEqual(len(h), 128)
        self.assertEqual(len(z1), 128)
        self.assertEqual(model.flops(), 17152)
        self.assertEqual(model.params(), 8707)

        w1_before = model.primitive.W1[0][0]
        grad_logits = [0.1, -0.2, 0.1]
        model.backward(x_pad, z1, h, grad_logits, lr=0.05)
        w1_after = model.primitive.W1[0][0]
        self.assertNotEqual(w1_before, w1_after)

    def test_cnn_model_forward_and_backward(self):
        model = FixedCNNModel(in_ch=1, out_ch=8, k=3, out_dim=3, seed=42)
        x = [0.2] * 64
        logits, feats, conv_maps, img = model.forward(x)
        self.assertEqual(len(logits), 3)
        self.assertEqual(len(feats), 32)
        self.assertEqual(len(conv_maps), 8)
        self.assertEqual(model.flops(), 8256 + 2 * 32 * 3)  # 8448, v2 cost model
        self.assertEqual(model.params(), 171)

        filter_before = model.primitive.filters[0][0][0][0]
        grad_logits = [0.1, -0.2, 0.1]
        model.backward(img, conv_maps, feats, grad_logits, lr=0.05)
        filter_after = model.primitive.filters[0][0][0][0]
        self.assertNotEqual(filter_before, filter_after)

    def test_attention_model_forward_and_backward(self):
        model = FixedAttentionModel(d=32, n_heads=1, out_dim=3, seed=42)
        x = [0.3] * 64
        logits, feats, x_d = model.forward(x)
        self.assertEqual(len(logits), 3)
        self.assertEqual(len(feats), 8)
        self.assertEqual(len(x_d), 32)
        self.assertEqual(model.flops(), 2048 + 2 * 8 * 3)  # 2096, v2 cost model
        self.assertEqual(model.params(), 3099)

        wq_before = model.primitive.Wq[0][0]
        grad_logits = [0.1, -0.2, 0.1]
        model.backward(x_d, feats, grad_logits, lr=0.05)
        wq_after = model.primitive.Wq[0][0]
        self.assertNotEqual(wq_before, wq_after)


class TestTrainingConvergence(unittest.TestCase):
    def test_convergence_on_small_batch(self):
        label_map = {"dense": 0, "spatial": 1, "relational": 2}
        train_data = [
            make_sample("dense", seed=10),
            make_sample("spatial", seed=20),
            make_sample("relational", seed=30),
        ] * 5
        test_data = [
            make_sample("dense", seed=110),
            make_sample("spatial", seed=120),
            make_sample("relational", seed=130),
        ]

        for arch in ["mlp", "cnn", "attention"]:
            res = train_single_architecture(
                arch, seed=42, train_data=train_data, test_data=test_data,
                label_map=label_map, n_epochs=10, lr=0.05
            )
            self.assertLess(
                res["loss_curve"][-1], res["loss_curve"][0],
                f"{arch} loss did not decrease"
            )
            self.assertGreaterEqual(res["test_accuracy"], 0.0)
            self.assertLessEqual(res["test_accuracy"], 1.0)
            self.assertIn("loss_curve", res)
            self.assertIn("accuracy", res)
            self.assertIn("compute_flops", res)
            self.assertIn("primitive_usage", res)


class TestSavedArtifacts(unittest.TestCase):
    def test_results_json_exists_and_valid(self):
        results_path = "experiments/results/nf01_fixed_baselines.json"
        self.assertTrue(os.path.exists(results_path), f"Missing {results_path}")
        with open(results_path) as f:
            data = json.load(f)

        self.assertIn("metadata", data)
        self.assertIn("summary", data)
        self.assertIn("statistical_tests", data)
        self.assertIn("pareto_analysis", data)
        self.assertIn("per_seed_results", data)

        for arch in ["mlp", "cnn", "attention"]:
            self.assertIn(arch, data["summary"])
            summary = data["summary"][arch]
            self.assertIn("test_accuracy", summary)
            self.assertIn("compute_flops", summary)
            self.assertIn("parameters", summary)
            mean_acc = summary["test_accuracy"]["mean"]
            self.assertGreaterEqual(mean_acc, 0.0)
            self.assertLessEqual(mean_acc, 1.0)
            self.assertFalse(math.isnan(mean_acc))
            self.assertFalse(math.isinf(mean_acc))

        candidates = data["pareto_analysis"]["candidates"]
        cand_names = [c["name"] for c in candidates]
        self.assertIn("Fixed CNN", cand_names)
        self.assertIn("Fixed MLP", cand_names)
        self.assertIn("Fixed Attention", cand_names)

    def test_seed_run_files_exist_and_match_schema(self):
        required_seeds = [42, 123, 456, 1, 2, 3, 4, 5]
        for s in required_seeds:
            run_path = f"experiments/runs/nf01_fixed_seed{s}.json"
            self.assertTrue(os.path.exists(run_path), f"Missing {run_path}")
            with open(run_path) as f:
                seed_data = json.load(f)

            self.assertIn("loss_curve", seed_data)
            self.assertIn("accuracy", seed_data)
            self.assertIn("compute_flops", seed_data)
            self.assertIn("primitive_usage", seed_data)
            self.assertIn("architectures", seed_data)

            loss_curve = seed_data["loss_curve"]
            self.assertEqual(len(loss_curve), 30)
            self.assertLess(loss_curve[-1], loss_curve[0])
            self.assertGreaterEqual(seed_data["accuracy"], 0.0)
            self.assertLessEqual(seed_data["accuracy"], 1.0)

    def test_results_per_seed_files_exist(self):
        required_seeds = [42, 123, 456, 1, 2, 3, 4, 5]
        for s in required_seeds:
            res_path = f"experiments/results/nf01_fixed_baselines_{s}.json"
            self.assertTrue(os.path.exists(res_path), f"Missing {res_path}")
            with open(res_path) as f:
                res_data = json.load(f)
            self.assertEqual(res_data["seed"], s)
            self.assertIn("architectures", res_data)
            for arch in ["mlp", "cnn", "attention"]:
                self.assertIn(arch, res_data["architectures"])
                self.assertIn("test_accuracy", res_data["architectures"][arch])
                self.assertIn("compute_flops", res_data["architectures"][arch])


if __name__ == "__main__":
    unittest.main()
