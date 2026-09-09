"""Tests for NF-14 hierarchical transfer (fast, deterministic)."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return json.load(f)


class TestCompositionalBenchmark(unittest.TestCase):
    def test_modes_and_labels(self):
        from evaluation.compositional import hierarchical_dataset, make_hierarchical
        s = make_hierarchical("A", "spatial", seed=3)
        self.assertEqual(s["label"], "spatial")
        self.assertEqual(s["coarse"], "A")
        self.assertEqual(len(s["x"]), 64)
        d = hierarchical_dataset(n_per_cell=2, seed=1)
        self.assertEqual(len(d), 12)
        modes = {(t["coarse"], t["label"]) for t in d}
        self.assertEqual(len(modes), 6)

    def test_signal_moves_with_coarse_tag(self):
        from evaluation.compositional import make_hierarchical
        a = make_hierarchical("A", "spatial", seed=9)["x"]
        b = make_hierarchical("B", "spatial", seed=9)["x"]
        # spatial signal (values >0.8) lives in dims 0-8 for coarse A...
        self.assertTrue(all(v > 0.8 for v in a[:8]))
        # ...and in dims 32-40 for coarse B
        self.assertTrue(all(v > 0.8 for v in b[32:40]))
        # determinism: same triple reproduces exactly
        a2 = make_hierarchical("A", "spatial", seed=9)["x"]
        self.assertEqual(a, a2)


class TestNF14Artifact(unittest.TestCase):
    def test_transfer_refuted_and_null_persists(self):
        d = _load(os.path.join("experiments", "results",
                               "nf14_hierarchical.json"))
        self.assertTrue(d["transfer_refuted"])
        self.assertTrue(d["hier_null_persists"])
        self.assertLess(d["paired"]["transfer_vs_soft_p"], 0.05)
        self.assertGreaterEqual(d["paired"]["soft_vs_best_fixed_p"], 0.05)
        self.assertEqual(d["seeds"], [1, 2, 3, 4, 5])

    def test_hier_training_deterministic(self):
        sys.path.insert(0, os.path.join(REPO, "experiments"))
        from nf14_hierarchical import eval_router, train_hier_soft
        from evaluation.compositional import hierarchical_dataset
        test = hierarchical_dataset(n_per_cell=3, seed=77)
        r1 = eval_router(train_hier_soft(2), test)
        r2 = eval_router(train_hier_soft(2), test)
        self.assertEqual(r1["test_accuracy"], r2["test_accuracy"])
        self.assertEqual(r1["exec_flops"], r2["exec_flops"])


if __name__ == "__main__":
    unittest.main()
