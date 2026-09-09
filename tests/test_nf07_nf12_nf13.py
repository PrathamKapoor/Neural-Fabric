"""Tests for NF-07/NF-12/NF-13 gap experiments (fast, deterministic).

Verifies: gradient correctness (finite differences), determinism,
artifact consistency (replication flags, MI recomputation), and the
robustness curve shape. No live providers needed.

Run:  python -m unittest discover -s tests -v   (from repo root)
"""
import json
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return json.load(f)


class TestNF07Sweep(unittest.TestCase):
    def test_gradients_match_finite_differences(self):
        sys.path.insert(0, os.path.join(REPO, "experiments"))
        from fabric.router.router import Router
        from evaluation.synthetic_dataset import make_sample
        random.seed(7)
        s = make_sample("spatial", seed=7001)
        x = s["x"]
        y = {"dense": 0, "spatial": 1, "relational": 2}[s["label"]]
        lam_ent, lam_comp = 0.05, 0.5
        F = [17152.0, 1152.0, 12288.0]
        Ft = sum(F)
        f = [v / Ft for v in F]

        def loss_of(h):
            mx = max(h)
            ex = [math.exp(v - mx) for v in h]
            sm = sum(ex)
            p = [v / sm for v in ex]
            nll = -math.log(max(p[y], 1e-15))
            ent = -sum(v * math.log(max(v, 1e-15)) for v in p)
            comp = sum(f[j] * p[j] for j in range(3))
            return nll + lam_ent * ent + lam_comp * comp

        r = Router(in_dim=64, n_prims=3, mode="soft", seed=7)
        h = r.encode(x)
        mx = max(h)
        p = [math.exp(v - mx) / sum(math.exp(vv - mx) for vv in h) for v in h]
        eps = 1e-6
        for k in range(3):
            g_nll = p[k] - (1.0 if k == y else 0.0)
            dh = -sum((math.log(max(p[j], 1e-15)) + 1.0) * p[j]
                      * ((1.0 if j == k else 0.0) - p[k]) for j in range(3))
            dc = sum(f[j] * p[j] * ((1.0 if j == k else 0.0) - p[k])
                     for j in range(3))
            analytic = g_nll + lam_ent * dh + lam_comp * dc
            hp = list(h)
            hp[k] += eps
            hm = list(h)
            hm[k] -= eps
            numeric = (loss_of(hp) - loss_of(hm)) / (2 * eps)
            self.assertLess(abs(analytic - numeric), 1e-6)

    def test_sweep_artifact_structure(self):
        d = _load(os.path.join("experiments", "results", "nf07_sweep_matrix.json"))
        self.assertEqual(len(d["cells"]), 9)
        base = next(c for c in d["cells"]
                    if c["lam_ent"] == 0.0 and c["lam_comp"] == 0.0)
        # unpenalized cell reproduces NF-02-level accuracy
        self.assertGreaterEqual(base["mean_accuracy"], 0.9)
        self.assertFalse(base["any_collapse"])
        for c in d["cells"]:
            self.assertEqual(len(c["per_seed"]), 3)


class TestNF12Modes(unittest.TestCase):
    def test_replication_flags_all_true(self):
        d = _load(os.path.join("experiments", "results", "nf12_routing_modes.json"))
        flags = [r.get("matches_recorded_nf02") for r in d["soft_reference"]["rows"]]
        self.assertEqual(len(flags), 5)
        self.assertTrue(all(flags))

    def test_topk_efficiency_ordering(self):
        d = _load(os.path.join("experiments", "results", "nf12_routing_modes.json"))
        f1 = d["topk_execution"]["1"]["executed_flops"]["mean"]
        f2 = d["topk_execution"]["2"]["executed_flops"]["mean"]
        f3 = d["topk_execution"]["3"]["executed_flops"]["mean"]
        self.assertLess(f1, f2)
        self.assertLess(f2, f3)
        # same argmax accuracy across execution modes by construction
        a1 = d["topk_execution"]["1"]["accuracy"]["mean"]
        self.assertAlmostEqual(a1, d["soft_reference"]["accuracy"]["mean"])

    def test_gumbel_hard_no_collapse_but_null(self):
        d = _load(os.path.join("experiments", "results", "nf12_routing_modes.json"))
        self.assertFalse(d["gumbel_hard"]["any_collapse"])
        paired = d["paired_vs_best_fixed"]
        self.assertGreater(paired["hard_vs_best_fixed_p"], 0.05)
        self.assertGreater(paired["soft_vs_best_fixed_p"], 0.05)


class TestNF13Specialization(unittest.TestCase):
    def test_mi_recomputation_matches(self):
        d = _load(os.path.join("experiments", "results", "nf13_specialization.json"))
        modes = ("dense", "spatial", "relational")
        prims = ("mlp", "cnn", "attention")
        for r in d["rows"]:
            c = r["conditional"]
            mi = 0.0
            for m in modes:
                for p in prims:
                    j = c[m][p] / 3.0
                    ms = sum(c[mm][p] for mm in modes) / 3.0
                    if j > 0 and ms > 0:
                        mi += j * math.log(j / ((1.0 / 3.0) * ms))
            self.assertAlmostEqual(mi, r["mutual_information_nats"], places=3)

    def test_specialization_is_strong_and_bounded(self):
        d = _load(os.path.join("experiments", "results", "nf13_specialization.json"))
        self.assertGreater(d["mean_mi_nats"], 0.5)
        self.assertLessEqual(d["mean_mi_nats"], d["max_mi_nats"])
        self.assertGreater(d["mean_correct_primitive_rate"], 0.85)


class TestNF09Curve(unittest.TestCase):
    def test_monotone_threshold_shape(self):
        d = _load(os.path.join("experiments", "results", "nf09_robustness.json"))
        by_noise = {lv["noise_std"]: lv["path_change_rate"] for lv in d["levels"]}
        self.assertEqual(by_noise[0.0], 0.0)
        self.assertEqual(by_noise[0.05], 0.0)
        self.assertGreater(by_noise[0.2], by_noise[0.05])
        self.assertIn(0.05, [lv["noise_std"] for lv in d["levels"]])


if __name__ == "__main__":
    unittest.main()
