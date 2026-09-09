"""Tests for the robustness pass: NF-07 10-seed, NF-01 10-seed, NX-ATTN-01.

- train_cell provenance schema (routing distribution, loss, status).
- Genuine-attention backward gradient correctness (finite differences).
- Genuine model cost identity (5104 FLOPs / 219 params).
- New artifact files exist with required provenance keys.
"""
import copy
import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return json.load(f)


class TestNF07Provenance(unittest.TestCase):
    def test_train_cell_schema(self):
        from nf07_sweep import train_cell
        r = train_cell(1, 0.0, 0.5)
        for k in ("seed", "status", "test_accuracy", "mean_entropy",
                  "test_selection_counts", "test_selection_frequencies",
                  "train_nll_final", "train_objective_final",
                  "top1_executed_flops", "collapse_detected"):
            self.assertIn(k, r)
        self.assertEqual(r["status"], "completed")
        self.assertEqual(sum(r["test_selection_counts"].values()), 50)
        self.assertAlmostEqual(sum(r["test_selection_frequencies"].values()),
                               1.0, places=3)

    def test_10seed_artifact(self):
        d = _load("experiments/results/nf07_sweep_10seed_corrected.json")
        self.assertEqual(d["seeds"], list(range(1, 11)))
        self.assertEqual(d["failures"], [])
        self.assertEqual(d["cost_model"], "v2-corrected")
        self.assertEqual(len(d["cells"]), 9)
        for c in d["cells"]:
            self.assertEqual(c["n_completed"], 10)
            self.assertEqual(c["n_failed"], 0)

    def test_statistics_artifact(self):
        d = _load("experiments/results/nf07_10seed_statistics.json")
        c = d["contrasts_vs_unpenalized_baseline"]["le=0.0_lc=2.0_vs_baseline"]
        # compute reduction on all 10 seeds; accuracy cost small but real
        self.assertEqual(c["flops_paired"]["n_seeds_lower"], 10)
        self.assertLess(c["flops_paired"]["p_two_tailed"], 0.01)
        self.assertLess(c["accuracy_paired"]["mean_diff"], 0.0)
        self.assertGreater(c["accuracy_paired"]["mean_diff"], -0.10)


class TestNF01TenSeed(unittest.TestCase):
    def test_artifact_schema(self):
        d = _load("experiments/results/nf01_fixed_10seed_corrected.json")
        self.assertEqual(d["config"]["seeds"], list(range(1, 11)))
        self.assertEqual(d["failures"], [])
        self.assertEqual(d["cost_model"], "v2-corrected")
        for a in ("mlp", "cnn", "attention"):
            self.assertEqual(d["summary"][a]["test_accuracy"]["n"], 10)
        # efficiency ordering stable: legacy projection cheapest per point
        apk = {a: d["summary"][a]["accuracy_per_kflop"]["mean"]
               for a in ("mlp", "cnn", "attention")}
        self.assertGreater(apk["attention"], apk["cnn"])
        self.assertGreater(apk["cnn"], apk["mlp"])
        self.assertEqual(d["efficiency_winner_mean"], "attention")


class TestGenuineAttention(unittest.TestCase):
    def test_cost_identity(self):
        from nx_attn_genuine import FixedGenuineAttentionModel
        m = FixedGenuineAttentionModel(seed=0)
        self.assertEqual(m.flops(), 5104)
        self.assertEqual(m.params(), 219)

    def test_backward_finite_differences(self):
        from evaluation.synthetic_dataset import make_sample
        from nx_attn_genuine import FixedGenuineAttentionModel
        s = make_sample("spatial", seed=7001)
        x, y = s["x"], 1
        m = FixedGenuineAttentionModel(seed=7)

        def loss_of(model):
            lg, _, _ = model.forward(x)
            mx = max(lg)
            ex = [math.exp(v - mx) for v in lg]
            pr = [v / sum(ex) for v in ex]
            return -math.log(max(pr[y], 1e-15))

        lg, _, cache = m.forward(x)
        mx = max(lg)
        ex = [math.exp(v - mx) for v in lg]
        pr = [v / sum(ex) for v in ex]
        grad = [pr[k] - (1.0 if k == y else 0.0) for k in range(3)]
        eps = 1e-6
        for name, i, j in (("Wq", 3, 5), ("Wk", 2, 6),
                           ("Wv", 4, 1), ("Wh", 5, 2)):
            m3 = copy.deepcopy(m)
            _, _, c3 = m3.forward(x)
            m3.backward(c3, list(grad), lr=1.0)
            m4 = copy.deepcopy(m)
            if name == "Wh":
                analytic = m.W_head[i][j] - m3.W_head[i][j]
                m4.W_head[i][j] += eps
            else:
                analytic = getattr(m.trunk, name)[i][j] - getattr(
                    m3.trunk, name)[i][j]
                getattr(m4.trunk, name)[i][j] += eps
            numeric = (loss_of(m4) - loss_of(m)) / eps
            self.assertLess(abs(analytic - numeric), 1e-6,
                            f"grad mismatch at {name}[{i}][{j}]")

    def test_nx_artifact(self):
        d = _load("experiments/results/nx_attn_genuine_vs_legacy.json")
        self.assertEqual(d["failures"], [])
        self.assertEqual(d["cost_model"], "v2-corrected")
        self.assertEqual(len(d["rows"]), 10)
        p = d["paired_genuine_minus_legacy"]
        # 10/10 legacy wins under the fixed protocol (MEASURED)
        self.assertEqual(p["n_legacy_wins"], 10)
        self.assertEqual(p["n_genuine_wins"], 0)
        self.assertLess(p["mean_diff"], 0.0)


if __name__ == "__main__":
    unittest.main()
