"""Tests for the v2-corrected cost model and attention provenance.

Verifies (independently of flops() implementations):
- MLP/CNN/legacy-attention flops() match brute-force operation counts that
  replicate the forward loop bounds (boundary-aware for CNN).
- The legacy Attention primitive provably computes only a Wq projection:
  zeroing Wk/Wv leaves output bit-identical; output equals a hand-computed
  Wq matvec truncated to 8 entries.
- The NEW SoftmaxAttention primitive genuinely uses Q/K/V (zeroing Wk
  changes output), matches an independent manual computation, carries a
  distinct identity, and is NOT wired into historical experiments/engine.

Run: python -m pytest tests/test_cost_model.py -v
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fabric.primitives.mlp import MLP
from fabric.primitives.cnn import CNN
from fabric.primitives.attention import Attention
from fabric.primitives.genuine_attention import SoftmaxAttention
from fabric.primitives import cost_model


def count_mlp_ops(in_dim, hidden, out_dim):
    # forward: h[j] = max(0, sum_64 mult+add incl. bias); y likewise.
    # Count muls + adds only (ReLU comparisons excluded per convention).
    layer1 = hidden * (in_dim + in_dim)  # in_dim mults + in_dim adds (bias incl.)
    layer2 = out_dim * (hidden + hidden)
    return layer1 + layer2


def count_cnn_ops(in_ch, out_ch, k, h=8, w=8):
    # Replicates the forward loop bounds: only in-bounds taps execute.
    pad = k // 2
    mults = adds = 0
    for _oc in range(out_ch):
        for r in range(h):
            for c in range(w):
                for _ic in range(in_ch):
                    for kr in range(k):
                        ir = r + kr - pad
                        if 0 <= ir < h:
                            for kc in range(k):
                                ic_col = c + kc - pad
                                if 0 <= ic_col < w:
                                    mults += 1
                                    adds += 1  # val += ...
    # 2x2-quadrant average pooling: 15 adds + 1 divide per quadrant.
    pooling = out_ch * 4 * 16
    return mults + adds + pooling


def count_legacy_attention_ops(d):
    # Single Wq matvec: d mults + d adds per output row.
    return 2 * d * d


class TestCostModelValues(unittest.TestCase):
    def test_convention_ordering(self):
        self.assertLess(cost_model.PRIM_FLOPS["attention"],
                        cost_model.PRIM_FLOPS["cnn"])
        self.assertLess(cost_model.PRIM_FLOPS["cnn"],
                        cost_model.PRIM_FLOPS["mlp"])
        self.assertEqual(cost_model.COST_MODEL_VERSION, "v2-corrected")

    def test_mlp_matches_brute_force(self):
        for dims in [(64, 128, 3), (16, 32, 2), (8, 8, 3)]:
            m = MLP(in_dim=dims[0], hidden=dims[1], out_dim=dims[2])
            self.assertEqual(m.flops(), count_mlp_ops(*dims))
        self.assertEqual(MLP().flops(), 17152)

    def test_cnn_matches_brute_force(self):
        for dims in [(1, 8, 3), (2, 4, 5), (1, 2, 3)]:
            c = CNN(in_ch=dims[0], out_ch=dims[1], k=dims[2])
            self.assertEqual(c.flops(), count_cnn_ops(*dims),
                             f"mismatch for {dims}")
        self.assertEqual(CNN().flops(), 8256)
        # v1 undercount witness: old formula gives 1152, real count is 8256.
        self.assertEqual(1 * 8 * 3 * 3 * 16, 1152)
        self.assertGreater(CNN().flops(), 1152)

    def test_legacy_attention_matches_brute_force(self):
        for d in (32, 16, 8):
            a = Attention(d=d)
            self.assertEqual(a.flops(), count_legacy_attention_ops(d))
        self.assertEqual(Attention().flops(), 2048)
        # v1 overcount witness.
        self.assertEqual(3 * 32 * 32 * 4, 12288)


class TestLegacyAttentionProvenance(unittest.TestCase):
    def test_k_and_v_weights_are_dead(self):
        """Zeroing Wk/Wv leaves forward output bit-identical: they are dead."""
        a = Attention(d=32, seed=7)
        x = [0.05 * (i % 5) for i in range(64)]
        before = a.forward(x)
        a.Wk = [[0.0] * 32 for _ in range(32)]
        a.Wv = [[0.0] * 32 for _ in range(32)]
        self.assertEqual(a.forward(x), before)

    def test_output_is_truncated_wq_projection(self):
        """Output equals hand-computed Wq matvec truncated to 8 entries."""
        a = Attention(d=32, seed=11)
        x = [0.02 * (i % 9) for i in range(20)]
        xd = (x + [0.0] * (32 - len(x)))[:32]
        manual = [sum(xd[i] * a.Wq[i][j] for i in range(32))
                  for j in range(32)][:8]
        self.assertEqual(a.forward(x), manual)

    def test_legacy_identity_marker(self):
        a = Attention()
        self.assertTrue(a.LEGACY)
        self.assertEqual(a.name, "attention")


class TestGenuineAttention(unittest.TestCase):
    def test_distinct_identity_and_not_in_engine(self):
        g = SoftmaxAttention()
        self.assertEqual(g.name, "softmax_attention")
        from fabric.engine import FabricEngine
        eng = FabricEngine()
        self.assertNotIn("softmax_attention", eng.prims)
        self.assertIn("attention", eng.prims)  # history untouched

    def test_k_and_v_are_live(self):
        """Zeroing Wk changes output: K genuinely participates (unlike legacy)."""
        g = SoftmaxAttention(seed=7)
        x = [0.05 * (i % 5) for i in range(64)]
        before = g.forward(x)
        g.Wk = [[0.0] * 8 for _ in range(8)]
        self.assertNotEqual(g.forward(x), before)
        g2 = SoftmaxAttention(seed=7)
        g2.Wv = [[0.0] * 8 for _ in range(8)]
        self.assertNotEqual(g2.forward(x), before)

    def test_matches_independent_manual_computation(self):
        g = SoftmaxAttention(seed=11)
        x = [0.03 * (i % 7) for i in range(64)]
        n, d = g.n_tok, g.d_tok
        toks = [x[t * d:(t + 1) * d] for t in range(n)]
        Q = [[sum(t[i] * g.Wq[i][j] for i in range(d)) for j in range(d)]
             for t in toks]
        K = [[sum(t[i] * g.Wk[i][j] for i in range(d)) for j in range(d)]
             for t in toks]
        V = [[sum(t[i] * g.Wv[i][j] for i in range(d)) for j in range(d)]
             for t in toks]
        sc = 1.0 / math.sqrt(d)
        W = []
        for t in range(n):
            row = [sum(Q[t][i] * K[s][i] for i in range(d)) * sc
                   for s in range(n)]
            mx = max(row)
            ex = [math.exp(v - mx) for v in row]
            ssum = sum(ex)
            W.append([v / ssum for v in ex])
            # softmax rows normalize (genuine weighting, absent in legacy)
            self.assertAlmostEqual(sum(W[-1]), 1.0, places=9)
        O = [[sum(W[t][s] * V[s][j] for s in range(n)) for j in range(d)]
             for t in range(n)]
        manual = [sum(O[t][j] for t in range(n)) / n for j in range(d)]
        got = g.forward(x)
        self.assertEqual(len(got), d)
        for a_, b_ in zip(got, manual):
            self.assertAlmostEqual(a_, b_, places=12)

    def test_flops_formula(self):
        g = SoftmaxAttention()
        n, d = g.n_tok, g.d_tok
        expect = (3 * n * 2 * d * d) + (n * n * (2 * d - 1) + n * n) + \
                 (n * n * d + n * d * (n - 1))
        self.assertEqual(g.flops(), expect)
        self.assertEqual(g.flops(), 5056)

    def test_output_differs_from_legacy_projection(self):
        """Genuine attention is not a relabeled truncation of the legacy path."""
        leg = Attention(seed=5)
        gen = SoftmaxAttention(seed=5)
        x = [0.04 * (i % 6) for i in range(64)]
        self.assertNotEqual(leg.forward(x), gen.forward(x))


if __name__ == "__main__":
    unittest.main()
