"""Tests for the canonical demo (scripts/demo_routing.py).

All assertions are structural (keys, types, real engine values) — the demo
itself performs no mocking, so there is nothing to mock here either.

Run: python -m pytest tests/test_demo.py -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.demo_routing import demo_case, MODES, EVIDENCE


class TestDemo(unittest.TestCase):
    def test_all_modes_produce_records(self):
        for i, mode in enumerate(MODES):
            c = demo_case(mode, seed=i)
            self.assertEqual(c["mode"], mode)
            self.assertEqual(len(c["route_probs"]), 3)
            self.assertAlmostEqual(sum(c["route_probs"]), 1.0, places=3)
            self.assertEqual(len(c["selected"]), 1)
            self.assertIn(c["selected"][0], ("mlp", "cnn", "attention"))
            self.assertGreater(c["executed_flops"], 0)
            self.assertLessEqual(c["executed_flops"],
                                 c["baseline_mlp_flops"])
            self.assertIn("accuracy_note", c)

    def test_predicted_class_honesty(self):
        # 3-dim head output -> class index; feature maps -> None, never fake.
        c = demo_case("dense", seed=2)
        if len(c["prediction"]) == 3:
            self.assertIn(c["predicted_class"], (0, 1, 2))
        else:
            self.assertIsNone(c["predicted_class"])

    def test_evidence_pointers_exist(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for _, rel in EVIDENCE.items():
            self.assertTrue(os.path.isfile(os.path.join(repo, rel)), rel)


if __name__ == "__main__":
    unittest.main()
