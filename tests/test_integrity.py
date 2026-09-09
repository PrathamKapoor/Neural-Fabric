"""Integrity regression tests: fabricated metrics must never (re)appear.

Covers PART 1 (hardcoded 0.92 removal) and PART 7 (hallucinated metrics):
bench() returns None/unmeasured, and source scans prove no invented
accuracy constants exist in the serving path.

Run:  python -m unittest discover -s tests -v   (from repo root)
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# e.g. "accuracy_est": 0.92 / accuracy_est=0.92 / accuracy 0.92 (invented const)
_INVENTED = re.compile(
    r"""accuracy\w*['"]?\s*[:=]\s*0\.\d+""", re.IGNORECASE)


def _read(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return f.read()


class TestNoFabricatedAccuracy(unittest.TestCase):
    def test_bench_returns_unmeasured_not_invented(self):
        from fabric.engine import FabricEngine
        res = FabricEngine().bench([0.1] * 64)
        self.assertIsNone(res["fabric_accuracy_est"])
        self.assertIn("unmeasured", res["accuracy_status"])
        # compute figures remain real measurements
        self.assertGreater(res["fabric_compute"], 0)
        self.assertGreater(res["baseline_compute"], 0)

    def test_engine_source_has_no_invented_accuracy(self):
        src = _read(os.path.join("fabric", "engine.py"))
        self.assertIsNone(_INVENTED.search(src),
                          "invented accuracy constant in fabric/engine.py")

    def test_api_source_has_no_invented_accuracy(self):
        src = _read(os.path.join("backend", "api", "app.py"))
        self.assertIsNone(_INVENTED.search(src),
                          "invented accuracy constant in backend/api/app.py")

    def test_serving_path_sources_scan_clean(self):
        bad = []
        for root, _, files in os.walk(os.path.join(REPO, "fabric")):
            for fn in files:
                if fn.endswith(".py"):
                    rel = os.path.relpath(os.path.join(root, fn), REPO)
                    if _INVENTED.search(_read(rel)):
                        bad.append(rel)
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
