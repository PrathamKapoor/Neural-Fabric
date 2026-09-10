"""Live serving-path tests (stdlib http only; no framework, no mocks).

Spins up backend/api/app.py's FabricHandler on an ephemeral port and checks:
- GET /health returns ok + engine mode
- POST /predict and /route return real engine output
- unknown POST path returns 404 (no stray 404 after successful writes)

Run: python -m pytest tests/test_serve.py -v
"""
import json
import os
import sys
import threading
import unittest
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import HTTPServer
from backend.api.app import FabricHandler


def _post(port, path, payload):
    req = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (port, path),
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {}


class TestServe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), FabricHandler)
        cls.port = cls.srv.server_address[1]
        cls.thread = threading.Thread(target=cls.srv.serve_forever,
                                      kwargs={"poll_interval": 0.05})
        cls.thread.daemon = True
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.thread.join(timeout=10)

    def test_health(self):
        with urllib.request.urlopen(
                "http://127.0.0.1:%d/health" % self.port,
                timeout=10) as r:
            body = json.loads(r.read().decode())
        self.assertEqual(r.status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["service"], "neural-fabric")
        self.assertEqual(body["engine_mode"], "topk")

    def test_predict_and_route(self):
        status, body = _post(self.port, "/predict",
                             {"input": [0.1] * 64, "graph_mode": True})
        self.assertEqual(status, 200)
        self.assertGreater(body["compute"], 0)
        status, body = _post(self.port, "/route", {"input": [0.1] * 64})
        self.assertEqual(status, 200)
        self.assertIn("selected", body)

    def test_unknown_path_is_404(self):
        status, _ = _post(self.port, "/nope", {})
        self.assertEqual(status, 404)

    def test_env_port_config(self):
        import importlib
        import backend.api.app as app
        importlib.reload(app)
        self.assertTrue(hasattr(app, "main"))
        self.assertIn("NF_PORT", open(
            os.path.join(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))),
                "backend", "api", "app.py")).read())


if __name__ == "__main__":
    unittest.main()
