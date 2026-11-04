import http.client
import json
import threading
import unittest
from http.server import HTTPServer
from neural_fabric.model import RoutedModel
from neural_fabric.server import handler_for, MAX_BODY


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), handler_for(RoutedModel().eval(), {
            "architecture": "test", "dataset": "test", "labels": ["a", "b", "c"], "metadata": {}}))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        conn.request(method, path, body=body, headers={"Content-Type": "application/json", "Origin": "https://example.com"})
        response = conn.getresponse()
        raw = response.read()
        status, headers = response.status, response.headers
        conn.close()
        return status, headers, raw

    def test_http_workflow_and_validation(self):
        status, headers, body = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["model_loaded"])
        self.assertIsNone(headers.get("Access-Control-Allow-Origin"))
        status, _, body = self.request("POST", "/predict", json.dumps({"input": [0.1]*64}))
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(body)["probabilities"]), 3)
        for payload in ("{", json.dumps({"input": [0]}),
                        json.dumps({"input": ["x"]*64}),
                        json.dumps({"input": [float("nan")]*64}),
                        json.dumps({"input": [True]*64}),
                        json.dumps({"input": [0]*64, "mode": "bad"})):
            self.assertEqual(self.request("POST", "/predict", payload)[0], 400)
        self.assertEqual(self.request("POST", "/predict", "x" * (MAX_BODY + 1))[0], 413)
        self.assertEqual(self.request("GET", "/missing")[0], 404)
        self.assertEqual(self.request("OPTIONS", "/predict")[0], 405)
        self.assertIn(b"Neural Fabric", self.request("GET", "/")[2])

    def test_missing_model(self):
        server = HTTPServer(("127.0.0.1", 0), handler_for(None, None))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
            conn.request("POST", "/predict", json.dumps({"input": [0]*64}))
            self.assertEqual(conn.getresponse().status, 503)
            conn.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
