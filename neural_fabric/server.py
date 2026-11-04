"""Local development HTTP server for the trained classifier."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import torch
from .checkpoint import load
from .workflow import predict

MAX_BODY = 16384
PAGE = Path(__file__).with_name("index.html")


def handler_for(model, metadata):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, data):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/health":
                self.respond(200, {"status": "ok", "model_loaded": model is not None})
            elif self.path == "/model":
                if model is None:
                    self.respond(503, {"error": "model unavailable"})
                else:
                    self.respond(200, {"architecture": metadata["architecture"],
                                       "dataset": metadata["dataset"], "input_dim": 64,
                                       "labels": metadata["labels"], "training": metadata["metadata"]})
            else:
                self.respond(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/predict":
                self.respond(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if length < 0:
                    raise ValueError("valid Content-Length required")
                if length > MAX_BODY:
                    self.respond(413, {"error": "request too large"})
                    return
                payload = json.loads(self.rfile.read(length).decode("utf-8"),
                                     parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON number")))
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object")
                mode, k = payload.get("mode", "top1"), payload.get("k", 1)
                if mode not in ("soft", "top1", "topk"):
                    raise ValueError("invalid mode")
                if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 3:
                    raise ValueError("k must be an integer from 1 to 3")
                from .workflow import validate_input
                validate_input(payload.get("input"))
                if model is None:
                    self.respond(503, {"error": "model unavailable"})
                    return
                self.respond(200, predict(model, payload["input"], mode, k))
            except (ValueError, UnicodeDecodeError, TypeError, OverflowError):
                self.respond(400, {"error": "invalid request: expected JSON with 64 finite numeric inputs and valid mode/k"})

        def do_OPTIONS(self):
            self.respond(405, {"error": "method not allowed"})

        do_PUT = do_DELETE = do_PATCH = do_OPTIONS

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        model, data = load(args.model)
    except ValueError as exc:
        print(str(exc))
        return 1
    server = HTTPServer((args.host, args.port), handler_for(model, data))
    print("Listening on http://%s:%s" % server.server_address, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
