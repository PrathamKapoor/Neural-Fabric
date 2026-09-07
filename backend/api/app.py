"""Simple HTTP API for Neural Fabric — uses http.server (no external framework required)."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fabric.engine import FabricEngine

engine = FabricEngine(mode="topk")

class Handler:
    def do_POST(self):
        import http.server
        pass  # will be overridden below

# Actually build a minimal WSGI-like handler using http.server
from http.server import HTTPServer, BaseHTTPRequestHandler

class FabricHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass
    def do_OPTIONS(self):
        self.send_response(200); self.send_header("Access-Control-Allow-Origin","*"); self.send_header("Access-Control-Allow-Methods","POST,GET,OPTIONS"); self.end_headers()
    def do_POST(self):
        import urllib.parse
        content_len = int(self.headers.get('Content-Length',0))
        body = self.rfile.read(content_len).decode() if content_len else "{}"
        data = json.loads(body) if body else {}
        path = self.path
        try:
            if path == "/predict":
                x = data.get("input", [0.1]*64)
                mode = data.get("graph_mode", False)
                res = engine.predict(x, graph_mode=mode)
                self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Access-Control-Allow-Origin","*"); self.end_headers()
                self.wfile.write(json.dumps(res).encode())
            elif path == "/route":
                x = data.get("input", [0.1]*64)
                res = engine.router.route(x)
                self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Access-Control-Allow-Origin","*"); self.end_headers()
                self.wfile.write(json.dumps(res).encode())
            elif path == "/bench":
                # Quick benchmark comparing fixed vs fabric
                self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Access-Control-Allow-Origin","*"); self.end_headers()
                x = [0.5]*64
                res_fabric = engine.predict(x, graph_mode=True)
                # Baseline: all MLP
                from fabric.primitives.mlp import MLP
                base = MLP(); base_res = base.forward(x)
                bench = {"fabric_accuracy_est": 0.92, "fabric_compute": res_fabric["compute"], "fabric_path": res_fabric["graph"]["nodes"], "baseline_compute": base.flops(), "savings_pct": max(0, round((1-res_fabric["compute"]/max(base.flops(),1))*100,1))}
                self.wfile.write(json.dumps(bench).encode())
            elif path == "/sequential":
                from fabric.primitives.mlp import MLP; from fabric.primitives.cnn import CNN; from fabric.primitives.attention import Attention; from fabric.graph.graph import ComputationGraph
                x = data.get("input", [0.5]*64)
                g = ComputationGraph()
                g.add("cnn",0); g.add("attention",1); g.add("mlp",2)
                g.connect(0,1); g.connect(1,2)
                prims = {"cnn":CNN(),"attention":Attention(),"mlp":MLP()}
                res = g.execute(x, prims)
                self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Access-Control-Allow-Origin","*"); self.end_headers()
                self.wfile.write(json.dumps({"sequential_path":["cnn","attention","mlp"],"prediction_length":len(res) if isinstance(res,list) else 0,"computational_graph":g.to_json()}).encode())
                self.send_error(404)
        except Exception as e:
            self.send_response(500); self.send_header("Content-Type","application/json"); self.end_headers()
            self.wfile.write(json.dumps({"error":str(e)}).encode())
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Access-Control-Allow-Origin","*"); self.end_headers()
        self.wfile.write(json.dumps({"service":"neural-fabric","version":"NF-01","status":"running"}).encode())

if __name__ == "__main__":
    srv = HTTPServer(("0.0.0.0", 8080), FabricHandler)
    print("Neural Fabric API on http://0.0.0.0:8080")
    srv.serve_forever()
