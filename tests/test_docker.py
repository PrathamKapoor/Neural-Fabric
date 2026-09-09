"""Static verification for the Docker security correction.

Docker itself is unavailable in this environment, so these tests validate
statically (no dynamic build claimed):
- .dockerignore exists and excludes .env variants, key material, .git,
  virtualenvs, caches, and experiment scratch — while keeping serving files.
- Dockerfile fails fast (no `|| true`, no silenced pip stderr), installs no
  torch, and smoke-tests the serving closure at build time.

Run: python -m pytest tests/test_docker.py -v
"""
import fnmatch
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths that MUST be excluded from the image build context.
MUST_IGNORE = [
    ".env",
    ".env.production",
    ".env.local",
    "service.key",
    "cert.pem",
    ".git/config",
    ".git/HEAD",
    ".venv/lib/python3.11/site.py",
    "venv/Scripts/activate",
    "fabric/__pycache__/engine.pyc",
    "experiments/runs/nf02_soft_seed1.json",
    "experiments/checkpoints/ckpt.pt",
    "frontend/node_modules/pkg/index.js",
    "model.safetensors",
]

# Serving-closure paths that MUST remain in the build context.
MUST_KEEP = [
    "Dockerfile",
    "backend/api/app.py",
    "fabric/engine.py",
    "fabric/primitives/mlp.py",
    "fabric/router/router.py",
    "evaluation/synthetic_dataset.py",
]


def load_patterns():
    path = os.path.join(REPO, ".dockerignore")
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f
                if ln.strip() and not ln.strip().startswith("#")]


def ignored(rel, patterns):
    """Approximate Docker's ignore matching for the patterns we use."""
    rel = rel.replace(os.sep, "/")
    base = rel.rsplit("/", 1)[-1]
    parts = rel.split("/")
    for pat in patterns:
        p = pat.rstrip("/")
        is_dir = pat.endswith("/")
        if "/" not in p:
            # Docker semantics: a pattern without a slash matches the
            # basename at any depth AND excludes whole subtrees when it
            # names a directory (.git ignores .git/config too).
            if fnmatch.fnmatch(base, p) or fnmatch.fnmatch(rel, p):
                return True
            if p in parts:
                return True
        else:
            if is_dir:
                if rel == p or rel.startswith(p):
                    return True
            else:
                if fnmatch.fnmatch(rel, p) or rel.startswith(p + "/"):
                    return True
    return False


class TestDockerignore(unittest.TestCase):
    def test_dockerignore_exists(self):
        self.assertTrue(os.path.exists(os.path.join(REPO, ".dockerignore")))

    def test_secrets_and_scratch_excluded(self):
        pats = load_patterns()
        for rel in MUST_IGNORE:
            self.assertTrue(ignored(rel, pats), f"NOT ignored: {rel}")

    def test_serving_closure_kept(self):
        pats = load_patterns()
        for rel in MUST_KEEP:
            self.assertFalse(ignored(rel, pats), f"wrongly ignored: {rel}")


class TestDockerfile(unittest.TestCase):
    def _body(self):
        with open(os.path.join(REPO, "Dockerfile"), encoding="utf-8") as f:
            return f.read()

    def _code(self):
        # Non-comment Dockerfile instructions only (comments may legitimately
        # name the old anti-patterns they prohibit).
        return "\n".join(ln for ln in self._body().splitlines()
                         if not ln.strip().startswith("#"))

    def test_no_failure_swallowing(self):
        code = self._code()
        self.assertNotIn("|| true", code)
        self.assertNotIn("2>/dev/null", code)
        self.assertNotIn("2>nul", code)

    def test_no_torch_or_unused_deps(self):
        low = self._code().lower()
        self.assertNotIn("torch", low)
        # flask/numpy were installed-but-unused by the server; they must not
        # be reintroduced as image dependencies.
        for line in self._code().splitlines():
            s = line.strip()
            if s.startswith("RUN python -c"):
                continue
            self.assertNotIn("pip install", s)

    def test_build_time_smoke_of_serving_closure(self):
        body = self._body()
        self.assertIn("FabricEngine", body)
        self.assertIn("assert", body)

    def test_serving_closure_is_stdlib_only(self):
        # The production server must not import third-party packages:
        # walk its transitive local imports and assert none reference
        # torch/numpy/flask/etc.
        import ast
        third_party = {"torch", "numpy", "flask", "django", "fastapi",
                       "uvicorn", "pandas", "sklearn", "scipy"}
        seen, stack = set(), [
            os.path.join(REPO, "backend", "api", "app.py"),
            os.path.join(REPO, "fabric", "engine.py"),
        ]
        imported_local = set()
        while stack:
            path = stack.pop()
            if path in seen or not os.path.exists(path):
                continue
            seen.add(path)
            tree = ast.parse(open(path, encoding="utf-8").read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        top = a.name.split(".")[0]
                        self.assertNotIn(top, third_party,
                                         f"{path} imports {top}")
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:
                        top = node.module.split(".")[0]
                        self.assertNotIn(top, third_party,
                                         f"{path} imports {top}")
                    elif node.level and node.module:
                        base = os.path.dirname(path)
                        cand = os.path.normpath(
                            os.path.join(base, *([os.pardir] * (node.level - 1)),
                                         node.module.replace(".", os.sep) + ".py"))
                        imported_local.add(cand)
                        pkg = os.path.normpath(
                            os.path.join(base, *([os.pardir] * (node.level - 1)),
                                         node.module.replace(".", os.sep),
                                         "__init__.py"))
                        imported_local.add(pkg)
        # resolve relative fabric imports of engine.py
        for extra in ["fabric/primitives/mlp.py", "fabric/primitives/cnn.py",
                      "fabric/primitives/attention.py",
                      "fabric/router/router.py", "fabric/graph/graph.py"]:
            imported_local.add(os.path.join(REPO, *extra.split("/")))
        for path in imported_local:
            if os.path.exists(path) and path.endswith(".py"):
                stack.append(path)
        # second pass over discovered locals
        while stack:
            path = stack.pop()
            if path in seen or not os.path.exists(path):
                continue
            seen.add(path)
            tree = ast.parse(open(path, encoding="utf-8").read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        self.assertNotIn(a.name.split(".")[0], third_party,
                                         f"{path} imports {a.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:
                        self.assertNotIn(node.module.split(".")[0],
                                         third_party,
                                         f"{path} imports {node.module}")


if __name__ == "__main__":
    unittest.main()
