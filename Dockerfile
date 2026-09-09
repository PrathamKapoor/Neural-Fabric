# Neural Fabric serving image.
#
# Runtime dependency audit (v2-corrected pass):
# - Serving entrypoint backend/api/app.py uses stdlib (http.server, json)
#   plus fabric/ (pure-Python engine/router/primitives). Verified: no
#   `import torch/numpy/flask` anywhere on the serving import closure
#   (torch appears only in fabric/primitives/torch_primitives.py,
#   fabric/router/torch_router.py and experiments/nf02_torch_train.py,
#   none of which the server imports).
# - Therefore: NO torch/flask/numpy in the production image. Do NOT add
#   torch "because research code uses it".
# - Secrets: never COPY credentials; .dockerignore excludes .env/.env.*,
#   key material, .git, venvs, caches, and experiments scratch. Inject
#   secrets at runtime (compose env_file / orchestrator secrets), never bake.
# - Fail-fast: dependency/smoke steps MUST fail the build on error (no
#   `|| true`, no silenced stderr).
FROM python:3.11-slim
WORKDIR /app
COPY . /app/
# Build-time smoke test of the exact serving closure; fails the build if
# the server cannot import or execute (replaces the old venv/pip line that
# swallowed failures with `|| true` and installed unused packages).
RUN python -c "from fabric.engine import FabricEngine; e = FabricEngine(); r = e.predict([0.1]*64, graph_mode=True); assert r['compute'] > 0, 'engine smoke failed'; print('engine smoke OK, compute =', r['compute'])"
EXPOSE 8080
CMD ["python", "backend/api/app.py"]
