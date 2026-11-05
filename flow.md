# Neural Fabric — Work Flow

## Goal
Provide reproducible CPU training and inference with actual routed expert predictions.

## Current State
Baseline from clean main at c431584: 343 tests passed. A separate trainable synthetic classifier, checkpoint, CLI, local API and UI are implemented. Legacy artifacts and serving are preserved.

2026-11-05: merged latest main into the feature branch after resolving the README
conflict; historical routing interpretation from both sides is preserved.
The numeric CSV CLI now saves train-only standardization and split indices,
with separate fixed baselines and an exploratory UCI Iris evaluation.

## Work Plan
- [x] Inspect repository and establish baseline.
- [x] Implement real end-to-end routing.
- [x] Implement model training and checkpoint lifecycle.
- [x] Implement inference CLI and API.
- [x] Fix graph and compute-accounting issues for the new model; legacy limitations documented.
- [x] Update frontend where applicable.
- [x] Improve evaluation and research documentation.
- [x] Add or improve CI.
- [x] Run end-to-end verification.
- [ ] Commit and push.

## Verification
- GitHub Actions run 42032860004 on c06d3c2 reported 15 failed, 333 passed
  and 2 skipped on Linux. Windows-only process APIs caused detached-job
  failures; TLS tests assumed Windows certificate APIs and an empty trust
  store; the AGY test depended on a local executable.
- Linux WSL targeted checks: `python3 -m unittest discover -s tests -p
  test_lab_jobs.py`: 18 passed; `test_tls.py`: 10 passed, 2 Windows-store
  skips; `test_agy_worker.py`: 32 passed.
- Windows local `python -m pytest tests/ -q`: 350 passed in 128.56s.
- GitHub Actions `Trained model` on 24dabcd: success, https://github.com/PrathamKapoor/Neural-Fabric/actions/runs/41884306514.
- `python -m pytest tests/test_csv_workflow.py -q`: 2 passed in 32.82s (before baseline reporting was added).
- `python -m pytest tests/ -q`: 350 passed in 81.79s on Python 3.13 after CSV changes and main integration.
- `python -m neural_fabric predict-csv --csv artifacts/iris-new.csv --model artifacts/iris-1.pt`: returned three predictions from a CSV without the target column.
- A fresh isolated dependency installation was not performed. The existing trained-demo browser was exercised for screenshots in the earlier update; the CSV workflow is CLI-only.
- `python -m neural_fabric train-csv --csv artifacts/iris.csv --target species --output artifacts/iris-1.pt --seed 1 --epochs 40` and corresponding `evaluate-csv` and `predict-csv`: completed; 150 predictions. Seeds 2 and 3 were run through the Python API. See docs/CSV_EXPERIMENT.md.
- `docker info --format '{{.ServerVersion}}'`: Docker daemon unavailable, so trained image build and execution remain unverified.
- `python -m pytest tests/ -q`: baseline 343 passed in 73.70s, Python 3.13.14.
- `python -c "import torch; print(torch.__version__)"`: 2.13.0+cpu.
- `python -m pytest tests/test_end_to_end.py tests/test_trained_api.py -q`: 5 passed in 7.41s.
- `python -m neural_fabric train-demo --output artifacts/demo-model.pt`: completed, seed 42, 20 epochs.
- `python -m neural_fabric evaluate --model artifacts/demo-model.pt --mode top1` and `--mode soft`: completed, 150 held-out synthetic examples.
- `python -m neural_fabric predict --model artifacts/demo-model.pt --input examples/sample.json`: passed.
- Real subprocess HTTP smoke: health, prediction and frontend returned 200.
- `python -m pytest tests/ -q`: 348 passed in 81.33s after implementation.
- `python -m pytest tests/test_end_to_end.py tests/test_trained_api.py -q`: 5 passed in 8.98s after final evaluation changes.
- `python -m neural_fabric evaluate --model artifacts/demo-model.pt --mode top1 --output artifacts/evaluation.json`: passed; output is excluded from Git.
- Additional exploratory runs used `python -m neural_fabric train-demo --output artifacts/seed-N.pt --seed N`, followed by `python -m neural_fabric evaluate --model artifacts/seed-N.pt --mode MODE`, for seeds 1, 2, 3 and modes soft/top1 (equivalent workflow executed through the Python API, 20 epochs, 150 test examples each). Top-1 accuracies: 0.847, 0.787, 0.800; fixed baseline: 0.833, 0.887, 0.900. Top-1 modeled compute exceeded the fixed baseline on all three.
- A virtual environment using system site packages loaded the saved model and ran CLI inference; a fresh dependency download was not verified.
- `docker info --format '{{.ServerVersion}}'`: failed because the Docker daemon was unavailable. Container build/runtime were not verified.

## Blockers
Docker daemon unavailable; Docker build and runtime checks skipped.

## Next Actions
Review staged changes and push the feature branch.

## Final State
Working branch: feature/usable-neural-fabric. Commit and push status will be reported separately. Real-data generalization, a fresh dependency installation, hosted CI, browser rendering and Docker runtime remain unverified.
