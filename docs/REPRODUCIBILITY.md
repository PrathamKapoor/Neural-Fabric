# Reproducibility

- Python 3.11 venv (`.venv/`), stdlib + torch CPU + numpy only.
- Measured environment (2026-09-09): Python 3.11.16, numpy 2.4.6,
  torch 2.14.0+cpu. No lockfile is committed; these versions are recorded
  here so a third party can reconstruct the environment exactly.
- Every experiment fixes seeds in-code and records them in artifacts.
- Full suite: `.venv\Scripts\python.exe -m unittest discover -s tests`.
- Reruns verified byte-identical excluding `timestamp`/`elapsed`/
  wall-clock leaves (see `docs/PROJECT_COMPLETE.md`).
- Provenance scripts: `experiments/provenance/`.
- Never commit `.env` (keys); `.env.example` documents every knob.
