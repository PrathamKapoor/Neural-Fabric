# Reproducibility

- Python 3.11 venv (`.venv/`), stdlib + torch CPU + numpy only.
- Every experiment fixes seeds in-code and records them in artifacts.
- Full suite: `.venv\Scripts\python.exe -m unittest discover -s tests`.
- Reruns verified byte-identical excluding `timestamp`/`elapsed`/
  wall-clock leaves (see `docs/PROJECT_COMPLETE.md`).
- Provenance scripts: `experiments/provenance/`.
- Never commit `.env` (keys); `.env.example` documents every knob.
