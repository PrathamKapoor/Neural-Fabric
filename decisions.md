# Neural Fabric — Decisions

## Decision: Platform-specific process supervision with shared coverage

**Status:** Accepted

**Context:** Detached jobs used Windows process APIs even on Linux and returned UNKNOWN.

**Decision:** Keep Windows supervision and use Linux memory probing, POSIX
session-based process groups, liveness checks, and tracked child exit codes.
Run the full suite on both hosted operating systems. Windows certificate-store
tests remain Windows-only, while portable TLS behavior is tested on both.

**Alternatives considered:** Skipping the entire detached-job suite on Linux.

**Reason:** Linux job operation is useful and can be exercised with real child processes.

**Consequences:** POSIX recovery cannot recover the exit code of an untracked
child; OS process permissions can still make liveness unknown. Process-group
termination depends on jobs starting as session leaders.

**Verification:** Linux WSL targeted job, TLS and AGY suites; see flow.md.

## Decision: Numeric CSV with training-only preprocessing

**Status:** Accepted

**Context:** Users need predictions on their own tabular data without leaking held-out statistics.

**Decision:** Split rows by class first, then fit per-column means and scales on training rows. Save feature names, labels, split indices, preprocessing and source SHA-256 in a versioned checkpoint.

**Alternatives considered:** Fitting preprocessing on the full file; automatically encoding categorical input features.

**Reason:** The former leaks test information; the latter adds ambiguity and complexity.

**Consequences:** CSV features must be finite numeric values and each class must have at least five rows. Evaluation requires the exact original file.

**Verification:** `tests/test_csv_workflow.py` and the UCI Iris example in `docs/CSV_EXPERIMENT.md`.

## Decision: Preserve historical experiments

**Status:** Accepted

**Context:** Historical router-selection accuracy does not measure the accuracy of outputs from executed experts.

**Decision:** Preserve existing artifacts and distinguish historical results from any new end-to-end evaluation.

**Alternatives considered:** Rewriting old artifacts or changing their interpretation silently.

**Reason:** Preserving provenance allows prior results to be audited.

**Consequences:** New results require separate outputs and explicit comparison limitations.

**Verification:** Inspection of `experiments/nf12_routing_modes.py` and its committed results.

## Decision: Offline synthetic classification

**Status:** Accepted

**Context:** A usable task must run without network access or credentials.

**Decision:** Classify generated 64-value vectors using three block-sum labels and separate seeded splits.

**Alternatives considered:** An external real dataset.

**Reason:** The task is fast, inspectable and reproducible without downloads.

**Consequences:** It does not establish real-world generalization.

**Verification:** `neural_fabric/data.py`; training and evaluation commands in the README.

## Decision: PyTorch experts and sparse dispatch

**Status:** Accepted

**Context:** Experts must learn the same prediction task and sparse execution must bypass unused experts.

**Decision:** Use CPU PyTorch, a linear router and three differently sized linear/MLP experts. Soft combines logits using all router weights; top-k renormalizes selected weights and dispatches selected rows only.

**Alternatives considered:** Extending the legacy pure-Python demo.

**Reason:** Autograd reduces training complexity while explicit dispatch makes sparse execution testable.

**Consequences:** PyTorch is required; routed and fixed models have different parameter budgets.

**Verification:** `python -m pytest tests/test_end_to_end.py -q`.

## Decision: Versioned checkpoint and modeled FLOPs

**Status:** Accepted

**Context:** Users need repeatable inference and an honest compute comparison.

**Decision:** Save tensor weights and metadata together with PyTorch weights-only loading; count router and expert multiply/add operations and preserve signed savings.

**Alternatives considered:** Unsafe unrestricted pickle loading; presenting latency as FLOPs.

**Reason:** Compatibility is checked and modeled operations remain distinct from timed latency.

**Consequences:** Checkpoints are PyTorch-specific; FLOPs exclude activations and dispatch.

**Verification:** Save/load and cost tests in `tests/test_end_to_end.py`.

## Decision: Same-origin local API and UI

**Status:** Accepted

**Context:** A browser must access the trained model without permissive CORS defaults.

**Decision:** Use Python's HTTP server bound to loopback by default and serve the new frontend from the same origin.

**Alternatives considered:** Enabling CORS for arbitrary local pages.

**Reason:** No additional web framework or cross-origin permission is required.

**Consequences:** Development server has no authentication or TLS; production exposure needs independent protection.

**Verification:** `python -m pytest tests/test_trained_api.py -q` and real HTTP smoke.

## Decision: Preserve legacy graph and agent loop

**Status:** Accepted

**Context:** Historical experiment code and autonomous development tooling have different execution and security semantics.

**Decision:** Leave legacy behavior intact and document that graph edges are not executed as a DAG and command filters are not an OS sandbox.

**Alternatives considered:** A general DAG engine or sandbox platform.

**Reason:** Neither is needed for the new classifier, and changing old behavior risks research provenance.

**Consequences:** Legacy code remains a demonstration and agent isolation is deferred.

**Verification:** Source inspection and `docs/TRAINED_MODEL.md`.
