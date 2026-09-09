# Laboratory Architecture (v2 + lab extensions)

```
IDEA → RESEARCHER → HYPOTHESIS → TASK CONTRACT → MANAGER → BUILDER
→ DETERMINISTIC VERIFIER → ADVERSARIAL TESTER → EXPERIMENTER
→ ANALYST → CRITIC → REPAIR/REPLAN → DOCUMENT → COMMIT → NEXT
```

## What existed (reused, not replaced)

- `agent_loop/` — manager/worker RoleChains, fallback chains, invocation
  records, budgets, 24h deadline, resume, evidence collection, TerminalUI.
- `agent_loop/verify.py` — claim/artifact, scope, overclaim gates.
- `agent_loop/tester.py` — PASS/FAIL/INCONCLUSIVE verdicts, opencode + api
  backends, combination policy.
- `agent_loop/state.py` — counters, research state, history, progress.

## What was added (`lab/`)

| Module | Responsibility |
|---|---|
| `registry.py` | Questions, hypotheses (falsification enforced), failure lessons, ideas. Persisted JSON. |
| `experiment.py` | Standard artifact envelope, multi-seed runner (instability preserved), budget parity checker. |
| `analyze.py` | Claim classification MEASURED/DERIVED/INFERRED/SPECULATIVE; doc↔artifact number reconciliation. |
| `critic.py` | Alternative explanations + discriminating experiments; conclusion-strength review. |
| `researcher.py` | Novelty attack over local code/docs (+ optional web fetcher; never invents sources). |
| `idea.py` | Check-gated follow-up generation (novelty/testability/dedup decided; feasibility/baseline/falsification honestly deferred to design). |

## Loop integration

Each iteration now runs: manager decision (PLANNED) → builder (BUILDING) →
tester + gates (VERIFYING → ACCEPTED/REPAIR, stagnation → REPLAN) →
analyst/critic (claim kinds, alternatives, doc self-audit on changed `.md`).
Reports carry `verification` + `analysis` blocks into manager review;
FAIL verdicts count toward failure budgets; REPLAN signals force replans.

Role separation is structural: manager never edits, worker never decides,
tester never modifies, analyst/critic are deterministic (LLM critic optional
via `TESTER_CRITIC`, default off).
