# agent_loop - autonomous manager/worker development loop for Neural Fabric

Local orchestration for the Neural Fabric research lifecycle
(Research -> Design -> Implement -> Test -> Benchmark -> Analyze -> Critique ->
Improve -> Re-test -> Re-benchmark -> Document -> Commit -> Continue).

```
USER OBJECTIVE -> MANAGER -> task -> WORKER (tools) -> worker report
  -> automatic evidence (git status/diff, artifacts) -> MANAGER review
  -> EXPECTED vs ACTUAL -> complete | continue | retry | blocked
```

Standalone: direct provider HTTPS APIs (stdlib only). No IDE, harness, or
external process needed.

## Topology (chains never cross)

- **Manager chain**: `nvidia/nemotron-3-ultra-550b-a55b:free` (OpenRouter,
  `https://openrouter.ai/api/v1`, constrained free endpoint — one coherent
  work unit per call, compact capped context, gentle retries honoring
  `Retry-After`) — on infrastructure failure only, falls back to
  `moonshotai/kimi-k3` (NVIDIA, `https://integrate.api.nvidia.com/v1`,
  explicit `MANAGER_FALLBACK_API_KEY`). Both are MANAGER ONLY
  and never receive tools.
- **Worker chain**: AGY headless CLI (**Gemini 3.8 Flash High**, local Antigravity
  session, no credentials configured) — on infrastructure failure only, falls
  back to `nvidia/nemotron-3-super-120b-a12b:free` (OpenRouter,
  `https://openrouter.ai/api/v1`), then to `deepseek-ai/deepseek-v4-flash-0731`
  (NVIDIA). AGY and both API fallbacks are WORKER ONLY and can never become
  manager (separate chain objects + constructor cross-contamination guard
  over all fallback levels).
- **Tester layer** (`tester.py`, `verify.py`): every completed builder task
  passes deterministic gates (claim-vs-artifact, scope vs `allowed_dirs`,
  overclaim language) then an adversarial LLM tester, before the manager
  reviews it. Backends: `opencode` (headless `opencode run`, read-only, no
  `--auto`), `api` (manager-fallback provider, review only), `off`
  (gates only). Verdicts PASS / FAIL / INCONCLUSIVE with evidence; any FAIL
  wins; FAIL verdicts count toward the worker failure budget. Task states
  (`PLANNED → BUILDING → VERIFYING → EXPERIMENTING → EVIDENCE_CHECK →
  ACCEPTED`, with `REPAIR` / `REPLAN` failure paths) and stagnation signals
  persist in `.agent_loop` and feed the manager's next decision.
- **Analyst / critic** (deterministic, zero quota): every iteration's claims
  are classified (MEASURED/DERIVED/INFERRED/SPECULATIVE), critic alternatives
  are generated for accepted work, and changed docs are reconciled against
  iteration evidence. Optional LLM critic via `TESTER_CRITIC=true`.
- **Research lab** (`lab/`): persistent question/hypothesis registry with
  enforced falsification conditions, standardized experiment envelopes,
  multi-seed runner, budget-parity checker, novelty attack, check-gated
  idea generation, and a failure-pattern log. See `docs/ARCHITECTURE.md`.

Topology note (2026-09-08, user-authorized): the previous DeepSeek V4 Pro /
Inkling-Small topology proved unreachable (measured hangs + HTTP 403) and was
replaced first by Nemotron models, then by the current Kimi-manager /
AGY-worker topology after live probes confirmed reachability.

## AGY worker backend (`agent_loop/agy.py`)

`AgyWorkerClient` implements the same `chat()` contract as the API clients,
so the worker chain is AGY -> Nemotron Super -> DeepSeek Flash with the
existing infrastructure-only fallback semantics. One `chat()` call runs one
headless `agy --print` task (`--output-format json --add-dir <repo> --mode
accept-edits`); AGY's own tool loop does the file/command work, and the
result text is parsed as the normal worker report. Failure kinds
(missing/auth/permission/model/quota/timeout/exit-error) are classified from
exit codes + stderr and take the fallback path; only exit-0 content flows to
manager review. Loop-level evidence (git status/diff, artifacts) is collected
exactly as for API workers, and the zero-tool fabrication guard applies to
API workers only (AGY reports carry `_executor: agy` provenance).

Fallback is for infrastructure failures (connection/DNS/TLS, timeout, 5xx,
provider/model unavailable, rate limits, malformed responses) — never for bad
task results. A bad result flows to evidence -> manager review. Every attempt
is recorded (`role/provider/model/attempt/status/reason/fallback_from`) in
`.agent_loop/history/` and a capped tail in `state.json`, and shown live.

## Roles

- **Manager** (`prompts.MANAGER_SYSTEM`, `loop.AgentLoop._manager_step`):
  supervises only. Reads objective + loop state + previous decision + worker
  report + objective evidence, judges EXPECTED vs ACTUAL, emits exactly ONE
  structured decision
  (`schemas`: `continue | complete | retry | blocked` + `next_task` +
  `expected_result` + `validation`). Never edits files (it has no tools).
- **Worker** (`prompts.WORKER_SYSTEM`, `loop.AgentLoop._worker_step`):
  executes the single assigned task with real function tools
  (`tools.WORKER_TOOLS` + `ToolExecutor`): list/read/write/edit files, run
  commands, git status/diff, list artifacts. Ends with exactly ONE structured
  report (`done | partial | blocked | failed`). Never fabricates results.

## Architecture

```
agent_loop/
  __main__.py   CLI: config check, connectivity probe, repo inspection,
                state init, objective intake, loop start, Ctrl+C/STOP handling
  loop.py       AgentLoop orchestration + termination + TerminalUI log
  clients.py    Isolated OpenAI-compatible adapter (stdlib urllib only) +
                MockClient for tests; manager_client / worker_client built
                together in build_clients(), provider details live ONLY here
  prompts.py    Neural Fabric system prompts + message builders
                (objective + constraints + state + report + evidence)
  schemas.py    Lenient JSON extraction, strict schema validation, safe
                fallbacks for malformed model output
  tools.py      Worker function-tool schemas + ToolExecutor with git safety
                (push always blocked, destructive git blocked, commit only
                when the task says "commit", paths confined to repo)
  evidence.py   Objective post-pass snapshot: git status/diff, changed files,
                experiment artifacts, recent commits
  state.py      Persistent .agent_loop/ state: objective.md, state.json,
                manager_decision.json, worker_report.json, history/iter_*,
                logs/, STOP file; resume-safe, history never overwritten
  config.py     Env-driven LoopConfig (MANAGER_*/WORKER_*/MAX_*), no secrets
                in code
```

## Installation

No new dependencies - stdlib only. Use the repo venv:

```bat
C:\Pratham\neural-fabric\.venv\Scripts\python.exe -m agent_loop --check
```

## Environment

Copy `.env.example` to `.env` (never commit `.env`) or export variables:

```
MANAGER_API_KEY / WORKER_API_KEY / WORKER_BACKUP_API_KEY /
MANAGER_FALLBACK_API_KEY  (or shared OPENAI_API_KEY as last resort)
MANAGER_BASE_URL              (default https://openrouter.ai/api/v1)
MANAGER_MODEL                 (default nvidia/nemotron-3-ultra-550b-a55b:free)
MANAGER_DECISION_CHARS        (default 4000) MANAGER_REPORT_CHARS (6000)
MANAGER_EVIDENCE_CHARS        (default 8000 - evidence quality floor 500)
MANAGER_FALLBACK_ENABLED      (default true)
MANAGER_FALLBACK_BASE_URL     (no default - must be set for Kimi fallback)
MANAGER_FALLBACK_API_KEY
MANAGER_FALLBACK_MODEL        (default moonshotai/kimi-k3)
AGY_ENABLED                   (default true - AGY is the primary worker)
AGY_COMMAND                   (default agy)
AGY_MODEL                     (default empty = AGY default model)
AGY_MODE                      (default accept-edits; plan = read-only)
AGY_EFFORT                    (default empty = CLI default)
AGY_TIMEOUT_S                 (default 360 per headless AGY run)
AGY_DANGEROUSLY_SKIP_PERMISSIONS (default true on this machine by explicit
  user authorization - AGY subprocess only; loop-level git/scope/evidence
  protections stay in force)
WORKER_BASE_URL               (default https://openrouter.ai/api/v1)
WORKER_MODEL                  (default nvidia/nemotron-3-super-120b-a12b:free)
WORKER_BACKUP_API_KEY
WORKER_BACKUP_BASE_URL        (default https://integrate.api.nvidia.com/v1)
WORKER_BACKUP_MODEL           (default deepseek-ai/deepseek-v4-flash-0731)
MAX_ITERATIONS   (default 10)   MAX_TOOL_STEPS (default 12, model turns/task)
WORKER_MAX_TOOL_CALLS (default 50)  WORKER_TASK_TIMEOUT_SECONDS (default 2700)
REQUEST_TIMEOUT_S (default 120, per model-call attempt)  REQUEST_RETRIES (default 3)
TESTER_BACKEND (default opencode; opencode|api|off)
TESTER_TIMEOUT_S (default 300)  TESTER_MODEL (default empty = backend default)
STAGNATION_REPEAT (default 3)  STAGNATION_WINDOW (default 6)
MAX_FAILURES (default 3)  MAX_API_FAILURES (default 5)
SESSION_HOURS (default 24, 0 = no deadline)
```

## How to start / stop / resume

```bat
:: check only (no loop, safe without keys for repo inspection)
.venv\Scripts\python.exe -m agent_loop --check

:: offline smoke test with mocked models (no API calls)
.venv\Scripts\python.exe -m agent_loop --mock --objective "Smoke test" --yes

:: real 24-hour autonomous research session (first run)
.venv\Scripts\python.exe -m agent_loop --new-session --objective "..." --yes

:: resume after crash / API failure / restart (SAME session, deadline kept)
.venv\Scripts\python.exe -m agent_loop --resume --yes

:: shorter session, e.g. 2 hours
.venv\Scripts\python.exe -m agent_loop --new-session --hours 2 --objective "..." --yes

:: short REAL validation with fail-fast provider settings (~minutes, same models)
set REQUEST_TIMEOUT_S=30& set REQUEST_RETRIES=1& set WORKER_TASK_TIMEOUT_SECONDS=180& set MAX_API_FAILURES=2& .venv\Scripts\python.exe -m agent_loop --new-session --duration-hours 0.1 --max-iterations 2 --objective "..." --yes
```

Stop: `Ctrl+C`, or create `.agent_loop\STOP` (the loop exits gracefully at
the next boundary, state preserved). `--new-session` archives the previous
`.agent_loop` (never deletes it) and starts a fresh session clock; it is the
only way to replace the stored objective. Without it, `--resume` (or a plain
run) continues the existing session and objective.

## State persistence

`.agent_loop/` holds `objective.md` (the objective never expires),
`state.json` (counters + session + research state + timing totals), pointer
copies of the latest decisions/reports, immutable `history/iter_NNNN_*.json`
records (decisions, reports, evidence, per-attempt invocation/duration logs)
plus per-iteration summaries with a `durations` block
(`iteration_s/manager_s/worker_s/turns/tool_calls`), a `progress.json`
operator snapshot rewritten every iteration (elapsed/remaining, current and
last tasks, latest evidence/tests/failures, next direction) with an immutable
per-iteration copy, and `final_report.md`
on session end. Every model call prints a `-> role call:` start line before
blocking, a duration line after, and a retry-wait line before each backoff
sleep, so a slow endpoint is always visible.
Restart-safe: `--resume` picks up counters, objective, research state, and
the SAME absolute deadline (the clock is never reset).

## Detached experiment jobs (`lab/jobs.py`)

Three timeouts that must never be conflated: LLM call
(`REQUEST_TIMEOUT_S`), worker task (`WORKER_TASK_TIMEOUT_SECONDS`),
experiment deadline (per-job, hours allowed). A worker queues long compute
with `queue_job` (argv list, jailed paths, no shell) and returns
immediately; `poll_job` reports status/elapsed/heartbeat/log tail without
blocking. Jobs persist in `.agent_loop/jobs/`, survive LLM termination
(detached process group + redirected stdio + PID tracking), and support
heartbeats, checkpoints with resume, per-seed tracking, resource-aware
scheduling (CPU/RAM/disk/GPU probe; queue reasons recorded), deadline kill
(process tree), cancel, orphan detection, requeue, and restart recovery
(`recover()`). Provider outages never kill running compute: jobs are OS
processes, not model calls.

## Provider failure behavior

Failures are classified (`PROVIDER_RATE_LIMITED/UNAVAILABLE/TIMEOUT/
AUTH_FAILURE/ERROR`) with Retry-After + jitter. An anti-tarpit hard
deadline bounds every call even when a provider holds connections open
indefinitely. Exhaustion is recorded as BLOCKED (never as research
success); running jobs continue and orchestration resumes via `--resume`.

## Hardware

`lab.jobs.resources()` reports CPU/RAM/disk plus GPU presence. This
machine: 6 CPUs, ~8GB RAM, NVIDIA GTX 1650 (4GB) present — but the venv
torch build is CPU-only, so no GPU execution is performed or claimed.
The scheduler records GPU presence for future CUDA-capable stacks.

## Safety behavior

- Manager has no file tools by construction; worker paths are jailed to the repo.
- `git push` always refused; `reset --hard` / `clean -fd` / `checkout -- .`
  refused; `git commit` only when the task text explicitly says "commit";
  destructive shell patterns refused; commands time out (default 120 s).
- Termination: manager `complete` / `blocked`, `MAX_ITERATIONS`,
  consecutive worker failures (`MAX_FAILURES`), API failures
  (`MAX_API_FAILURES`), `STOP` file, `Ctrl+C`.

## Scientific integrity

The manager verifies worker claims against evidence artifacts and treats
failures as research data (reproduce -> diagnose -> fix/redesign -> re-test ->
re-benchmark). It distinguishes engineering completion from scientific
validation and probes hypothesis / baseline / confounders / ablation /
falsification before accepting research-phase claims. Negative results are
acceptable outcomes, never failures to hide.

## Troubleshooting

- `configuration problems` -> copy `.env.example` to `.env`, set keys.
- `connectivity ... FAILED` -> check base URL / key / network; `--mock` for offline.
- `malformed manager/worker output` warnings -> automatic corrective fallback;
  persistent ones mean the model needs narrower tasks or another provider.
- Loop exits `max_iterations` -> inspect `.agent_loop/history/` and re-run
  with `--resume` and a refined objective.
- Tests: `.venv\Scripts\python.exe -m unittest discover -s tests -v`
