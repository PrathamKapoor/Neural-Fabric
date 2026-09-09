"""CLI: python -m agent_loop [--objective "..."] [--objective-file F] ...

Standalone: direct provider HTTPS APIs only. No IDE, harness, or external
process required. Runnable from C:\\Pratham\\neural-fabric with the repo venv:

  .venv\\Scripts\\python.exe -m agent_loop --check
  .venv\\Scripts\\python.exe -m agent_loop --mock --objective "..." --yes

First run: verifies config -> probes all configured model endpoints ->
inspects the repo -> initializes .agent_loop/ state -> accepts the objective
-> starts the loop.

Stop anytime with Ctrl+C or by creating the file .agent_loop/STOP.
`--check` only verifies config + connectivity + repo inspection (no loop).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent_loop.clients import MockClient  # noqa: E402
from agent_loop.config import LoopConfig  # noqa: E402
from agent_loop.dotenv import dotenv_info  # noqa: E402
from agent_loop.loop import AgentLoop, TerminalUI, build_chains, ensure_utf8_stdio  # noqa: E402
from agent_loop.roles import RoleChain  # noqa: E402
from agent_loop.state import LoopState  # noqa: E402


def inspect_repo(repo_root: Path) -> dict:
    info: dict = {"root": str(repo_root)}
    try:
        p = subprocess.run(["git", "status", "--short", "--branch"],
                           cwd=str(repo_root), capture_output=True,
                           text=True, timeout=20)
        info["git_status"] = (p.stdout or "").strip() or "(clean)"
    except Exception as e:  # noqa: BLE001
        info["git_status"] = "(git unavailable: %s)" % e
    for key, rel in (("has_fabric", "fabric"), ("has_backend", "backend"),
                     ("has_experiments", "experiments"), ("has_tests", "tests"),
                     ("has_docs", "docs"), ("has_agent_loop", "agent_loop")):
        info[key] = (repo_root / rel).is_dir()
    info["experiment_scripts"] = sorted(
        p.name for p in (repo_root / "experiments").glob("*.py")) \
        if (repo_root / "experiments").is_dir() else []
    return info


def print_repo_inspection(info: dict):
    print("repository: %s" % info["root"])
    print("  fabric/backend/experiments/tests/docs/agent_loop: %s/%s/%s/%s/%s/%s" % (
        info.get("has_fabric"), info.get("has_backend"),
        info.get("has_experiments"), info.get("has_tests"),
        info.get("has_docs"), info.get("has_agent_loop")))
    print("  git: %s" % str(info.get("git_status", ""))[:300])
    print("  experiments: %s" % ", ".join(info.get("experiment_scripts", [])[:12]))


def resolve_objective(args) -> str:
    if args.objective:
        return args.objective
    if args.objective_file:
        return Path(args.objective_file).read_text(encoding="utf-8").strip()
    existing = LoopState(REPO_ROOT / args.state_dir).get_objective()
    if existing and args.resume:
        return existing
    if existing:
        print("previous objective found in %s/objective.md" % args.state_dir)
        print("  %s" % existing[:200])
        if args.yes:
            return existing
        ans = input("reuse it? [Y/n/new text]: ").strip()
        if ans.lower() not in ("n", "no"):
            return existing if ans.lower() in ("", "y", "yes") else ans
    return input("objective: ").strip()


def _probe(label: str, client) -> bool:
    res = client.check_connectivity()
    if res.get("ok"):
        extra = ""
        if res.get("model_listed") is False:
            extra = " (WARNING: configured model not in provider model list)"
        print("connectivity %s [%s] %s: OK%s" % (
            label, res.get("provider"), res.get("model"), extra))
        return True
    print("connectivity %s [%s] %s: FAILED: %s" % (
        label, res.get("provider"), res.get("model"), res.get("error")))
    return False


def main(argv=None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="Neural Fabric autonomous manager/worker loop")
    ap.add_argument("--objective", default="", help="User objective text")
    ap.add_argument("--objective-file", default="", help="File containing the objective")
    ap.add_argument("--max-iterations", type=int, default=0, help="Override MAX_ITERATIONS")
    ap.add_argument("--state-dir", default=".agent_loop")
    ap.add_argument("--resume", action="store_true", help="Resume existing state")
    ap.add_argument("--yes", action="store_true", help="Non-interactive: reuse stored objective")
    ap.add_argument("--check", action="store_true", help="Config + connectivity + repo check only")
    ap.add_argument("--mock", action="store_true",
                    help="Offline smoke test with mocked models (no API calls)")
    ap.add_argument("--hours", type=float, default=0.0,
                    help="Session time budget in hours (overrides SESSION_HOURS)")
    ap.add_argument("--duration-hours", type=float, default=0.0,
                    help="Alias for --hours (short validation, e.g. 0.1)")
    ap.add_argument("--new-session", action="store_true",
                    help="Archive existing .agent_loop state and start a fresh "
                         "session clock (required to replace the objective)")
    args = ap.parse_args(argv)

    cfg = LoopConfig.from_env(repo_root=str(REPO_ROOT), state_dir=args.state_dir)
    if args.max_iterations:
        cfg.max_iterations = args.max_iterations
    if args.hours or args.duration_hours:
        cfg.session_hours = args.hours or args.duration_hours

    # Fresh session handling: archive (never delete) prior state incl. history.
    state_path = REPO_ROOT / args.state_dir
    if args.new_session and state_path.exists():
        archive = REPO_ROOT / (args.state_dir + "_archive_%d"
                               % int(time.time()))
        shutil.move(str(state_path), str(archive))
        print("archived previous session to %s" % archive)

    # .env source line: existence + key COUNT only, never names or values.
    _env_exists, _env_n = dotenv_info(REPO_ROOT / ".env")
    print(".env: %s (%s)" % (REPO_ROOT / ".env",
                             ("%d keys" % _env_n) if _env_exists else "not present"))

    # 1. config ---------------------------------------------------------
    errors = cfg.validate()
    if errors and not (args.check or args.mock):
        print("configuration problems:")
        for e in errors:
            print("  - %s" % e)
        print("see .env.example; Never commit .env.")
        return 2
    elif errors:
        print("(check/mock mode) configuration problems (would block a real run):")
        for e in errors:
            print("  - %s" % e)
    print("topology (keys redacted):")
    red = cfg.redacted()
    agy = red["agy"]
    print("  WORKER PRIMARY: agy [%s] mode=%s timeout=%.0fs -> %s" % (
        agy["command"], agy["mode"], agy["timeout_s"],
        "enabled" if agy["enabled"] else "disabled"))
    tst = red["tester"]
    print("  TESTER: backend=%s timeout=%.0fs stagnation=%dx/%d -> %s" % (
        tst["backend"], tst["timeout_s"], tst["stagnation_repeat"],
        tst["stagnation_window"],
        "enabled" if tst["backend"] != "off" else "gates-only"))
    rows = (("MANAGER PRIMARY", red["manager"], True),
            ("MANAGER FALLBACK", red["manager_fallback"],
             cfg.manager_fallback_enabled),
            ("WORKER FALLBACK 1", red["worker"], True),
            ("WORKER FALLBACK 2", red["worker_backup"], True))
    for label, c, enabled in rows:
        if not enabled:
            status = "disabled"
        elif c.get("api_key"):
            status = "configured"
        else:
            status = "missing"
        print("  %s: %s [%s] %s -> %s" % (
            label, c.get("model"), c.get("provider"),
            c.get("base_url"), status))

    # 2. repo inspection --------------------------------------------------
    info = inspect_repo(REPO_ROOT)
    print_repo_inspection(info)

    # 3. clients + connectivity --------------------------------------------
    if args.mock:
        cfg.tester_backend = "off"  # mock = hermetic: gates only, no subprocess
        manager = RoleChain("manager", MockClient(
            [{"role": "assistant", "content": _MOCK_MANAGER}],
            model="mock-manager", provider="mock"))
        worker = RoleChain("worker", MockClient(
            [{"role": "assistant", "content": _MOCK_WORKER}],
            model="mock-worker", provider="mock"))
        print("connectivity: MOCK mode (no network)")
    else:
        try:
            manager, worker = build_chains(cfg)
        except ValueError as e:
            print("configuration error: %s" % e)
            return 2
        ok = True
        ok &= _probe("manager primary", manager.primary)
        if manager.fallback is not None:
            ok &= _probe("manager fallback", manager.fallback)
        else:
            print("connectivity manager fallback: DISABLED")
        ok &= _probe("worker primary", worker.primary)
        if not worker.fallbacks:
            print("connectivity worker fallbacks: NONE CONFIGURED")
        for i, fb in enumerate(worker.fallbacks):
            ok &= _probe("worker fallback %d" % (i + 1), fb)
        if not ok:
            print("WARNING: one or more endpoints unreachable; "
                  "fallback chains cover infrastructure failure only.")
    if args.check:
        print("check complete.")
        return 0 if not errors else 2

    # 4. state + objective --------------------------------------------------
    state = LoopState(REPO_ROOT / args.state_dir)
    objective = resolve_objective(args)
    if not objective:
        print("no objective given - nothing to do.")
        return 2

    # 5. loop (Ctrl+C / STOP file safe) --------------------------------------
    loop = AgentLoop(cfg, manager, worker, state=state, ui=TerminalUI())
    try:
        summary = loop.run(objective)
    except KeyboardInterrupt:
        print("\ninterrupted by user (Ctrl+C) - state preserved; resume with --resume.")
        return 130
    print("done: %s" % summary)
    return 0 if summary.get("status") == "complete" else 1


_MOCK_MANAGER = ('{"status": "continue", "assessment": "mock assessment", '
                 '"completed": [], "gaps": [], "problems": [], '
                 '"next_task": "Inspect fabric/engine.py via read_file and report its contents.", '
                 '"expected_result": "A worker report describing the engine.", '
                 '"validation": ["worker used read_file"], "constraints": [], "reason": "mock"}')

_MOCK_WORKER = ('{"status": "done", "task": "inspect engine", "changes": [], '
                '"tests": [], "commands": [], "evidence": [], "artifacts": [], '
                '"problems": [], "recommendation": "continue"}')

if __name__ == "__main__":
    raise SystemExit(main())
