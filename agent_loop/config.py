"""Configuration for the autonomous manager/worker loop.

Standalone system: direct model-provider HTTPS APIs only. No IDE, no agent
harness, no external configuration source. All secrets come from the
environment - never hardcoded, never printed.

Topology (chains NEVER cross) - SOURCE OF TRUTH 2026-09-08 (final):

  MANAGER: nvidia/nemotron-3-ultra-550b-a55b:free (OpenRouter, MANAGER ONLY)
             -- infra failure --> moonshotai/kimi-k3 (NVIDIA, MANAGER ONLY)
  WORKER:  AGY headless CLI, Gemini 3.8 Flash High (local session,
             EXECUTOR ONLY, --dangerously-skip-permissions by explicit user
             authorization on this machine)
             -- infra failure --> nvidia/nemotron-3-super-120b-a12b:free
                 (OpenRouter, EXECUTOR ONLY)
             -- infra failure --> deepseek-ai/deepseek-v4-flash-0731
                 (NVIDIA, EXECUTOR ONLY)

The manager runs on a constrained free endpoint: orchestration favors one
coherent work unit per manager call and gentle retries (Retry-After, jitter).
No OAuth material is ever configured here: AGY authenticates through the
user's own local Antigravity session.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .dotenv import load_dotenv


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
BLUESMINDS_BASE_URL = "https://api.bluesminds.com/v1"

DEFAULT_MANAGER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_MANAGER_BASE_URL = OPENROUTER_BASE_URL
DEFAULT_MANAGER_FALLBACK_MODEL = "moonshotai/kimi-k3"
# User-specified provider for the Kimi fallback (not a default assumption).
DEFAULT_MANAGER_FALLBACK_BASE_URL = NVIDIA_BASE_URL
DEFAULT_WORKER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
DEFAULT_WORKER_BASE_URL = OPENROUTER_BASE_URL
DEFAULT_WORKER_BACKUP_MODEL = "deepseek-ai/deepseek-v4-flash-0731"
DEFAULT_WORKER_BACKUP_BASE_URL = NVIDIA_BASE_URL

# AGY headless worker (primary executor), Gemini 3.8 Flash High by explicit
# user choice (capability over token conservation). No credentials: AGY
# authenticates through the user's own local Antigravity session. Empty
# AGY_MODEL means "let the authenticated AGY environment select its default
# model" - but the configured default below is Flash High, do not downgrade
# it to save tokens.
DEFAULT_AGY_ENABLED = True
DEFAULT_AGY_COMMAND = "agy"
DEFAULT_AGY_MODEL = "gemini-3.8-flash-high"
DEFAULT_AGY_MODE = "accept-edits"
DEFAULT_AGY_EFFORT = "high"
DEFAULT_AGY_TIMEOUT_S = 360.0
# User-authorized on this machine (backup box, explicit instruction):
# AGY runs with --dangerously-skip-permissions so permission prompts never
# block autonomous engineering. This covers the AGY subprocess ONLY - all
# loop-level protections (repo scope, git guards, timeouts, evidence,
# no secret logging) stay in force. Set AGY_DANGEROUSLY_SKIP_PERMISSIONS=false
# to return to scoped --mode operation.
DEFAULT_AGY_DANGEROUS = True

# Manager context economy (orchestration-level, never at the cost of
# load-bearing evidence): hard character caps for each manager-input section.
# Large raw evidence stays persisted on disk in .agent_loop/history/.
DEFAULT_MANAGER_DECISION_CHARS = 4000
DEFAULT_MANAGER_REPORT_CHARS = 6000
DEFAULT_MANAGER_EVIDENCE_CHARS = 8000

# v2 independent tester layer (manager -> builder -> tester -> manager).
# opencode = headless `opencode run` subprocess (read-only, no --auto);
# api = adversarial review via the manager-fallback provider (review only);
# off = deterministic gates only (LLM verdict recorded INCONCLUSIVE).
DEFAULT_TESTER_BACKEND = "opencode"
DEFAULT_TESTER_TIMEOUT_S = 300.0
DEFAULT_TESTER_MODEL = ""
DEFAULT_TESTER_CRITIC = False
DEFAULT_STAGNATION_REPEAT = 3
DEFAULT_STAGNATION_WINDOW = 6

DEFAULT_MAX_ITERATIONS = 10
DEFAULT_MAX_TOOL_STEPS = 12
DEFAULT_WORKER_MAX_TOOL_CALLS = 50
DEFAULT_WORKER_TASK_TIMEOUT_S = 2700.0  # 45 min wall clock per worker task
DEFAULT_SESSION_HOURS = 24.0
# Per-attempt model HTTP timeout / client-level transient retries. Fail-slow
# hangs (provider accepts the request but never answers) are the dominant
# live-loop stall: keep these bounded and fail fast instead of silently
# blocking. Short validation runs MUST set small values (e.g.
# REQUEST_TIMEOUT_S=30, REQUEST_RETRIES=1); full sessions use the defaults.
DEFAULT_REQUEST_TIMEOUT_S = 120.0
DEFAULT_REQUEST_RETRIES = 3


def provider_name(base_url: str) -> str:
    """Short provider label for invocation records / logs (no secrets)."""
    low = (base_url or "").lower()
    if "integrate.api.nvidia.com" in low:
        return "nvidia"
    if "openrouter.ai" in low:
        return "openrouter"
    if "api.bluesminds.com" in low:
        return "bluesminds"
    return "custom"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


@dataclass
class LoopConfig:
    manager_api_key: str = ""
    manager_base_url: str = DEFAULT_MANAGER_BASE_URL
    manager_model: str = DEFAULT_MANAGER_MODEL
    manager_fallback_enabled: bool = True
    manager_fallback_api_key: str = ""
    manager_fallback_base_url: str = DEFAULT_MANAGER_FALLBACK_BASE_URL
    manager_fallback_model: str = DEFAULT_MANAGER_FALLBACK_MODEL
    worker_api_key: str = ""
    worker_base_url: str = DEFAULT_WORKER_BASE_URL
    worker_model: str = DEFAULT_WORKER_MODEL
    worker_backup_api_key: str = ""
    worker_backup_base_url: str = DEFAULT_WORKER_BACKUP_BASE_URL
    worker_backup_model: str = DEFAULT_WORKER_BACKUP_MODEL
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_tool_steps: int = DEFAULT_MAX_TOOL_STEPS  # max model turns per worker task
    worker_max_tool_calls: int = DEFAULT_WORKER_MAX_TOOL_CALLS  # total tool invocations per task
    worker_task_timeout_s: float = DEFAULT_WORKER_TASK_TIMEOUT_S
    session_hours: float = DEFAULT_SESSION_HOURS  # absolute session budget; <=0 disables deadline
    max_consecutive_failures: int = 3
    max_api_failures: int = 5
    request_timeout_s: float = 120.0
    request_retries: int = 3
    agy_enabled: bool = True
    agy_command: str = DEFAULT_AGY_COMMAND
    agy_model: str = DEFAULT_AGY_MODEL
    agy_mode: str = DEFAULT_AGY_MODE
    agy_effort: str = DEFAULT_AGY_EFFORT
    agy_timeout_s: float = DEFAULT_AGY_TIMEOUT_S
    agy_dangerous: bool = DEFAULT_AGY_DANGEROUS
    manager_decision_chars: int = DEFAULT_MANAGER_DECISION_CHARS
    manager_report_chars: int = DEFAULT_MANAGER_REPORT_CHARS
    manager_evidence_chars: int = DEFAULT_MANAGER_EVIDENCE_CHARS
    tester_backend: str = DEFAULT_TESTER_BACKEND
    tester_timeout_s: float = DEFAULT_TESTER_TIMEOUT_S
    tester_model: str = DEFAULT_TESTER_MODEL
    tester_critic: bool = DEFAULT_TESTER_CRITIC
    stagnation_repeat: int = DEFAULT_STAGNATION_REPEAT
    stagnation_window: int = DEFAULT_STAGNATION_WINDOW
    state_dir: str = ".agent_loop"
    repo_root: str = "."
    dry_run: bool = False  # when True, worker shell/file tools still run but git commits are blocked

    @classmethod
    def from_env(cls, repo_root: str = ".", state_dir: str = ".agent_loop") -> "LoopConfig":
        # Precedence: 1) explicit process env  2) <repo>/.env  3) defaults.
        # The loader never overwrites existing process variables and never
        # exposes values; missing .env is a silent no-op.
        load_dotenv(os.path.join(str(repo_root), ".env"))
        # Shared credentials are a last resort: any explicit per-role value wins.
        shared_key = os.environ.get("OPENAI_API_KEY", "")
        shared_base = os.environ.get("OPENAI_BASE_URL", "")

        def _key(*names: str) -> str:
            for n in names:
                v = os.environ.get(n, "")
                if v:
                    return v
            return shared_key

        def _url(*names: str, default: str = "") -> str:
            for n in names:
                v = os.environ.get(n, "")
                if v:
                    return v
            if shared_base and default in (NVIDIA_BASE_URL, OPENROUTER_BASE_URL):
                return shared_base
            return default

        return cls(
            manager_api_key=_key("MANAGER_API_KEY"),
            manager_base_url=_url("MANAGER_BASE_URL", default=DEFAULT_MANAGER_BASE_URL),
            manager_model=os.environ.get("MANAGER_MODEL", DEFAULT_MANAGER_MODEL),
            manager_fallback_enabled=_env_flag("MANAGER_FALLBACK_ENABLED", True),
            manager_fallback_api_key=_key("MANAGER_FALLBACK_API_KEY"),
            # No shared-base fill here: Kimi's provider URL is unknown, never assume one.
            manager_fallback_base_url=os.environ.get(
                "MANAGER_FALLBACK_BASE_URL", DEFAULT_MANAGER_FALLBACK_BASE_URL),
            manager_fallback_model=os.environ.get(
                "MANAGER_FALLBACK_MODEL", DEFAULT_MANAGER_FALLBACK_MODEL),
            worker_api_key=_key("WORKER_API_KEY"),
            worker_base_url=_url("WORKER_BASE_URL", default=DEFAULT_WORKER_BASE_URL),
            worker_model=os.environ.get("WORKER_MODEL", DEFAULT_WORKER_MODEL),
            worker_backup_api_key=_key("WORKER_BACKUP_API_KEY"),
            worker_backup_base_url=_url("WORKER_BACKUP_BASE_URL",
                                        default=DEFAULT_WORKER_BACKUP_BASE_URL),
            worker_backup_model=os.environ.get("WORKER_BACKUP_MODEL",
                                               DEFAULT_WORKER_BACKUP_MODEL),
            max_iterations=int(os.environ.get("MAX_ITERATIONS", str(DEFAULT_MAX_ITERATIONS))),
            max_tool_steps=int(os.environ.get("MAX_TOOL_STEPS", str(DEFAULT_MAX_TOOL_STEPS))),
            worker_max_tool_calls=int(os.environ.get(
                "WORKER_MAX_TOOL_CALLS", str(DEFAULT_WORKER_MAX_TOOL_CALLS))),
            worker_task_timeout_s=float(os.environ.get(
                "WORKER_TASK_TIMEOUT_SECONDS", str(DEFAULT_WORKER_TASK_TIMEOUT_S))),
            session_hours=float(os.environ.get("SESSION_HOURS", str(DEFAULT_SESSION_HOURS))),
            max_consecutive_failures=int(os.environ.get("MAX_FAILURES", "3")),
            max_api_failures=int(os.environ.get("MAX_API_FAILURES", "5")),
            request_timeout_s=float(os.environ.get(
                "REQUEST_TIMEOUT_S", str(DEFAULT_REQUEST_TIMEOUT_S))),
            request_retries=int(os.environ.get(
                "REQUEST_RETRIES", str(DEFAULT_REQUEST_RETRIES))),
            agy_enabled=_env_flag("AGY_ENABLED", DEFAULT_AGY_ENABLED),
            agy_command=os.environ.get("AGY_COMMAND", DEFAULT_AGY_COMMAND),
            agy_model=os.environ.get("AGY_MODEL", DEFAULT_AGY_MODEL),
            agy_mode=os.environ.get("AGY_MODE", DEFAULT_AGY_MODE),
            agy_effort=os.environ.get("AGY_EFFORT", DEFAULT_AGY_EFFORT),
            agy_timeout_s=float(os.environ.get(
                "AGY_TIMEOUT_S", str(DEFAULT_AGY_TIMEOUT_S))),
            agy_dangerous=_env_flag("AGY_DANGEROUSLY_SKIP_PERMISSIONS",
                                     DEFAULT_AGY_DANGEROUS),
            manager_decision_chars=int(os.environ.get(
                "MANAGER_DECISION_CHARS", str(DEFAULT_MANAGER_DECISION_CHARS))),
            manager_report_chars=int(os.environ.get(
                "MANAGER_REPORT_CHARS", str(DEFAULT_MANAGER_REPORT_CHARS))),
            manager_evidence_chars=int(os.environ.get(
                "MANAGER_EVIDENCE_CHARS", str(DEFAULT_MANAGER_EVIDENCE_CHARS))),
            tester_backend=os.environ.get(
                "TESTER_BACKEND", DEFAULT_TESTER_BACKEND).strip().lower(),
            tester_timeout_s=float(os.environ.get(
                "TESTER_TIMEOUT_S", str(DEFAULT_TESTER_TIMEOUT_S))),
            tester_model=os.environ.get("TESTER_MODEL", DEFAULT_TESTER_MODEL),
            tester_critic=_env_flag("TESTER_CRITIC", DEFAULT_TESTER_CRITIC),
            stagnation_repeat=int(os.environ.get(
                "STAGNATION_REPEAT", str(DEFAULT_STAGNATION_REPEAT))),
            stagnation_window=int(os.environ.get(
                "STAGNATION_WINDOW", str(DEFAULT_STAGNATION_WINDOW))),
            state_dir=os.environ.get("STATE_DIR", state_dir),
            repo_root=repo_root,
        )

    def validate(self) -> list:
        """Human-readable configuration errors (empty = ok to run)."""
        errors = []
        if not self.manager_api_key:
            errors.append("MANAGER_API_KEY (or shared OPENAI_API_KEY) is not set.")
        if not self.worker_api_key:
            errors.append("WORKER_API_KEY (or shared OPENAI_API_KEY) is not set.")
        if not self.worker_backup_api_key:
            errors.append("WORKER_BACKUP_API_KEY (or shared OPENAI_API_KEY) is not set; "
                          "worker fallback would be unavailable.")
        for label, url in (("MANAGER_BASE_URL", self.manager_base_url),
                           ("WORKER_BASE_URL", self.worker_base_url),
                           ("WORKER_BACKUP_BASE_URL", self.worker_backup_base_url)):
            if not url.startswith("http"):
                errors.append("%s looks invalid: %r" % (label, url))
            elif "https:https" in url or "http:http" in url:
                errors.append("%s has a duplicated scheme: %r"
                              % (label, url))
        if self.manager_fallback_enabled:
            if not self.manager_fallback_base_url:
                errors.append(
                    "MANAGER_FALLBACK_BASE_URL is required but missing: no default "
                    "provider exists for %r. Set MANAGER_FALLBACK_BASE_URL (and "
                    "MANAGER_FALLBACK_API_KEY), or set MANAGER_FALLBACK_ENABLED=false "
                    "to run without manager fallback." % self.manager_fallback_model)
            elif not self.manager_fallback_base_url.startswith("http"):
                errors.append("MANAGER_FALLBACK_BASE_URL looks invalid: %r"
                              % self.manager_fallback_base_url)
            elif not self.manager_fallback_api_key:
                errors.append("MANAGER_FALLBACK_API_KEY is not set; manager fallback "
                              "to %r would be unavailable." % self.manager_fallback_model)
        if self.max_iterations < 1:
            errors.append("MAX_ITERATIONS must be >= 1.")
        if self.max_tool_steps < 1:
            errors.append("MAX_TOOL_STEPS must be >= 1.")
        if self.worker_max_tool_calls < 1:
            errors.append("WORKER_MAX_TOOL_CALLS must be >= 1.")
        if self.worker_task_timeout_s <= 0:
            errors.append("WORKER_TASK_TIMEOUT_SECONDS must be > 0.")
        if self.request_timeout_s <= 0:
            errors.append("REQUEST_TIMEOUT_S must be > 0.")
        if self.request_retries < 1:
            errors.append("REQUEST_RETRIES must be >= 1.")
        if self.agy_enabled:
            if not self.agy_command:
                errors.append("AGY_COMMAND is required but missing while "
                              "AGY_ENABLED is true.")
            if self.agy_mode not in ("accept-edits", "plan"):
                errors.append("AGY_MODE must be accept-edits or plan, got %r."
                              % self.agy_mode)
            if self.agy_timeout_s <= 0:
                errors.append("AGY_TIMEOUT_S must be > 0.")
        for label, val in (("MANAGER_DECISION_CHARS",
                            self.manager_decision_chars),
                           ("MANAGER_REPORT_CHARS", self.manager_report_chars),
                           ("MANAGER_EVIDENCE_CHARS",
                            self.manager_evidence_chars)):
            if val < 500:
                errors.append("%s must be >= 500 (evidence quality floor)."
                              % label)
        if self.tester_backend not in ("opencode", "api", "off"):
            errors.append("TESTER_BACKEND must be opencode|api|off, got %r."
                          % self.tester_backend)
        if self.tester_timeout_s <= 0:
            errors.append("TESTER_TIMEOUT_S must be > 0.")
        if self.stagnation_repeat < 2:
            errors.append("STAGNATION_REPEAT must be >= 2.")
        if self.stagnation_window < self.stagnation_repeat:
            errors.append("STAGNATION_WINDOW must be >= STAGNATION_REPEAT.")
        if self.session_hours < 0:
            errors.append("SESSION_HOURS must be >= 0 (0 disables the deadline).")
        return errors

    def redacted(self) -> dict:
        """Config snapshot safe for logs (keys redacted, never printed raw)."""
        def _red(k):
            return ("*" * 4 + k[-4:]) if len(k) > 8 else ("****" if k else "")
        return {
            "manager": {"model": self.manager_model,
                        "provider": provider_name(self.manager_base_url),
                        "base_url": self.manager_base_url,
                        "api_key": _red(self.manager_api_key)},
            "manager_fallback": {"enabled": self.manager_fallback_enabled,
                                 "model": self.manager_fallback_model,
                                 "provider": provider_name(self.manager_fallback_base_url),
                                 "base_url": self.manager_fallback_base_url or "(not configured)",
                                 "api_key": _red(self.manager_fallback_api_key)},
            "worker": {"model": self.worker_model,
                       "provider": provider_name(self.worker_base_url),
                       "base_url": self.worker_base_url,
                       "api_key": _red(self.worker_api_key)},
            "worker_backup": {"model": self.worker_backup_model,
                              "provider": provider_name(self.worker_backup_base_url),
                              "base_url": self.worker_backup_base_url,
                              "api_key": _red(self.worker_backup_api_key)},
            "agy": {"enabled": self.agy_enabled,
                    "command": self.agy_command,
                    "model": self.agy_model or "(agy default)",
                    "mode": self.agy_mode,
                    "effort": self.agy_effort or "(default)",
                    "timeout_s": self.agy_timeout_s,
                    "dangerously_skip_permissions": self.agy_dangerous},
            "tester": {"backend": self.tester_backend,
                       "timeout_s": self.tester_timeout_s,
                       "model": self.tester_model or "(default)",
                       "stagnation_repeat": self.stagnation_repeat,
                       "stagnation_window": self.stagnation_window},
            "max_iterations": self.max_iterations,
            "max_tool_steps": self.max_tool_steps,
            "worker_max_tool_calls": self.worker_max_tool_calls,
            "worker_task_timeout_s": self.worker_task_timeout_s,
            "request_timeout_s": self.request_timeout_s,
            "request_retries": self.request_retries,
            "session_hours": self.session_hours,
            "state_dir": self.state_dir,
        }
