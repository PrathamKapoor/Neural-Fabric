"""Per-role fallback chains. Chains NEVER cross.

  MANAGER chain: Nemotron 3 Ultra free (OpenRouter) -> Kimi K3 (NVIDIA).
                 Both are MANAGER ONLY. The manager chain NEVER receives tools.
  WORKER chain:  AGY headless CLI, Gemini 3.8 Flash High (local) ->
                 Nemotron 3 Super (OpenRouter) -> DeepSeek V4 Flash (NVIDIA).
                 The API fallbacks are WORKER ONLY and always receive the
                 worker tools.

Fallback triggers ONLY on infrastructure failures that prevent reliable model
invocation: connection/DNS/TLS errors, timeouts, HTTP 5xx, provider or model
unavailable, rate limiting, malformed provider responses, unrecoverable
API/protocol failures, and (for AGY) missing binary / broken session /
permission denial / quota / CLI timeout / nonzero CLI exit.

Every failure is labeled with a provider state so the operator (and the
manager's history view) can distinguish quota/availability/timeout/auth
conditions from genuine task outcomes:
PROVIDER_RATE_LIMITED | PROVIDER_UNAVAILABLE | PROVIDER_TIMEOUT |
PROVIDER_AUTH_FAILURE | PROVIDER_ERROR. A bad TASK result is never labeled:
a valid HTTP response NEVER triggers fallback, whatever its content.
                 Flash is WORKER ONLY and always receives the worker tools.

Fallback triggers ONLY on infrastructure failures that prevent reliable model
invocation: connection/DNS/TLS errors, timeouts, HTTP 5xx, provider or model
unavailable, rate limiting, malformed provider responses, unrecoverable
API/protocol failures.

A successfully returned response with a bad TASK result (wrong code, invalid
JSON report, poor decision) is a TASK FAILURE, not an infrastructure failure:
no automatic fallback happens. The result flows to evidence -> manager review,
which decides retry/reformulate/reassign.

Every invocation records {role, provider, model, attempt, status, reason,
fallback_from} for state/history observability. Fallback usage is never hidden.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .clients import ApiError, TransientApiError


@dataclass
class Invocation:
    role: str
    provider: str
    model: str
    attempt: int
    status: str  # "success" | "infrastructure_failure"
    reason: str = ""
    fallback_from: str = ""
    duration_s: float = 0.0  # wall time of this single model call

    def as_dict(self) -> dict:
        return {"role": self.role, "provider": self.provider,
                "model": self.model, "attempt": self.attempt,
                "status": self.status, "reason": self.reason,
                "fallback_from": self.fallback_from,
                "duration_s": round(self.duration_s, 1)}


class ChainExhausted(RuntimeError):
    """Both primary and fallback failed with infrastructure errors."""

    def __init__(self, invocations: list):
        self.invocations = invocations
        reasons = "; ".join("%s: %s" % (i.model, i.reason) for i in invocations)
        super().__init__("all providers in chain failed [%s]" % reasons)


def _retry_cb(client, on_retry):
    """Adapt the client-level on_retry(attempt, wait_s, error) hook."""
    if on_retry is None:
        return None

    def _cb(attempt, wait_s, error):
        try:
            on_retry(getattr(client, "provider", "?"),
                     getattr(client, "model", "?"),
                     attempt, wait_s, error)
        except Exception:
            pass

    return _cb


def _provider_state(error: Exception) -> str:
    """Classify an infrastructure error for invocation records.

    Never applied to valid responses: only raised exceptions reach here, so
    a bad task result can never be mislabeled as a provider condition.
    """
    from .clients import ApiError as _ApiError
    from .clients import TransientApiError as _Transient
    text = str(error)
    code = None
    if "HTTP " in text:
        try:
            code = int(text.split("HTTP ", 1)[1].split(":", 1)[0].strip())
        except (ValueError, IndexError):
            code = None
    if isinstance(error, _Transient):
        if code == 429:
            return "PROVIDER_RATE_LIMITED"
        if code is not None and 500 <= code <= 599:
            return "PROVIDER_UNAVAILABLE"
        return "PROVIDER_TIMEOUT"
    if isinstance(error, _ApiError):
        if code in (401, 403):
            return "PROVIDER_AUTH_FAILURE"
        return "PROVIDER_ERROR"
    return "PROVIDER_TIMEOUT"


class RoleChain:
    """One role (manager OR worker) with primary + ordered fallback clients.

    `fallback` accepts a single client or a list (worker chain uses two:
    Nemotron Super, then DeepSeek Flash). `.fallback` stays as the first
    fallback for backward compatibility; `.fallbacks` is the full list.
    """

    def __init__(self, role: str, primary, fallback=None):
        assert role in ("manager", "worker"), role
        self.role = role
        self.primary = primary
        if fallback is None:
            self.fallbacks: list = []
        elif isinstance(fallback, (list, tuple)):
            self.fallbacks = list(fallback)
        else:
            self.fallbacks = [fallback]
        self.fallback = self.fallbacks[0] if self.fallbacks else None

    @property
    def models(self) -> list:
        return [self.primary.model] + [f.model for f in self.fallbacks]

    def chat(self, messages: list, tools: list | None = None,
             force_json: bool = False, on_call_start=None,
             on_retry=None) -> tuple:
        """Return (assistant_message, [Invocation, ...]).

        `on_call_start(provider, model, attempt, is_fallback)` is invoked
        BEFORE each blocking model call so the operator always sees what the
        loop is waiting on. `on_retry(provider, model, attempt, wait_s,
        error)` fires before each backoff sleep so fail-slow hangs are
        visible instead of silent. Raises ChainExhausted if every client
        fails infrastructurally. A valid HTTP response NEVER triggers
        fallback, whatever its content.
        """
        if self.role == "manager" and tools:
            raise AssertionError("manager chain must never receive worker tools")
        invocations: list = []
        clients = [self.primary] + list(self.fallbacks)
        attempt = 0
        for idx, client in enumerate(clients):
            attempt += 1
            if on_call_start is not None:
                try:
                    on_call_start(getattr(client, "provider", "?"),
                                  getattr(client, "model", "?"), attempt,
                                  idx > 0)
                except Exception:
                    pass
            t0 = time.time()
            try:
                msg = client.chat(messages, tools=tools,
                                  force_json=force_json,
                                  on_retry=_retry_cb(client, on_retry))
            except (TransientApiError, ApiError, TimeoutError, OSError) as e:
                invocations.append(Invocation(
                    role=self.role,
                    provider=getattr(client, "provider", "?"),
                    model=getattr(client, "model", "?"),
                    attempt=attempt, status="infrastructure_failure",
                    reason="[%s] %s: %s" % (_provider_state(e),
                                            type(e).__name__, str(e)[:300]),
                    fallback_from=clients[idx - 1].model if idx > 0 else "",
                    duration_s=time.time() - t0))
                continue
            invocations.append(Invocation(
                role=self.role,
                provider=getattr(client, "provider", "?"),
                model=getattr(client, "model", "?"),
                attempt=attempt, status="success",
                reason="",
                fallback_from=clients[idx - 1].model if idx > 0 else "",
                duration_s=time.time() - t0))
            return msg, invocations
        raise ChainExhausted(invocations)
