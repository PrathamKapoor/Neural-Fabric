"""Isolated model-provider adapter (direct HTTPS, stdlib only).

Everything here speaks OpenAI-compatible `POST {base_url}/chat/completions`
over urllib, so the rest of the system never imports provider-specific SDKs
and needs no IDE, agent harness, or external process. Each role gets its own
client instances; fallback chains live in roles.py.

Capability differences are normalized here:
- `response_format: json_object` is attempted when asked, but if the provider
  rejects it the SAME client retries once without it (never assume support).
- Provider-specific extras (reasoning_content, etc.) are preserved on the
  message and ignored by parsers.
- Malformed provider responses (missing choices/message) raise ApiError so the
  fallback chain treats them as infrastructure failures.

Retry economy (the manager runs on a constrained free endpoint):
- Transient retries use exponential backoff WITH jitter (no thundering herd).
- A provider `Retry-After` response header is honored (capped), so 429s back
  off exactly as long as asked instead of burning quota on fixed sleeps.
"""
from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request

from .tls import create_ssl_context


class ApiError(RuntimeError):
    """Non-transient provider error (bad key, bad request, unknown model...).

    Still fallback-eligible at the chain level (a dead/erroring provider is an
    infrastructure problem for this system), except when it stems from our own
    malformed request - the chain tries the next provider once anyway and then
    surfaces the failure.
    """


class TransientApiError(RuntimeError):
    """Transient failure worth retrying (429 / 5xx / network blip)."""


TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# Upper bound for honoring a provider Retry-After header: beyond this the
# loop's own budgets (not the provider's ask) govern.
MAX_RETRY_AFTER_S = 120.0

_JSON_REJECT_HINTS = ("response_format", "json_object", "response format",
                      "unsupported parameter", "unsupported value")


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat client (stdlib only, no extra deps)."""

    def __init__(self, api_key: str, base_url: str, model: str,
                 provider: str = "", timeout_s: float = 120.0, retries: int = 3):
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.provider = provider or _infer_provider(self.base_url)
        self.timeout_s = timeout_s
        self.retries = retries

    # -- low level -----------------------------------------------------
    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s,
                                        context=create_ssl_context()) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8")[:2000]
            except Exception:
                detail = ""
            if e.code in TRANSIENT_STATUS:
                err = TransientApiError("HTTP %s: %s" % (e.code, detail))
                err.retry_after = _retry_after_s(e.headers)
                raise err
            raise ApiError("HTTP %s: %s" % (e.code, detail))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise TransientApiError("network error: %s" % e)
        try:
            return json.loads(body)
        except ValueError as e:
            raise ApiError("malformed provider response (not JSON): %s" % e)

    # -- public --------------------------------------------------------
    def chat(self, messages: list, tools: list | None = None,
             force_json: bool = False, on_retry=None) -> dict:
        """Return the raw assistant `message` dict (may contain tool_calls).

        Retries transient failures with exponential backoff. If the provider
        rejects `response_format`, retries once without it. Raises ApiError
        (permanent) or TransientApiError (after retries exhausted).

        `on_retry(attempt, wait_s, error)` is called BEFORE each backoff
        sleep so the operator sees fail-slow hangs instead of silence.
        """
        payload = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if force_json:
            payload["response_format"] = {"type": "json_object"}
        try:
            return self._chat_once(payload, on_retry=on_retry)
        except ApiError as e:
            if force_json and _looks_like_json_rejection(str(e)):
                slim = dict(payload)
                slim.pop("response_format", None)
                return self._chat_once(slim, on_retry=on_retry)
            raise

    def _chat_once(self, payload: dict, on_retry=None) -> dict:
        last: Exception | None = None
        for attempt in range(max(1, self.retries)):
            try:
                raw = self._post(payload)
                return _extract_message(raw)
            except TransientApiError as e:
                last = e
                wait_s = _backoff_s(
                    attempt, getattr(e, "retry_after", None))
                if on_retry is not None and attempt + 1 < max(1, self.retries):
                    try:
                        on_retry(attempt + 1, wait_s, e)
                    except Exception:
                        pass
                time.sleep(wait_s)
            except ApiError:
                raise
        raise TransientApiError(str(last) if last else "unknown transient error")

    def check_connectivity(self) -> dict:
        """Lightweight `GET {base}/models` probe. Returns {'ok': bool, ...}."""
        req = urllib.request.Request(
            self.base_url + "/models",
            headers={"Authorization": "Bearer " + self.api_key},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=15,
                                        context=create_ssl_context()) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            names = []
            for m in body.get("data", []) if isinstance(body, dict) else []:
                if isinstance(m, dict) and m.get("id"):
                    names.append(m["id"])
            probe: dict = {"ok": True, "provider": self.provider,
                           "model": self.model, "models": names[:20]}
            if self.model not in names and names:
                probe["model_listed"] = False
            return probe
        except Exception as e:  # noqa: BLE001 - connectivity probe must not raise
            return {"ok": False, "provider": self.provider,
                    "model": self.model, "error": str(e)[:500]}


def _extract_message(raw: object) -> dict:
    """Validate provider envelope; raise ApiError on malformed responses."""
    if not isinstance(raw, dict):
        raise ApiError("malformed provider response: envelope is not an object")
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ApiError("malformed provider response: missing choices (%s)"
                       % str(raw)[:300])
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        raise ApiError("malformed provider response: missing message (%s)"
                       % str(raw)[:300])
    content = msg.get("content")
    if content is not None and not isinstance(content, (str, list)):
        raise ApiError("malformed provider response: bad content type")
    calls = msg.get("tool_calls")
    if calls is not None and not isinstance(calls, list):
        raise ApiError("malformed provider response: bad tool_calls type")
    return msg


def _looks_like_json_rejection(detail: str) -> bool:
    low = detail.lower()
    return any(h in low for h in _JSON_REJECT_HINTS)


def _retry_after_s(headers) -> float | None:
    """Parse a Retry-After response header (delta-seconds form)."""
    try:
        raw = headers.get("Retry-After") if headers else None
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return max(0.0, float(str(raw).strip()))
    except (TypeError, ValueError):
        return None


def _backoff_s(attempt: int, retry_after: float | None = None) -> float:
    """Gentle backoff: exponential (capped) + jitter; honors Retry-After."""
    if retry_after is not None:
        try:
            base = min(max(float(retry_after), 0.0), MAX_RETRY_AFTER_S)
        except (TypeError, ValueError):
            base = min(2 ** attempt, 15)
    else:
        base = min(2 ** attempt, 15)
    return base + random.uniform(0, 1.0)


def _infer_provider(base_url: str) -> str:
    low = (base_url or "").lower()
    if "integrate.api.nvidia.com" in low:
        return "nvidia"
    if "openrouter.ai" in low:
        return "openrouter"
    if "api.bluesminds.com" in low:
        return "bluesminds"
    return "custom"


class MockClient:
    """Scripted stand-in for tests / offline loop verification.

    `script` is a list of assistant-message dicts (or Exceptions to raise)
    returned in order; repeats the last entry once exhausted.
    """

    def __init__(self, script: list, model: str = "mock", provider: str = "mock"):
        self.script = list(script)
        self.model = model
        self.provider = provider
        self.calls: list = []

    def chat(self, messages: list, tools: list | None = None,
             force_json: bool = False, on_retry=None) -> dict:
        self.calls.append({"messages": messages, "tools": tools})
        if not self.script:
            return {"role": "assistant", "content": "{}"}
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item

    def check_connectivity(self) -> dict:
        return {"ok": True, "provider": self.provider,
                "model": self.model, "models": [], "mock": True}
