"""
Production nat_caller implementations for Node 2 and Node 3.

Uses NVIDIA's hosted NIM API (OpenAI-compatible chat completions).
Each factory returns a callable(prompt: str) -> str that drops into
the existing nat_caller parameter with zero changes to node files.
"""

from __future__ import annotations

import logging
import time

import requests

from nat.nat_config import (
    NAT_MODEL_CODE,
    NAT_MODEL_REASON,
    NAT_TIMEOUT,
    NVIDIA_API_BASE,
    NVIDIA_API_KEY,
)

logger = logging.getLogger(__name__)


class NATTimeoutError(TimeoutError):
    """Raised when an NVIDIA API call exceeds the configured timeout."""


class NATAuthError(RuntimeError):
    """Raised on 401 — bad or expired NVIDIA_API_KEY."""


class NATError(RuntimeError):
    """General NAT inference failure."""


# ---------------------------------------------------------------------------
# Core call function
# ---------------------------------------------------------------------------

_MAX_RETRIES_429 = 3
_BACKOFF_BASE = 2.0  # seconds


def _call_nvidia_chat(
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """
    Send a chat completion request to the NVIDIA hosted API.

    Returns the assistant message content string.
    """
    url = f"{NVIDIA_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES_429 + 1):
        t0 = time.monotonic()
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=NAT_TIMEOUT,
            )
        except requests.exceptions.Timeout as exc:
            latency = time.monotonic() - t0
            logger.error(
                "NAT timeout | model=%s latency=%.1fs timeout=%ds",
                model, latency, NAT_TIMEOUT,
            )
            raise NATTimeoutError(
                f"NVIDIA API call to {model} timed out after {NAT_TIMEOUT}s"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise NATError(f"Network error calling NVIDIA API: {exc}") from exc

        latency = time.monotonic() - t0

        # --- Auth error: fail immediately ---
        if resp.status_code == 401:
            logger.error("NAT auth failure (401) | model=%s", model)
            raise NATAuthError(
                "NVIDIA API returned 401 Unauthorized. "
                "Check NVIDIA_API_KEY in .env."
            )

        # --- Rate limit: retry with backoff ---
        if resp.status_code == 429:
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                "NAT rate-limited (429) | model=%s attempt=%d/%d backoff=%.1fs",
                model, attempt, _MAX_RETRIES_429, wait,
            )
            last_exc = NATError(f"Rate limited (429) on attempt {attempt}")
            time.sleep(wait)
            continue

        # --- Other HTTP errors ---
        if not resp.ok:
            body = resp.text[:500]
            raise NATError(
                f"NVIDIA API returned {resp.status_code}: {body}"
            )

        # --- Success ---
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        logger.info(
            "NAT call OK | model=%s latency=%.1fs prompt_tok=%s comp_tok=%s",
            model,
            latency,
            usage.get("prompt_tokens", "?"),
            usage.get("completion_tokens", "?"),
        )
        return content

    # Exhausted retries on 429
    raise last_exc or NATError("NAT call failed after retries")


# ---------------------------------------------------------------------------
# Public factories — return callables matching nat_caller(prompt) -> str
# ---------------------------------------------------------------------------

def make_nat_caller_code(
    max_tokens: int = 8192,
    temperature: float = 0.2,
):
    """
    Returns a nat_caller for Node 2 (PyTorch code generation).
    Routes to Qwen2.5-Coder 32B Instruct.
    """
    def caller(prompt: str) -> str:
        return _call_nvidia_chat(
            prompt=prompt,
            model=NAT_MODEL_CODE,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return caller


def make_nat_caller_reason(
    max_tokens: int = 4096,
    temperature: float = 0.3,
):
    """
    Returns a nat_caller for Node 3 (bottleneck analysis + CUDA stubs).
    Routes to Nemotron Super 49B.
    """
    def caller(prompt: str) -> str:
        return _call_nvidia_chat(
            prompt=prompt,
            model=NAT_MODEL_REASON,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return caller
