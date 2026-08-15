"""Retry policy and transient/permanent provider error classification."""

from __future__ import annotations

import asyncio
import random
import ssl
from dataclasses import dataclass
from typing import Any

import httpx

from src.resilience.budget import DeadlineBudget
from src.resilience.exceptions import (
    PermanentProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RuntimeResilienceError,
)

RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}
PERMANENT_HTTP_STATUS = {400, 401, 403, 404, 422}


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 1
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 4.0
    jitter_ratio: float = 0.1

    @property
    def total_attempts(self) -> int:
        return 1 + max(0, self.max_retries)

    def delay_for_retry(self, retry_index: int) -> float:
        delay = self.base_delay_seconds * (2 ** max(0, retry_index - 1))
        capped = min(delay, self.max_delay_seconds)
        if capped <= 0 or self.jitter_ratio <= 0:
            return capped
        jitter = random.uniform(0, capped * self.jitter_ratio)
        return min(capped + jitter, self.max_delay_seconds)


def is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, asyncio.CancelledError):
        return False
    if isinstance(exc, (ProviderTimeoutError, ProviderUnavailableError)):
        return True
    if isinstance(exc, PermanentProviderError):
        return False
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError)):
        return True
    if isinstance(exc, (ssl.SSLError, OSError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_HTTP_STATUS
    if isinstance(exc, (ValueError, TypeError, RuntimeError)):
        return False
    text = str(exc).lower()
    permanent_markers = (
        "api key",
        "invalid key",
        "authentication",
        "permission",
        "forbidden",
        "not found",
        "invalid request",
        "validation",
        "model",
        "schema",
        "content policy",
    )
    transient_markers = (
        "timeout",
        "timed out",
        "temporarily",
        "unavailable",
        "connection reset",
        "connection refused",
        "429",
        "500",
        "502",
        "503",
        "504",
    )
    if any(marker in text for marker in permanent_markers):
        return False
    return any(marker in text for marker in transient_markers)


def classify_provider_exception(exc: BaseException, *, provider_name: str) -> RuntimeResilienceError:
    if isinstance(exc, RuntimeResilienceError):
        return exc
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
        return ProviderTimeoutError(f"{provider_name} provider timed out.")
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError)):
        return ProviderUnavailableError(f"{provider_name} provider is temporarily unavailable.")
    if isinstance(exc, (ssl.SSLError, OSError)):
        return ProviderUnavailableError(f"{provider_name} provider transport is temporarily unavailable.")
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in RETRYABLE_HTTP_STATUS:
            return ProviderUnavailableError(f"{provider_name} provider returned retryable HTTP {status}.")
        if status in PERMANENT_HTTP_STATUS:
            return PermanentProviderError(f"{provider_name} provider returned non-retryable HTTP {status}.")
    if is_retryable_exception(exc):
        return ProviderUnavailableError(f"{provider_name} provider failed with a transient error.")
    return PermanentProviderError(f"{provider_name} provider failed with a non-retryable error.")


async def sleep_with_budget(
    delay_seconds: float,
    budget: DeadlineBudget,
    sleep: Any = asyncio.sleep,
) -> bool:
    if delay_seconds <= 0:
        return True
    effective = min(delay_seconds, budget.remaining_seconds())
    if effective <= 0:
        return False
    await sleep(effective)
    return not budget.expired()


__all__ = [
    "RetryPolicy",
    "classify_provider_exception",
    "is_retryable_exception",
    "sleep_with_budget",
]
