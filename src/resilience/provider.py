"""Provider call wrapper with deadline, retry and circuit breaker support."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

from src.resilience.budget import DeadlineBudget
from src.resilience.circuit_breaker import CircuitBreaker
from src.resilience.exceptions import (
    CircuitOpenError,
    ProviderTimeoutError,
    RetryExhaustedError,
)
from src.resilience.retry import RetryPolicy, classify_provider_exception, is_retryable_exception, sleep_with_budget

T = TypeVar("T")


async def call_provider_with_resilience(
    *,
    provider_name: str,
    operation: Callable[[float], Awaitable[T]],
    budget: DeadlineBudget,
    timeout_seconds: float,
    retry_policy: RetryPolicy,
    circuit_breaker: CircuitBreaker,
    sleep=asyncio.sleep,
) -> tuple[T, dict[str, object]]:
    """Call a provider with budget-aware timeout, retry and circuit breaker."""

    attempts: list[dict[str, object]] = []
    last_error: BaseException | None = None
    max_attempts = retry_policy.total_attempts

    for attempt_number in range(1, max_attempts + 1):
        if budget.expired():
            raise ProviderTimeoutError(f"No remaining budget before calling provider {provider_name}.")

        permit = circuit_breaker.before_call(provider_name)
        state_before = permit.state_before
        effective_timeout = budget.cap_timeout(timeout_seconds)
        if effective_timeout <= 0:
            raise ProviderTimeoutError(f"No remaining timeout for provider {provider_name}.")

        attempt_meta: dict[str, object] = {
            "provider_name": provider_name,
            "attempt_number": attempt_number,
            "max_attempts": max_attempts,
            "stage_timeout_seconds": round(effective_timeout, 3),
            "circuit_state_before": state_before.value,
            "retry_scheduled": False,
        }

        try:
            async with asyncio.timeout(effective_timeout):
                result = await operation(effective_timeout)
            state_after = circuit_breaker.record_success(provider_name)
            attempt_meta["circuit_state_after"] = state_after.value
            attempts.append(attempt_meta)
            return result, {
                "provider_name": provider_name,
                "attempt_number": attempt_number,
                "max_attempts": max_attempts,
                "circuit_state_before": state_before.value,
                "circuit_state_after": state_after.value,
                "attempts": attempts,
            }
        except asyncio.CancelledError:
            raise
        except CircuitOpenError:
            raise
        except Exception as exc:
            classified = classify_provider_exception(exc, provider_name=provider_name)
            retryable = is_retryable_exception(classified)
            state_after = circuit_breaker.record_failure(provider_name, transient=retryable)
            attempt_meta.update(
                {
                    "circuit_state_after": state_after.value,
                    "failure_class": classified.__class__.__name__,
                    "error_code": getattr(classified, "error_code", "provider_error"),
                }
            )
            last_error = classified

            if not retryable or attempt_number >= max_attempts:
                attempts.append(attempt_meta)
                if retryable and attempt_number >= max_attempts:
                    raise RetryExhaustedError(f"Retry exhausted for provider {provider_name}.") from classified
                raise classified from exc

            delay = retry_policy.delay_for_retry(attempt_number)
            if delay > budget.remaining_seconds():
                attempts.append(attempt_meta)
                raise RetryExhaustedError(f"Retry budget exhausted for provider {provider_name}.") from classified

            attempt_meta["retry_scheduled"] = True
            attempt_meta["retry_delay_seconds"] = round(delay, 3)
            attempts.append(attempt_meta)
            if not await sleep_with_budget(delay, budget, sleep=sleep):
                raise RetryExhaustedError(f"Retry budget exhausted for provider {provider_name}.") from classified

    raise RetryExhaustedError(f"Retry exhausted for provider {provider_name}.") from last_error


def safe_resilience_failure_metadata(exc: BaseException) -> dict[str, object]:
    return {
        "failure_class": exc.__class__.__name__,
        "error_code": getattr(exc, "error_code", "runtime_resilience_error"),
        "retryable": bool(getattr(exc, "retryable", True)),
    }


__all__ = ["call_provider_with_resilience", "safe_resilience_failure_metadata"]
