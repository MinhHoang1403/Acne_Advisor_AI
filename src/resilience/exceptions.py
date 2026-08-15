"""Typed runtime resilience exceptions."""

from __future__ import annotations


class RuntimeResilienceError(Exception):
    """Base class for safe runtime resilience failures."""

    error_code = "runtime_resilience_error"
    retryable = True


class AgentTimeoutError(RuntimeResilienceError):
    error_code = "agent_timeout"


class StageTimeoutError(RuntimeResilienceError):
    error_code = "stage_timeout"


class ProviderTimeoutError(RuntimeResilienceError):
    error_code = "provider_timeout"


class ProviderUnavailableError(RuntimeResilienceError):
    error_code = "provider_unavailable"


class RetryExhaustedError(RuntimeResilienceError):
    error_code = "retry_exhausted"


class PermanentProviderError(RuntimeResilienceError):
    error_code = "permanent_provider_error"
    retryable = False


__all__ = [
    "AgentTimeoutError",
    "PermanentProviderError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "RetryExhaustedError",
    "RuntimeResilienceError",
    "StageTimeoutError",
]
