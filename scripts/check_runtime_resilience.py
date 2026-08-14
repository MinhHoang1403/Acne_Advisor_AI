#!/usr/bin/env python3
"""Offline runtime resilience evaluation.

This script does not call live chat, Gemini, Ollama, Qdrant, Neo4j, Redis, or
PostgreSQL. It validates timeout, retry and circuit-breaker behavior with fake
async operations.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.resilience.budget import DeadlineBudget
from src.resilience.circuit_breaker import CircuitBreaker, CircuitState, InMemoryCircuitStateStore
from src.resilience.contracts import RuntimeResilienceSettings
from src.resilience.exceptions import CircuitOpenError, RetryExhaustedError
from src.resilience.provider import call_provider_with_resilience
from src.resilience.retry import RetryPolicy


async def _retry_eval() -> dict[str, Any]:
    attempts = 0

    async def operation(_: float) -> str:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("fake timeout")

    settings = RuntimeResilienceSettings(
        llm_max_retries=1,
        llm_retry_base_delay_seconds=0,
        llm_retry_max_delay_seconds=0,
        circuit_breaker_failure_threshold=10,
    )
    breaker = CircuitBreaker(settings, store=InMemoryCircuitStateStore())
    try:
        await call_provider_with_resilience(
            provider_name="fake",
            operation=operation,
            budget=DeadlineBudget.from_timeout(2),
            timeout_seconds=1,
            retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0, max_delay_seconds=0),
            circuit_breaker=breaker,
            sleep=lambda _: asyncio.sleep(0),
        )
    except RetryExhaustedError:
        return {"passed": attempts == 2, "attempts": attempts}
    return {"passed": False, "attempts": attempts, "error": "RetryExhaustedError not raised"}


async def _circuit_eval() -> dict[str, Any]:
    settings = RuntimeResilienceSettings(
        circuit_breaker_failure_threshold=2,
        circuit_breaker_recovery_seconds=60,
    )
    breaker = CircuitBreaker(settings, store=InMemoryCircuitStateStore())
    breaker.record_failure("fake", transient=True)
    breaker.record_failure("fake", transient=True)
    opened = breaker.current_state("fake") == CircuitState.OPEN
    try:
        breaker.before_call("fake")
    except CircuitOpenError:
        return {"passed": opened, "state": breaker.current_state("fake").value}
    return {"passed": False, "state": breaker.current_state("fake").value, "error": "circuit did not open"}


async def main() -> int:
    checks = {
        "retry_exhaustion": await _retry_eval(),
        "circuit_open": await _circuit_eval(),
    }
    report = {
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
