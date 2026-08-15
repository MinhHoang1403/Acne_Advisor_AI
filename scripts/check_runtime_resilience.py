#!/usr/bin/env python3
"""Offline deadline and bounded-retry evaluation with fake providers."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.resilience.budget import DeadlineBudget  # noqa: E402
from src.resilience.exceptions import ProviderTimeoutError, RetryExhaustedError  # noqa: E402
from src.resilience.provider import call_provider_with_resilience  # noqa: E402
from src.resilience.retry import RetryPolicy  # noqa: E402


async def _retry_eval() -> dict[str, Any]:
    attempts = 0

    async def operation(_: float) -> str:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("fake timeout")

    try:
        await call_provider_with_resilience(
            provider_name="fake",
            operation=operation,
            budget=DeadlineBudget.from_timeout(2),
            timeout_seconds=1,
            retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0, max_delay_seconds=0),
            sleep=lambda _: asyncio.sleep(0),
        )
    except RetryExhaustedError:
        return {"passed": attempts == 2, "attempts": attempts}
    return {"passed": False, "attempts": attempts}


async def _expired_deadline_eval() -> dict[str, Any]:
    called = False

    async def operation(_: float) -> str:
        nonlocal called
        called = True
        return "unexpected"

    expired = DeadlineBudget(started_at=0, deadline_at=0, clock=lambda: 1)
    try:
        await call_provider_with_resilience(
            provider_name="fake",
            operation=operation,
            budget=expired,
            timeout_seconds=1,
            retry_policy=RetryPolicy(max_retries=0),
        )
    except ProviderTimeoutError:
        return {"passed": not called, "provider_called": called}
    return {"passed": False, "provider_called": called}


async def main() -> int:
    checks = {
        "retry_exhaustion": await _retry_eval(),
        "deadline_before_call": await _expired_deadline_eval(),
    }
    report = {"passed": all(item["passed"] for item in checks.values()), "checks": checks}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
