from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from src.agent.llm import provider as llm_provider
from src.agent.nodes import cache as cache_node
from src.api import app as app_module
from src.database import retriever as retriever_module
from src.resilience.budget import DeadlineBudget
from src.resilience.circuit_breaker import CircuitBreaker, CircuitState, InMemoryCircuitStateStore
from src.resilience.contracts import RuntimeResilienceSettings, runtime_resilience_settings_from_env
from src.resilience.exceptions import (
    AgentTimeoutError,
    CircuitOpenError,
    PermanentProviderError,
    ProviderUnavailableError,
    RetryExhaustedError,
)
from src.resilience.provider import call_provider_with_resilience
from src.resilience.retry import RetryPolicy, is_retryable_exception


def test_runtime_resilience_settings_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_TOTAL_TIMEOUT_SECONDS", "9")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("CIRCUIT_BREAKER_ENABLED", "false")

    settings = runtime_resilience_settings_from_env()

    assert settings.agent_total_timeout_seconds == 9
    assert settings.llm_max_retries == 2
    assert settings.circuit_breaker_enabled is False


def test_runtime_resilience_defaults_reserve_time_for_one_compact_ollama_retry(monkeypatch):
    monkeypatch.delenv("AGENT_TOTAL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("OLLAMA_TIMEOUT_SECONDS", raising=False)

    settings = runtime_resilience_settings_from_env()

    assert settings.ollama_timeout_seconds == 160
    assert settings.agent_total_timeout_seconds == 210
    assert settings.agent_total_timeout_seconds > settings.ollama_timeout_seconds


def test_rerank_timeout_is_capped_below_retrieval_timeout(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("RERANK_TIMEOUT_SECONDS", "20")

    assert retriever_module._rerank_timeout_within_retrieval_budget() == 10


def test_deadline_budget_fake_clock_caps_and_expires():
    now = {"value": 100.0}
    budget = DeadlineBudget.from_timeout(5, clock=lambda: now["value"])

    assert budget.remaining_seconds() == 5
    assert budget.cap_timeout(10) == 5
    now["value"] = 103.0
    assert budget.elapsed_seconds() == 3
    assert budget.cap_timeout(10) == 2
    now["value"] = 106.0
    assert budget.expired() is True
    assert budget.cap_timeout(1) == 0


@pytest.mark.asyncio
async def test_retry_policy_retries_transient_once():
    attempts = 0

    async def operation(_: float) -> str:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("fake timeout")

    settings = RuntimeResilienceSettings(circuit_breaker_failure_threshold=10)
    breaker = CircuitBreaker(settings, store=InMemoryCircuitStateStore())

    with pytest.raises(RetryExhaustedError):
        await call_provider_with_resilience(
            provider_name="fake",
            operation=operation,
            budget=DeadlineBudget.from_timeout(2),
            timeout_seconds=1,
            retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0, max_delay_seconds=0),
            circuit_breaker=breaker,
            sleep=lambda _: asyncio.sleep(0),
        )

    assert attempts == 2
    assert is_retryable_exception(TimeoutError("timeout"))


@pytest.mark.asyncio
async def test_provider_successful_retry_and_deadline_exhausted_before_call():
    attempts = 0

    async def succeeds_on_retry(_: float) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("fake timeout")
        return "ok"

    settings = RuntimeResilienceSettings(circuit_breaker_failure_threshold=10)
    breaker = CircuitBreaker(settings, store=InMemoryCircuitStateStore())
    result, meta = await call_provider_with_resilience(
        provider_name="fake",
        operation=succeeds_on_retry,
        budget=DeadlineBudget.from_timeout(2),
        timeout_seconds=1,
        retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0, max_delay_seconds=0),
        circuit_breaker=breaker,
        sleep=lambda _: asyncio.sleep(0),
    )
    assert result == "ok"
    assert meta["attempt_number"] == 2

    called = False

    async def should_not_call(_: float) -> str:
        nonlocal called
        called = True
        return "bad"

    expired = DeadlineBudget(started_at=0, deadline_at=0, clock=lambda: 1)
    with pytest.raises(Exception):
        await call_provider_with_resilience(
            provider_name="fake",
            operation=should_not_call,
            budget=expired,
            timeout_seconds=1,
            retry_policy=RetryPolicy(max_retries=0),
            circuit_breaker=breaker,
        )
    assert called is False


def test_circuit_breaker_opens_after_threshold():
    settings = RuntimeResilienceSettings(
        circuit_breaker_failure_threshold=2,
        circuit_breaker_recovery_seconds=60,
    )
    breaker = CircuitBreaker(settings, store=InMemoryCircuitStateStore())

    breaker.record_failure("gemini", transient=True)
    breaker.record_failure("gemini", transient=True)

    assert breaker.current_state("gemini") == CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        breaker.before_call("gemini")


def test_circuit_half_open_recovery_and_provider_isolation():
    now = {"value": 0.0}
    settings = RuntimeResilienceSettings(
        circuit_breaker_failure_threshold=1,
        circuit_breaker_recovery_seconds=5,
    )
    breaker = CircuitBreaker(settings, store=InMemoryCircuitStateStore(), clock=lambda: now["value"])

    breaker.record_failure("gemini", transient=True)
    assert breaker.current_state("gemini") == CircuitState.OPEN
    assert breaker.current_state("ollama") == CircuitState.CLOSED

    now["value"] = 6.0
    permit = breaker.before_call("gemini")
    assert permit.state_before == CircuitState.HALF_OPEN
    assert breaker.record_success("gemini") == CircuitState.CLOSED

    breaker.record_failure("gemini", transient=True)
    now["value"] = 12.0
    breaker.before_call("gemini")
    assert breaker.record_failure("gemini", transient=True) == CircuitState.OPEN


def test_permanent_and_cancelled_errors_are_not_retryable_or_circuit_opening():
    settings = RuntimeResilienceSettings(circuit_breaker_failure_threshold=1)
    breaker = CircuitBreaker(settings, store=InMemoryCircuitStateStore())

    breaker.record_failure("gemini", transient=False)
    assert breaker.current_state("gemini") == CircuitState.CLOSED
    assert is_retryable_exception(PermanentProviderError("invalid api key")) is False
    assert is_retryable_exception(asyncio.CancelledError()) is False
    assert is_retryable_exception(ProviderUnavailableError("temporary")) is True


def test_ollama_model_default_matches_runtime_baseline(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    provider, model = llm_provider._resolve_model("ollama", None)

    assert provider == "ollama"
    assert model == "qwen3:8b"


@pytest.mark.asyncio
async def test_provider_fallback_error_message_is_sanitized(monkeypatch):
    async def fake_list_ollama_models(timeout_seconds=None):
        return ["qwen3:8b"]

    async def fake_call_provider_resilient(**kwargs):
        provider = kwargs["provider"]
        if provider == "gemini":
            raise RuntimeError("primary failed token=secret-value")
        raise RuntimeError("fallback failed password=hidden-value")

    monkeypatch.setattr(llm_provider, "list_ollama_models", fake_list_ollama_models)
    monkeypatch.setattr(llm_provider, "_call_provider_resilient", fake_call_provider_resilient)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    with pytest.raises(Exception) as exc_info:
        await llm_provider.generate_llm_response(
            prompt="mụn",
            provider="gemini",
            allow_fallback=True,
            resilience_settings=RuntimeResilienceSettings(llm_max_retries=0),
        )

    message = str(exc_info.value)
    assert "secret-value" not in message
    assert "hidden-value" not in message
    assert "token=[REDACTED]" in message
    assert "password=[REDACTED]" in message


@pytest.mark.asyncio
async def test_chat_endpoint_maps_agent_timeout_to_504(monkeypatch):
    async def fake_run_clinical_agent(**_: object) -> dict:
        raise AgentTimeoutError("fake timeout")

    monkeypatch.setattr(app_module, "run_clinical_agent", fake_run_clinical_agent)
    app_module.active_requests.clear()

    async with AsyncClient(
        transport=ASGITransport(app=app_module.app),
        base_url="http://test",
    ) as client:
        response = await client.post("/chat", json={"message": "Mụn viêm nên làm gì?"})

    assert response.status_code == 504
    detail = response.json()["detail"]
    assert detail["code"] == "agent_timeout"
    assert detail["retryable"] is True


@pytest.mark.asyncio
async def test_cache_store_skips_runtime_fallback_or_errors(monkeypatch):
    calls = []

    async def fake_set_answer_cache(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(cache_node, "set_answer_cache", fake_set_answer_cache)
    monkeypatch.setenv("CACHE_MIN_ANSWER_CHARS", "10")
    monkeypatch.setenv("CACHE_REQUIRED_ENTITY_CHECK", "false")

    base_state = {
        "cache_hit": False,
        "bypass_cache": False,
        "conversation_history": [],
        "cache_reason": "miss",
        "is_in_domain": True,
        "use_history_context": False,
        "user_question": "Benzoyl peroxide là gì?",
        "final_answer": "Benzoyl peroxide là hoạt chất bôi trị mụn.",
        "sources": ["source.pdf"],
        "actual_provider": "gemini",
        "actual_model": "gemini-2.5-flash",
        "answer_quality_report": {"passed": True, "issues": []},
        "errors": ["provider_timeout"],
    }

    await cache_node.cache_store_node(base_state)
    assert calls == []

    state_with_fallback = dict(base_state)
    state_with_fallback["errors"] = []
    state_with_fallback["llm_fallback_used"] = True
    await cache_node.cache_store_node(state_with_fallback)
    assert calls == []
