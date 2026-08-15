from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from src.agent.llm import provider as llm_provider
from src.agent.nodes import cache as cache_node
from src.api import app as app_module
from src.resilience.budget import DeadlineBudget
from src.resilience.contracts import RuntimeResilienceSettings, runtime_resilience_settings_from_env
from src.resilience.exceptions import (
    AgentTimeoutError,
    PermanentProviderError,
    ProviderUnavailableError,
    RetryExhaustedError,
)
from src.resilience.provider import call_provider_with_resilience
from src.resilience.retry import RetryPolicy, is_retryable_exception


def test_runtime_resilience_settings_have_only_deadline_retry_and_fallback(monkeypatch):
    monkeypatch.setenv("AGENT_TOTAL_TIMEOUT_SECONDS", "9")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    settings = runtime_resilience_settings_from_env()
    assert settings.agent_total_timeout_seconds == 9
    assert settings.llm_max_retries == 2
    assert not any("circuit" in name for name in settings.model_fields)


def test_runtime_defaults_leave_room_for_bounded_ollama_retry(monkeypatch):
    monkeypatch.delenv("AGENT_TOTAL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("OLLAMA_TIMEOUT_SECONDS", raising=False)
    settings = runtime_resilience_settings_from_env()
    assert settings.ollama_timeout_seconds == 160
    assert settings.agent_total_timeout_seconds == 210


def test_deadline_budget_caps_and_expires_with_fake_clock():
    now = {"value": 100.0}
    budget = DeadlineBudget.from_timeout(5, clock=lambda: now["value"])
    assert budget.cap_timeout(10) == 5
    now["value"] = 103.0
    assert budget.cap_timeout(10) == 2
    now["value"] = 106.0
    assert budget.expired() is True


@pytest.mark.asyncio
async def test_shared_provider_wrapper_retries_transient_failure_once():
    attempts = 0

    async def operation(_: float) -> str:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("fake timeout")

    with pytest.raises(RetryExhaustedError):
        await call_provider_with_resilience(
            provider_name="fake",
            operation=operation,
            budget=DeadlineBudget.from_timeout(2),
            timeout_seconds=1,
            retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0, max_delay_seconds=0),
            sleep=lambda _: asyncio.sleep(0),
        )
    assert attempts == 2


@pytest.mark.asyncio
async def test_shared_provider_wrapper_returns_retry_trace():
    attempts = 0

    async def operation(_: float) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("fake timeout")
        return "ok"

    result, metadata = await call_provider_with_resilience(
        provider_name="fake",
        operation=operation,
        budget=DeadlineBudget.from_timeout(2),
        timeout_seconds=1,
        retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0, max_delay_seconds=0),
        sleep=lambda _: asyncio.sleep(0),
    )
    assert result == "ok"
    assert metadata["attempt_number"] == 2
    assert len(metadata["attempts"]) == 2


def test_retry_classification_is_narrow():
    assert is_retryable_exception(PermanentProviderError("invalid api key")) is False
    assert is_retryable_exception(asyncio.CancelledError()) is False
    assert is_retryable_exception(ProviderUnavailableError("temporary")) is True


def test_ollama_model_default_matches_runtime_baseline(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert llm_provider._resolve_model("ollama", None) == ("ollama", "qwen3:8b")


@pytest.mark.asyncio
async def test_provider_fallback_error_message_is_sanitized(monkeypatch):
    async def fake_list_ollama_models(timeout_seconds=None):
        del timeout_seconds
        return ["qwen3:8b"]

    async def fake_call_provider_resilient(**kwargs):
        if kwargs["provider"] == "gemini":
            raise RuntimeError("primary failed token=secret-value")
        raise RuntimeError("fallback failed password=hidden-value")

    monkeypatch.setattr(llm_provider, "list_ollama_models", fake_list_ollama_models)
    monkeypatch.setattr(llm_provider, "_call_provider_resilient", fake_call_provider_resilient)
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
    assert "[REDACTED]" in message


@pytest.mark.asyncio
async def test_chat_endpoint_maps_agent_timeout_to_504(monkeypatch):
    async def fake_run_clinical_agent(**_: object) -> dict:
        raise AgentTimeoutError("fake timeout")

    monkeypatch.setattr(app_module, "run_clinical_agent", fake_run_clinical_agent)
    app_module.active_requests.clear()
    async with AsyncClient(transport=ASGITransport(app=app_module.app), base_url="http://test") as client:
        response = await client.post("/chat", json={"message": "Mụn viêm nên làm gì?"})
    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "agent_timeout"


@pytest.mark.asyncio
async def test_cache_store_skips_provider_fallback(monkeypatch):
    calls: list[object] = []

    async def fake_set_answer_cache(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(cache_node, "set_answer_cache", fake_set_answer_cache)
    await cache_node.cache_store_node(
        {
            "cache_hit": False,
            "bypass_cache": False,
            "cache_reason": "miss",
            "conversation_context": {"messages": [], "message_count": 0},
            "llm_fallback_used": True,
        }
    )
    assert calls == []
