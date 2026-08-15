from __future__ import annotations

import pytest

from src.agent.llm import provider as llm_provider
from src.resilience.contracts import RuntimeResilienceSettings
from src.resilience.exceptions import (
    PermanentProviderError,
    ProviderQuotaError,
    ProviderUnavailableError,
)


def _settings() -> RuntimeResilienceSettings:
    return RuntimeResilienceSettings(
        agent_total_timeout_seconds=10,
        gemini_timeout_seconds=2,
        ollama_timeout_seconds=2,
        llm_max_retries=0,
    )


async def _no_ollama_models(**_: object) -> list[str]:
    return []


async def _qwen3_ollama_model(**_: object) -> list[str]:
    return ["qwen3:8b"]


def test_parse_google_fallback_models_default_and_cleanup(monkeypatch):
    monkeypatch.delenv("GOOGLE_FALLBACK_MODELS", raising=False)
    assert llm_provider.parse_google_fallback_models(primary_model="gemini-3.5-flash") == [
        "gemini-3.1-flash-lite"
    ]
    assert llm_provider.parse_google_fallback_models(
        " gemini-3.1-flash-lite, gemini-3.5-flash, another-model, gemini-3.1-flash-lite ",
        primary_model="gemini-3.5-flash",
    ) == ["gemini-3.1-flash-lite", "another-model"]


@pytest.mark.asyncio
async def test_primary_gemini_success_does_not_call_fallback(monkeypatch):
    calls: list[tuple[str, str]] = []

    async def fake_call(**kwargs):
        calls.append((kwargs["provider"], kwargs["model"]))
        return "primary ok", {"provider_name": f'{kwargs["provider"]}:{kwargs["model"]}'}

    monkeypatch.setattr(llm_provider, "_call_provider_resilient", fake_call)
    result = await llm_provider.generate_llm_response(
        prompt="x",
        provider="gemini",
        model="gemini-3.5-flash",
        allow_fallback=True,
        resilience_settings=_settings(),
    )
    assert calls == [("gemini", "gemini-3.5-flash")]
    assert result["fallback_used"] is False


@pytest.mark.asyncio
async def test_gemini_429_falls_back_to_flash_lite(monkeypatch):
    calls: list[tuple[str, str]] = []

    async def fake_call(**kwargs):
        calls.append((kwargs["provider"], kwargs["model"]))
        if kwargs["model"] == "gemini-3.5-flash":
            raise ProviderQuotaError("Gemini daily project quota is exhausted (HTTP 429).")
        return "flash-lite ok", {"provider_name": f'{kwargs["provider"]}:{kwargs["model"]}'}

    monkeypatch.setenv("GOOGLE_FALLBACK_MODELS", "gemini-3.1-flash-lite")
    monkeypatch.setattr(llm_provider, "_call_provider_resilient", fake_call)
    monkeypatch.setattr(llm_provider, "list_ollama_models", _no_ollama_models)
    result = await llm_provider.generate_llm_response(
        prompt="x",
        provider="gemini",
        model="gemini-3.5-flash",
        allow_fallback=True,
        resilience_settings=_settings(),
    )
    assert calls == [("gemini", "gemini-3.5-flash"), ("gemini", "gemini-3.1-flash-lite")]
    assert result["model"] == "gemini-3.1-flash-lite"
    assert result["fallback_reason"] == "quota_exhausted"


@pytest.mark.asyncio
async def test_both_gemini_models_fail_then_ollama_success(monkeypatch):
    calls: list[tuple[str, str]] = []

    async def fake_call(**kwargs):
        calls.append((kwargs["provider"], kwargs["model"]))
        if kwargs["provider"] == "gemini":
            raise ProviderUnavailableError("provider returned retryable HTTP 503")
        return "ollama ok", {"provider_name": f'{kwargs["provider"]}:{kwargs["model"]}'}

    monkeypatch.setenv("GOOGLE_FALLBACK_MODELS", "gemini-3.1-flash-lite")
    monkeypatch.setattr(llm_provider, "_call_provider_resilient", fake_call)
    monkeypatch.setattr(llm_provider, "list_ollama_models", _qwen3_ollama_model)
    result = await llm_provider.generate_llm_response(
        prompt="x",
        provider="gemini",
        model="gemini-3.5-flash",
        allow_fallback=True,
        resilience_settings=_settings(),
    )
    assert calls == [
        ("gemini", "gemini-3.5-flash"),
        ("gemini", "gemini-3.1-flash-lite"),
        ("ollama", "qwen3:8b"),
    ]
    assert result["provider"] == "ollama"


@pytest.mark.asyncio
async def test_fallback_disabled_does_not_call_secondary(monkeypatch):
    calls: list[tuple[str, str]] = []

    async def fake_call(**kwargs):
        calls.append((kwargs["provider"], kwargs["model"]))
        raise ProviderUnavailableError("provider returned retryable HTTP 503")

    monkeypatch.setattr(llm_provider, "_call_provider_resilient", fake_call)
    with pytest.raises(ProviderUnavailableError):
        await llm_provider.generate_llm_response(
            prompt="x",
            provider="gemini",
            model="gemini-3.5-flash",
            allow_fallback=False,
            resilience_settings=_settings(),
        )
    assert calls == [("gemini", "gemini-3.5-flash")]


@pytest.mark.asyncio
async def test_permanent_provider_error_does_not_fallback(monkeypatch):
    calls: list[tuple[str, str]] = []

    async def fake_call(**kwargs):
        calls.append((kwargs["provider"], kwargs["model"]))
        raise PermanentProviderError("gemini returned non-retryable HTTP 401")

    monkeypatch.setattr(llm_provider, "_call_provider_resilient", fake_call)
    with pytest.raises(PermanentProviderError):
        await llm_provider.generate_llm_response(
            prompt="x",
            provider="gemini",
            model="gemini-3.5-flash",
            allow_fallback=True,
            resilience_settings=_settings(),
        )
    assert calls == [("gemini", "gemini-3.5-flash")]
