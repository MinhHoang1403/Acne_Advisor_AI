from __future__ import annotations

import asyncio

import pytest

from src.api import preflight


def test_dependency_error_detail_does_not_expose_raw_credentials(caplog):
    detail = preflight._safe_dependency_error(
        "PostgreSQL",
        RuntimeError("postgresql://user:secret-password@localhost/database"),
    )

    assert "secret-password" not in detail
    assert "secret-password" not in caplog.text
    assert "RuntimeError" in detail


@pytest.mark.asyncio
async def test_bounded_preflight_check_reports_timeout(monkeypatch):
    monkeypatch.setattr(preflight, "PREFLIGHT_CHECK_TIMEOUT_SECONDS", 0.001)

    async def slow_check():
        await asyncio.sleep(1)
        return preflight.CheckResult("ok")

    result = await preflight._bounded_check("slow", slow_check())

    assert result.status == "timeout"
    assert "slow health check exceeded" in result.detail


def test_runtime_provider_requirements_keep_ollama_optional_for_default_gemini(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_FALLBACK_ENABLED", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)

    requirements = preflight.get_runtime_provider_requirements()

    assert requirements["provider"] == "gemini"
    assert requirements["ollama_required"] is False
    assert "opportunistic" in requirements["ollama_requirement_reason"]


def test_runtime_provider_requirements_require_explicit_ollama_runtime(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    requirements = preflight.get_runtime_provider_requirements()

    assert requirements["provider"] == "ollama"
    assert requirements["ollama_required"] is True
    assert "primary provider" in requirements["ollama_requirement_reason"]


def test_runtime_provider_requirements_require_explicit_ollama_fallback(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_PROVIDER_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "ollama")

    requirements = preflight.get_runtime_provider_requirements()

    assert requirements["ollama_required"] is True
    assert requirements["fallback_provider"] == "ollama"
    assert "fallback explicitly requires" in requirements["ollama_requirement_reason"]


@pytest.mark.asyncio
async def test_preflight_keeps_optional_ollama_unavailability_out_of_core_status(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_PROVIDER_FALLBACK_ENABLED", "false")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    async def ok_check():
        return preflight.CheckResult("ok")

    async def unavailable_ollama():
        return preflight.CheckResult("unavailable", "connection refused")

    monkeypatch.setattr(preflight, "check_postgres", ok_check)
    monkeypatch.setattr(preflight, "check_qdrant", ok_check)
    monkeypatch.setattr(preflight, "check_neo4j", ok_check)
    monkeypatch.setattr(preflight, "check_redis", ok_check)
    monkeypatch.setattr(preflight, "check_ollama", unavailable_ollama)

    result = await preflight.run_runtime_preflight()

    assert result["status"] == "ok"
    assert result["checks"]["redis"]["status"] == "ok"
    assert result["checks"]["ollama"]["status"] == "unavailable"
    assert result["checks"]["ollama"]["optional"] is True
    assert result["checks"]["generation"]["status"] == "ok"


@pytest.mark.asyncio
async def test_preflight_preloads_dependencies_before_starting_bounded_checks(monkeypatch):
    events: list[str] = []

    def preload():
        events.append("preload")

    async def ok_check():
        assert events == ["preload"]
        return preflight.CheckResult("ok")

    monkeypatch.setattr(preflight, "_preload_check_dependencies", preload)
    monkeypatch.setattr(preflight, "check_postgres", ok_check)
    monkeypatch.setattr(preflight, "check_qdrant", ok_check)
    monkeypatch.setattr(preflight, "check_neo4j", ok_check)
    monkeypatch.setattr(preflight, "check_redis", ok_check)
    monkeypatch.setattr(preflight, "check_ollama", ok_check)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    result = await preflight.run_runtime_preflight()

    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_preflight_degrades_when_ollama_is_configured_as_primary(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    async def ok_check():
        return preflight.CheckResult("ok")

    async def unavailable_ollama():
        return preflight.CheckResult("unavailable", "connection refused")

    monkeypatch.setattr(preflight, "check_postgres", ok_check)
    monkeypatch.setattr(preflight, "check_qdrant", ok_check)
    monkeypatch.setattr(preflight, "check_neo4j", ok_check)
    monkeypatch.setattr(preflight, "check_redis", ok_check)
    monkeypatch.setattr(preflight, "check_ollama", unavailable_ollama)

    result = await preflight.run_runtime_preflight()

    assert result["status"] == "degraded"
    assert result["checks"]["ollama"]["required"] is True
    assert result["checks"]["generation"]["status"] == "unavailable"
    assert "Configured Ollama runtime is unavailable" in result["checks"]["generation"]["detail"]


@pytest.mark.asyncio
async def test_preflight_treats_redis_as_optional_runtime_dependency(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    async def ok_check():
        return preflight.CheckResult("ok")

    async def unavailable_redis():
        return preflight.CheckResult("unavailable", "redis connection refused")

    monkeypatch.setattr(preflight, "check_postgres", ok_check)
    monkeypatch.setattr(preflight, "check_qdrant", ok_check)
    monkeypatch.setattr(preflight, "check_neo4j", ok_check)
    monkeypatch.setattr(preflight, "check_redis", unavailable_redis)
    monkeypatch.setattr(preflight, "check_ollama", ok_check)

    result = await preflight.run_runtime_preflight()

    assert result["status"] == "ok"
    assert result["checks"]["redis"]["status"] == "unavailable"
    assert result["checks"]["redis"]["required"] is False
