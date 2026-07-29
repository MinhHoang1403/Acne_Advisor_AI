"""
tests/test_api_health.py – API Health Endpoint Tests
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import app, parse_cors_origins


@pytest.mark.asyncio
async def test_health_returns_ok(monkeypatch):
    async def fake_preflight():
        return {
            "status": "ok",
            "checks": {
                "postgres": {"status": "ok"},
                "qdrant": {"status": "ok"},
                "neo4j": {"status": "ok"},
                "redis": {"status": "ok"},
                "ollama": {"status": "ok"},
            },
        }

    monkeypatch.setattr("src.api.preflight.run_phase2_preflight", fake_preflight)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "acne-advisor-api"


@pytest.mark.asyncio
async def test_health_returns_degraded_with_reachable_backend(monkeypatch):
    async def fake_preflight():
        return {
            "status": "degraded",
            "checks": {
                "postgres": {"status": "ok"},
                "qdrant": {"status": "unavailable"},
                "neo4j": {"status": "ok"},
                "redis": {"status": "ok"},
                "ollama": {"status": "ok"},
            },
        }

    monkeypatch.setattr("src.api.preflight.run_phase2_preflight", fake_preflight)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["qdrant"] == "unavailable"
    assert data["checks"]["qdrant"]["status"] == "unavailable"


@pytest.mark.asyncio
async def test_health_reports_optional_ollama_without_degrading_core_runtime(monkeypatch):
    async def fake_preflight():
        return {
            "status": "ok",
            "checks": {
                "postgres": {"status": "ok"},
                "qdrant": {"status": "ok"},
                "neo4j": {"status": "ok"},
                "redis": {"status": "ok"},
                "ollama": {
                    "status": "unavailable",
                    "optional": True,
                    "required": False,
                    "requirement_reason": "gemini runtime uses Ollama only as an opportunistic fallback",
                },
                "generation": {"status": "ok", "provider": "gemini"},
            },
        }

    monkeypatch.setattr("src.api.preflight.run_phase2_preflight", fake_preflight)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ollama"] == "unavailable"
    assert data["checks"]["ollama"]["optional"] is True


def test_parse_cors_origins_defaults_and_dedupes():
    origins = parse_cors_origins(" http://localhost:5173/, http://127.0.0.1:5173, http://localhost:5173, * ")

    assert origins == ["http://localhost:5173", "http://127.0.0.1:5173"]


@pytest.mark.asyncio
async def test_cors_allows_localhost_and_127(monkeypatch):
    async def fake_preflight():
        return {
            "status": "ok",
            "checks": {
                "postgres": {"status": "ok"},
                "qdrant": {"status": "ok"},
                "neo4j": {"status": "ok"},
                "redis": {"status": "ok"},
                "ollama": {"status": "ok"},
            },
        }

    monkeypatch.setattr("src.api.preflight.run_phase2_preflight", fake_preflight)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        localhost = await client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        loopback = await client.options(
            "/health",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        unknown = await client.options(
            "/health",
            headers={
                "Origin": "http://malicious.local:5173",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert localhost.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert loopback.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert "access-control-allow-origin" not in unknown.headers


@pytest.mark.asyncio
async def test_models_exposes_current_qwen3_8b_default(monkeypatch):
    async def fake_list_ollama_models():
        return ["qwen3:8b"]

    monkeypatch.setenv("GOOGLE_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("GOOGLE_FALLBACK_MODELS", "gemini-3.1-flash-lite")
    monkeypatch.setattr(
        "src.agent.llm.ollama_client.list_ollama_models",
        fake_list_ollama_models,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/models")

    assert response.status_code == 200
    models = response.json()["models"]
    by_model = {item["model"]: item for item in models}
    assert response.json()["default_model"] == "gemini-3.5-flash"
    assert by_model["gemini-3.5-flash"]["display_name"] == "Gemini 3.5 Flash"
    assert by_model["gemini-3.5-flash"]["is_default"] is True
    assert by_model["gemini-3.1-flash-lite"]["display_name"] == "Gemini 3.1 Flash-Lite"
    assert by_model["gemini-3.1-flash-lite"]["is_default"] is False
    assert by_model["qwen3:8b"]["available"] is True
    assert "qwen2.5:latest" not in by_model
    assert len(by_model) == len(models)


@pytest.mark.asyncio
async def test_chat_metadata_exposes_requested_and_actual_model(monkeypatch):
    async def fake_run_clinical_agent(**kwargs):
        assert kwargs["llm_provider"] == "gemini"
        assert kwargs["llm_model"] == "gemini-3.5-flash"
        return {
            "answer": "Benzoyl peroxide không phải là kháng sinh.",
            "session_id": kwargs["session_id"],
            "sources": ["entity:active_ingredient", "fixture.pdf"],
            "vector_contexts": [
                {"source_file": "entity:active_ingredient", "retrieval_source": "entity"},
                {"source_file": "fixture.pdf", "document_title": "Fixture Acne Guide"},
            ],
            "symptoms": [],
            "safety_flags": [],
            "graph_facts": [],
            "retrieval_status": "hybrid",
            "fallback_applied": False,
            "fallback_type": "none",
            "fallback_cache_eligible": True,
            "is_in_domain": True,
            "guardrail": "in_domain",
            "cache_checked": True,
            "cache_hit": False,
            "cache_reason": "miss",
            "cache_metadata": {},
            "requested_provider": "gemini",
            "requested_model": "gemini-3.5-flash",
            "actual_provider": "gemini",
            "actual_model": "gemini-3.1-flash-lite",
            "llm_fallback_used": True,
            "fallback_provider": "gemini",
            "fallback_model": "gemini-3.1-flash-lite",
            "fallback_reason": "quota_exhausted",
            "fallback_chain": [
                {
                    "provider": "gemini",
                    "model": "gemini-3.5-flash",
                    "role": "primary",
                    "status": "failed",
                    "reason": "quota_exhausted",
                },
                {
                    "provider": "gemini",
                    "model": "gemini-3.1-flash-lite",
                    "role": "fallback",
                    "status": "success",
                },
            ],
            "pipeline_manifest": {"phase": "phase2e", "answer_cache_version": "v5"},
            "pipeline_fingerprint": "fixture-fingerprint",
            "answer_quality_report": {"passed": True, "issues": []},
        }

    async def fake_persist_chat_to_db(**kwargs):
        return None

    monkeypatch.setattr("src.api.app.run_clinical_agent", fake_run_clinical_agent)
    monkeypatch.setenv("RELEASE_READINESS_TEST_MODE", "")
    monkeypatch.setattr("src.api.app._persist_chat_to_db", fake_persist_chat_to_db)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/chat",
            json={
                "message": "Benzoyl peroxide có phải kháng sinh không?",
                "llm_provider": "gemini",
                "llm_model": "gemini-3.5-flash",
                "allow_model_fallback": True,
            },
        )

    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert metadata["provider"] == "gemini"
    assert metadata["model"] == "gemini-3.1-flash-lite"
    assert metadata["requested_provider"] == "gemini"
    assert metadata["requested_model"] == "gemini-3.5-flash"
    assert metadata["fallback_used"] is True
    assert metadata["fallback_reason"] == "quota_exhausted"
    assert metadata["fallback_chain"][1]["model"] == "gemini-3.1-flash-lite"
    assert metadata["response_origin"] == "llm"
    assert metadata["guardrail_applied"] is False
    body = response.json()
    assert body["sources"] == ["Cơ sở tri thức hoạt chất", "Fixture Acne Guide"]
    assert body["source_metadata"][0]["source_id"] == "entity:active_ingredient"
