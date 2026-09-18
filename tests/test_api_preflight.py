from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import qdrant_client

from src.api import preflight
from src.ingestion.build import BUILD_MANIFEST_SCHEMA


def _write_active_manifest(path: Path, build_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": BUILD_MANIFEST_SCHEMA,
                "status": "activated",
                "build_id": build_id,
                "collections": {
                    "knowledge_logical": "acne_knowledge",
                    "knowledge_physical": f"acne_knowledge__{build_id}",
                    "entity_logical": "acne_entities",
                    "entity_physical": f"acne_entities__{build_id}",
                },
            }
        ),
        encoding="utf-8",
    )


def test_dependency_error_detail_does_not_expose_raw_credentials(caplog):
    detail = preflight._safe_dependency_error(
        "PostgreSQL",
        RuntimeError("postgresql://user:secret-password@localhost/database"),
    )

    assert "secret-password" not in detail
    assert "secret-password" not in caplog.text
    assert "RuntimeError" in detail


@pytest.mark.asyncio
async def test_ollama_health_is_truthful_about_not_probing_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight,
        "_http_get_json",
        lambda *_args, **_kwargs: {"models": [{"name": preflight.OLLAMA_MODEL}]},
    )

    result = await preflight.check_ollama()

    assert result.status == "ok"
    assert result.extra == {
        "model": preflight.OLLAMA_MODEL,
        "model_list_probed": True,
        "generation_probed": False,
    }
    generation = preflight.check_generation_provider(
        {
            "provider": "ollama",
            "fallback_enabled": False,
            "fallback_provider": None,
            "ollama_required": True,
            "ollama_requirement_reason": "configured primary provider is ollama",
        },
        result,
    )
    assert generation.status == "ok"
    assert generation.extra["generation_probed"] is False


@pytest.mark.asyncio
async def test_qdrant_preflight_reports_active_manifest_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    build_id = "a" * 20
    manifest_path = tmp_path / "manifest.json"
    _write_active_manifest(manifest_path, build_id)
    monkeypatch.setattr(preflight, "DEFAULT_ACTIVE_KNOWLEDGE_MANIFEST", manifest_path)

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def get_collection(self, *, collection_name):
            assert collection_name == "acne_knowledge"
            return SimpleNamespace(
                config=SimpleNamespace(
                    params={
                        "vectors": {"dense": {"size": 3072}},
                        "sparse_vectors": {"bm25": {}},
                    }
                ),
                points_count=512,
            )

        async def get_aliases(self):
            return SimpleNamespace(
                aliases=[
                    SimpleNamespace(
                        alias_name="acne_knowledge",
                        collection_name=f"acne_knowledge__{build_id}",
                    )
                ]
            )

        async def close(self):
            pass

    monkeypatch.setattr(qdrant_client, "AsyncQdrantClient", FakeClient)

    result = await preflight.check_qdrant()

    assert result.status == "ok"
    assert result.extra["knowledge_build_id"] == build_id
    assert result.extra["active_alias_target"] == f"acne_knowledge__{build_id}"


@pytest.mark.asyncio
async def test_qdrant_preflight_detects_alias_build_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    build_id = "a" * 20
    manifest_path = tmp_path / "manifest.json"
    _write_active_manifest(manifest_path, build_id)
    monkeypatch.setattr(preflight, "DEFAULT_ACTIVE_KNOWLEDGE_MANIFEST", manifest_path)

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def get_collection(self, *, collection_name):
            return SimpleNamespace(
                config=SimpleNamespace(
                    params={
                        "vectors": {"dense": {"size": 3072}},
                        "sparse_vectors": {"bm25": {}},
                    }
                ),
                points_count=512,
            )

        async def get_aliases(self):
            return SimpleNamespace(
                aliases=[
                    SimpleNamespace(
                        alias_name="acne_knowledge",
                        collection_name="acne_knowledge__" + "b" * 20,
                    )
                ]
            )

        async def close(self):
            pass

    monkeypatch.setattr(qdrant_client, "AsyncQdrantClient", FakeClient)

    result = await preflight.check_qdrant()

    assert result.status == "schema_mismatch"
    assert "alias acne_knowledge" in result.detail


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
