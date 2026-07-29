from __future__ import annotations

from scripts import pre_ui_runtime_check


def test_pre_ui_env_summary_redacts_url_credentials(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:secret@localhost:5432/acne")
    monkeypatch.setenv("REDIS_URL", "redis://:secret@localhost:6379/0")
    monkeypatch.setenv("SEMANTIC_RERANK_MODEL_PATH", "C:\\Models\\local-reranker")

    summary = pre_ui_runtime_check._env_summary()

    assert summary["DATABASE_URL"] == "<CONFIGURED>"
    assert summary["REDIS_URL"] == "<CONFIGURED>"
    assert summary["SEMANTIC_RERANK_MODEL_PATH"].startswith("<CONFIGURED:")
    assert "C:\\Models" not in summary["SEMANTIC_RERANK_MODEL_PATH"]
    assert "secret" not in repr(summary).lower()


def test_pre_ui_frontend_config_has_api_contract():
    result = pre_ui_runtime_check._check_frontend_config()

    assert result["passed"] is True
    assert result["details"]["uses_vite_api_url"] is True
    assert result["details"]["fallback_local_api"] is True
    assert result["details"]["chat_endpoint"] is True
    assert result["details"]["health_endpoint"] is True


def test_pre_ui_reranker_status_accepts_local_rules(monkeypatch):
    monkeypatch.delenv("SEMANTIC_RERANK_MODEL_PATH", raising=False)

    result = pre_ui_runtime_check._reranker_runtime_status(
        {
            "rerank_provider": "local_rules",
            "semantic_rerank_model_identifier": "",
            "semantic_rerank_allow_fallback": True,
        }
    )

    assert result["passed"] is True
    assert result["details"]["semantic_model_configured"] is False


def test_pre_ui_reranker_status_accepts_hybrid_with_existing_model_path(monkeypatch, tmp_path):
    monkeypatch.setenv("SEMANTIC_RERANK_MODEL_PATH", str(tmp_path))

    result = pre_ui_runtime_check._reranker_runtime_status(
        {
            "rerank_provider": "hybrid",
            "semantic_rerank_model_identifier": tmp_path.name,
            "semantic_rerank_allow_fallback": True,
        }
    )

    assert result["passed"] is True
    assert result["details"]["semantic_model_path_exists"] is True


def test_pre_ui_reranker_status_rejects_active_hybrid_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("SEMANTIC_RERANK_MODEL_PATH", str(tmp_path / "missing"))

    result = pre_ui_runtime_check._reranker_runtime_status(
        {
            "rerank_provider": "hybrid",
            "semantic_rerank_model_identifier": "missing",
            "semantic_rerank_allow_fallback": True,
        }
    )

    assert result["passed"] is False
    assert result["details"]["semantic_model_path_exists"] is False


def test_pre_ui_health_helpers_distinguish_core_health_from_optional_ollama():
    health = {
        "postgres": "ok",
        "qdrant": "ok",
        "neo4j": "ok",
        "redis": "ok",
        "ollama": "unavailable",
        "checks": {
            "ollama": {
                "status": "unavailable",
                "optional": True,
                "required": False,
                "requirement_reason": "gemini runtime uses Ollama only as an opportunistic fallback",
            }
        },
    }

    assert all(status == "ok" for status in pre_ui_runtime_check._core_health_statuses(health).values())
    ollama = pre_ui_runtime_check._ollama_health_details(health)
    assert ollama["optional"] is True
    assert ollama["required"] is False
    assert ollama["status"] == "unavailable"
