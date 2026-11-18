from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agent.answer_formatting import (
    assess_structural_quality,
    repair_terminal_punctuation,
)
from src.agent.llm import ollama_client
from src.agent.nodes import quality as quality_node_mod
from src.agent.nodes.cache import _resolve_cache_model_key, cache_lookup_node
from src.agent.nodes.respond import finalize_response_node
from src.agent.requested_structure import parse_requested_structure
from src.agent.source_presentation import is_pure_source_request
from src.api import app as api_app
from src.ingestion.build import BUILD_MANIFEST_SCHEMA
from src.ingestion.manifest import load_build_manifest
from src.quality.answer_verifier import verify_answer_quality


def test_resolve_cache_model_key_normalizes_provider_aliases():
    provider, model = _resolve_cache_model_key(
        {"llm_provider": "local", "llm_model": "qwen3.5:9b"}
    )
    assert provider == "ollama"
    assert model == "qwen3.5:9b"

    g_provider, g_model = _resolve_cache_model_key(
        {"llm_provider": "gemini", "llm_model": "gemini-1.5-flash"}
    )
    assert g_provider == "gemini"
    assert g_model != "gemini-1.5-flash"


@pytest.mark.asyncio
async def test_cache_lookup_preserves_question_case_and_created_at(monkeypatch):
    from src.observability.versioning import (
        build_pipeline_version_manifest,
        compute_pipeline_fingerprint,
        get_answer_cache_version,
    )

    manifest = build_pipeline_version_manifest()
    fingerprint = compute_pipeline_fingerprint(manifest)
    fake_payload = {
        "answer": "Adapalene là retinoid bôi ngoài da.",
        "sources": ["aad_acne_2024"],
        "model_name": "gemini-3.1-flash-lite-preview",
        "created_at": "2026-11-18T00:00:00Z",
        "metadata": {
            "answer_version": get_answer_cache_version(),
            "pipeline_fingerprint": fingerprint,
            "selected_evidence_ids": ["aad_acne_2024"],
            "source_allowlist": [
                {"source_id": "aad_acne_2024", "display_name": "AAD Acne Guideline 2024"}
            ],
        },
    }

    async def fake_get_exact_cache(*_args, **_kwargs):
        return fake_payload

    monkeypatch.setattr("src.agent.nodes.cache.get_exact_cache", fake_get_exact_cache)

    raw_question = "Bỏ qua tretinoin. Adapalene có dùng được cho mụn ẩn không?"
    state = {
        "user_question": raw_question,
        "normalized_question": raw_question,
        "llm_provider": "gemini",
        "llm_model": "gemini-3.1-flash-lite-preview",
        "pipeline_manifest": manifest,
        "pipeline_fingerprint": fingerprint,
    }

    updated = await cache_lookup_node(state)
    assert updated["cache_hit"] is True
    assert "normalized_question" not in updated
    assert updated["cache_metadata"]["created_at"] == "2026-11-18T00:00:00Z"


def test_source_request_helper_separates_pure_source_queries_from_clinical():
    assert is_pure_source_request("Nguồn nào đã được dùng cho câu trả lời này?") is True
    assert is_pure_source_request("Theo kho dữ liệu, adapalene có gây khô da không?") is False


@pytest.mark.asyncio
async def test_finalize_response_preserves_safety_override_and_hybrid_clinical_draft():
    safety_state = {
        "user_question": "Nguồn nào cho câu trả lời này?",
        "normalized_question": "Nguồn nào cho câu trả lời này?",
        "draft_answer": "Cảnh báo an toàn y tế cần khám bác sĩ.",
        "safety_override": True,
        "fallback_applied": False,
        "generation_invoked": False,
        "vector_contexts": [
            {"chunk_id": "src_1", "document_title": "AAD 2024", "content": "Retinoid bôi"}
        ],
        "sources": ["src_1"],
        "confidence": 0.9,
    }
    safety_out = await finalize_response_node(safety_state)
    assert "Cảnh báo an toàn y tế" in safety_out["final_answer"]

    hybrid_state = {
        "user_question": "Theo nguồn đã truy hồi, adapalene dùng buổi tối đúng không?",
        "normalized_question": "Theo nguồn đã truy hồi, adapalene dùng buổi tối đúng không?",
        "draft_answer": "Adapalene thường được bôi một lớp mỏng vào buổi tối sau bước làm sạch.",
        "safety_override": False,
        "fallback_applied": False,
        "generation_invoked": True,
        "vector_contexts": [
            {
                "chunk_id": "src_1",
                "document_title": "AAD 2024",
                "authority": "AAD",
                "content": "Apply adapalene once daily in the evening.",
            }
        ],
        "sources": ["src_1"],
        "confidence": 0.9,
    }
    hybrid_out = await finalize_response_node(hybrid_state)
    assert "Adapalene thường được bôi một lớp mỏng" in hybrid_out["final_answer"]


def test_requested_structure_ignores_vietnamese_function_words_tu_and_sau():
    s1 = parse_requested_structure("Kể từ dấu hiệu đỏ rát ban đầu thì nên giảm tần suất không?")
    s2 = parse_requested_structure("Sau biểu hiện bong tróc có cần dưỡng ẩm phục hồi không?")
    assert s1.exact_item_count is None
    assert s2.exact_item_count is None


def test_terminal_punctuation_handles_closing_bold_and_quotes():
    sentence = "Đây là lựa chọn hiệu quả trong điều trị **mụn trứng cá nhẹ đến trung bình.**"
    repaired = repair_terminal_punctuation(sentence)
    assert repaired == sentence
    issues = assess_structural_quality("Adapalene giúp giảm nhân mụn **rất hiệu quả.**")
    assert not any("truncated" in str(issue).lower() or "cụt" in str(issue).lower() for issue in issues)


def test_answer_verifier_respects_explicit_topic_reset():
    report = verify_answer_quality(
        query="Bỏ qua tretinoin. Adapalene hoạt động như thế nào?",
        answer=(
            "Adapalene gắn chọn lọc lên thụ thể RAR-beta và RAR-gamma để bình thường hóa "
            "sừng hóa cổ nang lông và giảm phản ứng viêm."
        ),
        final_source_ids=["c1"],
    )
    assert not any("tretinoin" in str(issue).lower() for issue in report.issues)


@pytest.mark.asyncio
async def test_answer_quality_node_skips_double_finalize_when_already_validated(monkeypatch):
    calls: list[str] = []

    def fake_finalize(*_args, **_kwargs):
        calls.append("called")
        raise AssertionError("finalize_answer_presentation should not run a second time")

    monkeypatch.setattr(quality_node_mod, "finalize_answer_presentation", fake_finalize)

    state = {
        "user_question": "Adapalene bôi buổi nào?",
        "normalized_question": "Adapalene bôi buổi nào?",
        "final_answer": "Adapalene thường được bôi vào buổi tối.",
        "source_validation": {"status": "ok"},
        "sources": ["c1"],
        "vector_contexts": [{"chunk_id": "c1", "content": "Apply adapalene in the evening."}],
    }
    result = await quality_node_mod.answer_quality_node(state)
    assert calls == []
    assert "answer_quality_report" in result


@pytest.mark.asyncio
async def test_ollama_zero_budget_and_think_stripping(monkeypatch):
    assert await ollama_client.list_ollama_models(timeout_seconds=0) == []

    captured_payloads: list[dict] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "message": {"content": "<think>nháp nội bộ</think>\nAdapalene dùng buổi tối."}
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, _url, json=None, **_kwargs):
            captured_payloads.append(json or {})
            return FakeResponse()

    monkeypatch.setattr(ollama_client.httpx, "AsyncClient", FakeAsyncClient)

    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    text = await ollama_client.generate_ollama_response(
        model="qwen3.5:9b",
        system_prompt="Trả lời ngắn gọn.",
        prompt="Adapalene bôi lúc nào?",
        request_timeout=5.0,
        response_schema=schema,
    )
    assert text == "Adapalene dùng buổi tối."
    assert captured_payloads[0]["format"] == schema


def test_load_build_manifest_caches_by_stat_and_isolates_mutations(tmp_path):
    manifest_path = tmp_path / "knowledge_build_manifest.json"
    payload = {
        "schema": BUILD_MANIFEST_SCHEMA,
        "build_id": "0123456789abcdef0123",
        "counts": {"sources": 1},
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    first = load_build_manifest(manifest_path)
    first["counts"]["sources"] = 999
    second = load_build_manifest(manifest_path)
    assert second["counts"]["sources"] == 1

    payload["counts"]["sources"] = 42
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    third = load_build_manifest(manifest_path)
    assert third["counts"]["sources"] == 42


@pytest.mark.asyncio
async def test_delete_app_redis_cache_keys_fails_open_on_redis_error(monkeypatch):
    class BrokenRedis:
        async def scan(self, **_kwargs):
            raise ConnectionError("redis down")

    monkeypatch.setattr(
        "src.cache.redis_cache.get_redis",
        AsyncMock(return_value=BrokenRedis()),
    )

    deleted, patterns = await api_app._delete_app_redis_cache_keys()
    assert deleted == 0
    assert patterns == ["cache:answer:*"]


@pytest.mark.asyncio
async def test_sync_sessions_isolates_per_session_savepoints(monkeypatch):
    class FakeSavepoint:
        def __init__(self, session):
            self._session = session

        async def __aenter__(self):
            self._session.events.append("savepoint_enter")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            self._session.events.append(
                "savepoint_rollback" if exc_type is not None else "savepoint_release"
            )
            return False

    class FakeTx:
        def __init__(self, session):
            self._session = session

        async def __aenter__(self):
            self._session.events.append("tx_begin")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            self._session.events.append(
                "tx_rollback" if exc_type is not None else "tx_commit"
            )
            return False

    class FakeDBSession:
        def __init__(self):
            self.events: list[str] = []
            self.closed = False

        def begin(self):
            return FakeTx(self)

        def begin_nested(self):
            return FakeSavepoint(self)

        async def close(self):
            self.closed = True

    fake_db = FakeDBSession()
    monkeypatch.setattr(api_app, "_get_db_session", AsyncMock(return_value=fake_db))

    async def fake_session_exists(*, session_id: str, **_kwargs) -> bool:
        if session_id == "bad-session":
            raise RuntimeError("broken constraint")
        return True

    monkeypatch.setattr(
        "src.database.repositories.chat_history.session_exists",
        fake_session_exists,
    )
    monkeypatch.setattr(
        "src.database.repositories.chat_history.get_message_ids_for_session",
        AsyncMock(return_value=set()),
    )
    monkeypatch.setattr(
        "src.database.repositories.chat_history.save_message",
        AsyncMock(return_value=None),
    )

    body = SimpleNamespace(
        sessions=[
            SimpleNamespace(id="bad-session", title="Bad", hidden=False, messages=[]),
            SimpleNamespace(id="good-session", title="Good", hidden=False, messages=[]),
        ]
    )
    response = await api_app.sync_sessions(body)
    assert response.synced == 1
    assert response.errors == 1
    assert fake_db.closed is True
    assert "savepoint_rollback" in fake_db.events
    assert "tx_commit" in fake_db.events
