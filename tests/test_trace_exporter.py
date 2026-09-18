from __future__ import annotations

import json

import pytest

from src.observability import trace_exporter
from src.observability.trace_exporter import (
    build_observability_event,
    emit_request_completion,
    export_observability_event,
)

REQUEST_ID = "6a7e1f63-1c3a-4d96-b52c-1d8ca7d18f7c"


def test_exporter_disabled_does_not_write(tmp_path):
    event = build_observability_event(query="Mụn đầu đen là gì?", request_id=REQUEST_ID)

    exported = export_observability_event(event, output_dir=tmp_path, enabled=False)

    assert exported is False
    assert list(tmp_path.iterdir()) == []


def test_exporter_writes_jsonl_when_enabled(tmp_path):
    event = build_observability_event(
        query="Benzoyl peroxide có phải kháng sinh không?",
        request_id=REQUEST_ID,
        result={
            "answer_quality_report": {"passed": True, "issues": []},
            "cache_hit": False,
        },
        safe_payload={"api_key": "secret", "context_text": "x" * 20},
        max_text_chars=8,
    )

    exported = export_observability_event(event, output_dir=tmp_path, enabled=True)

    files = list(tmp_path.glob("phase2_traces-*.jsonl"))
    assert exported is True
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
    assert payload["safe_payload"]["api_key"] == "[REDACTED]"
    assert "truncated" in payload["safe_payload"]["context_text"]
    assert payload["summary"]["pipeline_fingerprint"]


def test_observability_event_redacts_raw_query_and_inline_secrets():
    query = "Tôi bị mụn gần mắt và số điện thoại 0912345678"
    event = build_observability_event(
        query=query,
        request_id=REQUEST_ID,
        result={
            "retrieval_status": "recoverable_error",
            "fallback_reason": "backend failed token=secret-value",
            "retry_history": [{"attempt": 1, "query": query, "status": "failed"}],
        },
        max_text_chars=200,
    )

    payload = event.model_dump(mode="json")
    serialized = json.dumps(payload, ensure_ascii=False)

    assert event.summary.query == f"[REDACTED_QUERY chars={len(query)}]"
    assert event.query_hash
    assert query not in serialized
    assert "0912345678" not in serialized
    assert "secret-value" not in serialized
    assert event.request_id == REQUEST_ID
    assert "trace_id" not in payload


def test_builder_requires_request_identity_created_before_export():
    with pytest.raises(ValueError, match="request_id"):
        build_observability_event(query="Mụn đầu đen là gì?")


@pytest.mark.parametrize(
    ("retrieval_status", "dense_error", "bm25_error", "expected_retrieval"),
    [
        ("no_evidence", None, None, "success"),
        ("degraded_dense", "ProviderUnavailableError", None, "degraded"),
        ("failed", "TimeoutError", "UnexpectedResponse", "failed"),
    ],
)
def test_stage_statuses_preserve_normal_degraded_and_failed_semantics(
    retrieval_status,
    dense_error,
    bm25_error,
    expected_retrieval,
):
    event = build_observability_event(
        query="Mụn đầu đen là gì?",
        request_id=REQUEST_ID,
        result={
            "retrieval_status": retrieval_status,
            "retrieval_attempt": 1,
            "retrieval_trace": {
                "channels": {
                    "dense": {"count": 0 if dense_error else 2, "error": dense_error},
                    "bm25": {"count": 0 if bm25_error else 2, "error": bm25_error},
                },
                "selected_ids": [],
                "reranker": {"enabled": False},
            },
        },
    )

    stages = {stage.stage: stage for stage in event.summary.stages}
    assert stages["retrieve"].status == expected_retrieval
    assert stages["dense"].status == ("failed" if dense_error else "success")
    assert stages["bm25"].status == ("failed" if bm25_error else "success")
    if dense_error:
        assert stages["dense"].error_family == "dense_channel"
        assert stages["dense"].error_owner == "retrieval"
    assert stages["request"].status == (
        "success" if expected_retrieval == "success" else "degraded"
    )


def test_request_completion_contains_complete_timings_and_generation_identity():
    event = build_observability_event(
        query="Benzoyl peroxide là gì?",
        request_id=REQUEST_ID,
        complete=True,
        result={
            "agent_decision": {
                "action": "generate",
                "provider": "gemini",
                "model": "gemini-3.5-flash-lite",
            },
            "generation_invoked": True,
            "generation_provider": "gemini",
            "generation_model": "gemini-3.1-flash-lite",
            "llm_fallback_used": True,
            "performance_timings": {
                "agent_total": 120.0,
                "persistence": 4.0,
                "total_request": 130.0,
            },
        },
    )

    stages = {stage.stage: stage for stage in event.summary.stages}
    assert event.summary.complete is True
    assert event.summary.timings_ms["agent_total"] == 120.0
    assert stages["persistence"].duration_ms == 4.0
    assert stages["request"].duration_ms == 130.0
    assert stages["generate"].provider == "gemini"
    assert stages["generate"].model == "gemini-3.1-flash-lite"
    assert stages["generate"].fallback is True
    assert stages["generate"].status == "degraded"


def test_event_uses_executed_pipeline_identity_and_reports_cache_stage():
    event = build_observability_event(
        query="Benzoyl peroxide là gì?",
        request_id=REQUEST_ID,
        result={
            "pipeline_manifest": {"phase": "production", "kb_version": "executed-build"},
            "pipeline_fingerprint": "executed-fingerprint",
            "cache_checked": True,
            "cache_hit": False,
            "cache_reason": "miss",
        },
    )

    stages = {stage.stage: stage for stage in event.summary.stages}
    assert event.summary.pipeline_fingerprint == "executed-fingerprint"
    assert event.summary.knowledge_build_id == "executed-build"
    assert stages["cache"].status == "success"


def test_bypassed_cache_is_reported_as_skipped():
    event = build_observability_event(
        query="Benzoyl peroxide là gì?",
        request_id=REQUEST_ID,
        result={"cache_checked": False, "cache_hit": False, "cache_reason": "bypassed"},
    )

    stages = {stage.stage: stage for stage in event.summary.stages}
    assert stages["cache"].status == "skipped"


def test_emit_request_completion_fails_open_on_export_error(monkeypatch):
    def broken_export(*_args, **_kwargs):
        raise OSError("sink unavailable")

    monkeypatch.setattr(trace_exporter, "export_observability_event", broken_export)

    assert emit_request_completion(
        query="Mụn đầu đen là gì?",
        request_id=REQUEST_ID,
        result={"performance_timings": {"total_request": 1.0}},
        enabled=True,
    ) is False
