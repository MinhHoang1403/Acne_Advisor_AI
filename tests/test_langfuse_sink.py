from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.observability import langfuse_sink
from src.observability.langfuse_sink import (
    LangfuseSettings,
    LangfuseSink,
    export_event_to_langfuse,
    langfuse_backend_status,
    mask_langfuse_otel_spans,
    submit_evaluation_score,
)
from src.observability.trace_exporter import (
    MAX_OBSERVABILITY_CANDIDATE_IDS,
    build_observability_event,
)

REQUEST_ID = "6a7e1f63-1c3a-4d96-b52c-1d8ca7d18f7c"
RAW_QUERY = "Tôi bị mụn và email patient@example.com"
RAW_HISTORY = "conversation-history-private"
RAW_PROMPT = "system-prompt-private"
RAW_ANSWER = "complete-medical-answer-private"
RAW_EXCEPTION = "database password=private-value"


class FakeObservation:
    def __init__(self, **kwargs: Any) -> None:
        self.created = kwargs
        self.children: list[FakeObservation] = []
        self.updates: list[dict[str, Any]] = []
        self.end_calls = 0

    def start_observation(self, **kwargs: Any) -> "FakeObservation":
        child = FakeObservation(**kwargs)
        self.children.append(child)
        return child

    def update(self, **kwargs: Any) -> "FakeObservation":
        self.updates.append(kwargs)
        return self

    def end(self, **_kwargs: Any) -> "FakeObservation":
        self.end_calls += 1
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "created": self.created,
            "updates": self.updates,
            "end_calls": self.end_calls,
            "children": [child.as_dict() for child in self.children],
        }


class FakeClient:
    def __init__(
        self,
        *,
        start_error: BaseException | None = None,
        score_error: BaseException | None = None,
        auth_result: bool = True,
        auth_error: BaseException | None = None,
    ) -> None:
        self.start_error = start_error
        self.score_error = score_error
        self.auth_result = auth_result
        self.auth_error = auth_error
        self.roots: list[FakeObservation] = []
        self.score_calls: list[dict[str, Any]] = []
        self.auth_calls = 0
        self.flush_calls = 0
        self.shutdown_calls = 0

    @staticmethod
    def create_trace_id(*, seed: str) -> str:
        return hashlib.sha256(seed.encode("utf-8")).digest()[:16].hex()

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        if self.start_error is not None:
            raise self.start_error
        root = FakeObservation(**kwargs)
        self.roots.append(root)
        return root

    def create_score(self, **kwargs: Any) -> None:
        if self.score_error is not None:
            raise self.score_error
        self.score_calls.append(kwargs)

    def auth_check(self) -> bool:
        self.auth_calls += 1
        if self.auth_error is not None:
            raise self.auth_error
        return self.auth_result

    def flush(self) -> None:
        self.flush_calls += 1

    def shutdown(self) -> None:
        self.shutdown_calls += 1


@pytest.fixture(autouse=True)
def reset_global_sink() -> None:
    langfuse_sink._reset_langfuse_sink_for_tests()
    yield
    langfuse_sink._reset_langfuse_sink_for_tests()


@pytest.fixture
def settings() -> LangfuseSettings:
    return LangfuseSettings(
        enabled=True,
        configured=True,
        base_url="http://langfuse.test",
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        environment="test",
    )


def _event(**overrides: Any):
    result: dict[str, Any] = {
        "agent_decision": {
            "action": "generate",
            "reason_code": "evidence_sufficient",
            "provider": "gemini",
            "model": "decision-model",
            "private_reasoning": "must-never-export",
        },
        "agent_decision_history": [
            {
                "action": "retrieve",
                "reason_code": "needs_evidence",
                "retrieval_query": RAW_QUERY,
                "missing_evidence": RAW_HISTORY,
            },
            {
                "action": "generate",
                "reason_code": "evidence_sufficient",
            },
        ],
        "retrieval_status": "ok",
        "retrieval_attempt": 1,
        "retrieval_trace": {
            "channels": {
                "dense": {"count": 2, "error": None},
                "bm25": {"count": 1, "error": None},
            },
            "candidate_trace": {
                "dense": [{"candidate_id": "dense-1"}, {"candidate_id": "dense-2"}],
                "bm25": [{"candidate_id": "bm25-1"}],
                "fused": [{"candidate_id": "dense-1"}, {"candidate_id": "bm25-1"}],
            },
            "fused_candidate_count": 2,
            "eligible_candidate_count": 2,
            "selected_ids": ["dense-1", "bm25-1"],
            "retry_evidence": {
                "retained_candidate_ids": ["dense-1"],
                "duplicate_candidate_ids": ["dense-1"],
            },
            "reranker": {"enabled": True, "model": "BAAI/bge-reranker-v2-m3"},
        },
        "generation_invoked": True,
        "generation_provider": "gemini",
        "generation_model": "gemini-3.5-flash-lite",
        "llm_fallback_used": False,
        "pipeline_manifest": {"phase": "production", "kb_version": "kb-build"},
        "pipeline_fingerprint": "pipeline-fingerprint",
        "performance_timings": {
            "agent_decision_1": 2.0,
            "retrieval_total": 7.0,
            "llm_generation": 11.0,
            "total_request": 25.0,
        },
    }
    result.update(overrides)
    return build_observability_event(
        query=RAW_QUERY,
        request_id=REQUEST_ID,
        session_id="patient@example.com",
        result=result,
        complete=True,
        safe_payload={
            "conversation_history": RAW_HISTORY,
            "prompt": RAW_PROMPT,
            "answer": RAW_ANSWER,
            "authorization": "Bearer private-token",
        },
    )


def _export(settings: LangfuseSettings, event: Any | None = None):
    client = FakeClient()
    sink = LangfuseSink(settings, client)
    assert sink.export(event or _event()) is True
    return client, client.roots[0]


def _serialized(root: FakeObservation) -> str:
    return json.dumps(root.as_dict(), ensure_ascii=False, sort_keys=True)


def _all_observations(root: FakeObservation) -> list[FakeObservation]:
    output = [root]
    for child in root.children:
        output.extend(_all_observations(child))
    return output


def test_settings_require_master_and_langfuse_toggles(monkeypatch):
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")

    assert LangfuseSettings.from_environment().enabled is False


def test_settings_require_all_credentials(monkeypatch):
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:3001")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert LangfuseSettings.from_environment().configured is False


def test_disabled_integration_does_not_construct_client(monkeypatch):
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "false")
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setattr(
        langfuse_sink,
        "_get_or_create_sink",
        lambda *_args, **_kwargs: pytest.fail("client factory must not run"),
    )

    assert export_event_to_langfuse(_event()) is False


def test_configured_sink_enqueues_safe_trace(settings):
    client, root = _export(settings)

    assert client.roots
    assert root.created["name"] == "chat-request"
    assert root.end_calls == 1


def test_request_export_never_flushes_or_probes_synchronously(settings):
    client, _ = _export(settings)

    assert client.flush_calls == 0
    assert client.shutdown_calls == 0
    assert client.auth_calls == 0


def test_trace_id_is_deterministically_correlated_with_request_id(settings):
    _, root = _export(settings)
    expected = hashlib.sha256(REQUEST_ID.encode("utf-8")).digest()[:16].hex()

    assert root.created["trace_context"]["trace_id"] == expected
    assert root.updates[-1]["metadata"]["request_id"] == REQUEST_ID


@pytest.mark.parametrize(
    "prohibited",
    [RAW_QUERY, RAW_HISTORY, RAW_PROMPT, RAW_ANSWER, "private-token"],
)
def test_prohibited_content_is_not_exported(settings, prohibited):
    _, root = _export(settings)

    assert prohibited not in _serialized(root)


def test_session_identifier_is_not_exported_without_opaque_validation(settings):
    _, root = _export(settings)

    assert "patient@example.com" not in _serialized(root)
    assert "session_id" not in _serialized(root)


def test_raw_exception_message_is_not_exported(settings):
    event = build_observability_event(
        query=RAW_QUERY,
        request_id=REQUEST_ID,
        error=RuntimeError(RAW_EXCEPTION),
        error_stage="api",
        complete=True,
    )
    _, root = _export(settings, event)

    assert RAW_EXCEPTION not in _serialized(root)
    assert "RuntimeError" in _serialized(root)


def test_client_start_error_fails_open(settings):
    sink = LangfuseSink(settings, FakeClient(start_error=RuntimeError("sdk failed")))

    assert sink.export(_event()) is False
    assert sink.last_error_category == "client_error"


def test_client_timeout_fails_open(settings):
    sink = LangfuseSink(settings, FakeClient(start_error=TimeoutError("slow")))

    assert sink.export(_event()) is False
    assert sink.last_error_category == "timeout"


def test_client_unavailable_fails_open(settings):
    sink = LangfuseSink(settings, FakeClient(start_error=ConnectionError("offline")))

    assert sink.export(_event()) is False
    assert sink.last_error_category == "unavailable"


def test_dense_failure_bm25_success_is_degraded_not_fatal(settings):
    event = _event(
        retrieval_status="degraded_dense",
        retrieval_trace={
            "channels": {
                "dense": {"count": 0, "error": "ProviderUnavailableError"},
                "bm25": {"count": 2, "error": None},
            },
            "selected_ids": ["bm25-1"],
            "reranker": {"enabled": False},
        },
    )
    _, root = _export(settings, event)
    text = _serialized(root)

    assert '"status": "degraded"' in text
    assert '"stage": "dense"' in text
    assert '"stage": "bm25"' in text


def test_both_retrieval_channels_failed_are_truthful(settings):
    event = _event(
        retrieval_status="failed",
        retrieval_trace={
            "channels": {
                "dense": {"count": 0, "error": "TimeoutError"},
                "bm25": {"count": 0, "error": "RuntimeError"},
            },
            "selected_ids": [],
            "reranker": {"enabled": False},
        },
    )
    _, root = _export(settings, event)
    stages = [
        obs.created.get("metadata", {})
        for obs in _all_observations(root)
        if obs.created.get("metadata", {}).get("stage") in {"dense", "bm25", "retrieve"}
    ]

    assert {stage["stage"]: stage["status"] for stage in stages} == {
        "dense": "failed",
        "bm25": "failed",
        "retrieve": "failed",
    }


def test_provider_fallback_exports_actual_generation_identity(settings):
    event = _event(
        generation_provider="ollama",
        generation_model="qwen3:8b",
        llm_fallback_used=True,
    )
    _, root = _export(settings, event)
    generation = next(
        obs for obs in _all_observations(root) if obs.created.get("as_type") == "generation"
    )

    assert generation.created["model"] == "qwen3:8b"
    assert generation.created["metadata"]["provider"] == "ollama"
    assert generation.created["metadata"]["fallback"] is True
    assert generation.created["metadata"]["status"] == "degraded"


def test_request_complete_timing_is_preserved(settings):
    _, root = _export(settings)

    assert root.updates[-1]["metadata"]["duration_ms"] == 25.0
    assert root.updates[-1]["metadata"]["complete"] is True


def test_candidate_ids_are_bounded_before_sink_mapping(settings):
    ids = [f"candidate-{index}" for index in range(50)]
    event = _event(
        retrieval_trace={
            "channels": {"dense": {"count": 50}, "bm25": {"count": 0}},
            "candidate_trace": {
                "dense": [{"candidate_id": candidate_id} for candidate_id in ids]
            },
            "selected_ids": ids,
            "reranker": {"enabled": False},
        }
    )

    assert max(len(stage.candidate_ids) for stage in event.summary.stages) == (
        MAX_OBSERVABILITY_CANDIDATE_IDS
    )


def test_agent_actions_and_reason_codes_are_mapped_without_reasoning(settings):
    _, root = _export(settings)
    agent_observations = [
        obs
        for obs in _all_observations(root)
        if obs.created.get("name") == "agent-decision"
    ]

    assert [obs.created["metadata"]["action"] for obs in agent_observations] == [
        "retrieve",
        "generate",
    ]
    assert "must-never-export" not in _serialized(root)


def test_retrieval_substages_are_nested_under_retrieval(settings):
    _, root = _export(settings)
    retrieval = next(obs for obs in root.children if obs.created.get("name") == "retrieve")

    assert {child.created["name"] for child in retrieval.children} >= {
        "dense",
        "bm25",
        "rrf",
        "rerank",
        "pack",
    }
    rrf = next(child for child in retrieval.children if child.created["name"] == "rrf")
    assert rrf.created["metadata"]["candidate_count"] == 2
    assert retrieval.created["metadata"]["retained_candidate_count"] == 1
    assert retrieval.created["metadata"]["duplicate_candidate_count"] == 1


def test_sink_does_not_mutate_business_event(settings):
    event = _event()
    before = event.model_dump(mode="json")

    _export(settings, event)

    assert event.model_dump(mode="json") == before


def test_supported_score_is_attached_to_correlated_trace(monkeypatch, settings):
    client = FakeClient()
    sink = LangfuseSink(settings, client)
    monkeypatch.setattr(langfuse_sink, "_get_or_create_sink", lambda *_args, **_kwargs: sink)
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_BASE_URL", settings.base_url)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

    assert submit_evaluation_score(REQUEST_ID, "Faithfulness", 0.73) is True
    assert client.score_calls == [
        {
            "name": "Faithfulness",
            "value": 0.73,
            "trace_id": hashlib.sha256(REQUEST_ID.encode()).digest()[:16].hex(),
            "data_type": "NUMERIC",
        }
    ]


def test_unsupported_score_name_is_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        submit_evaluation_score(REQUEST_ID, "Made Up Metric", 0.5)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_invalid_score_value_is_rejected(value):
    with pytest.raises(ValueError, match="score value"):
        submit_evaluation_score(REQUEST_ID, "Claim F1", value)


def test_optional_backend_outage_does_not_mark_application_dependency_required(
    monkeypatch, settings
):
    sink = LangfuseSink(settings, FakeClient(start_error=ConnectionError("offline")))
    sink.export(_event())
    monkeypatch.setattr(langfuse_sink, "_SINK", sink)
    monkeypatch.setattr(
        langfuse_sink,
        "_SINK_SIGNATURE",
        (settings.base_url, settings.public_key, settings.secret_key, settings.environment),
    )
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_BASE_URL", settings.base_url)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", settings.public_key)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", settings.secret_key)

    status = langfuse_backend_status()

    assert status["status"] == "degraded"
    assert status["optional"] is True
    assert status["required"] is False


def test_connectivity_probe_is_explicit_and_updates_reachability(settings):
    sink = LangfuseSink(settings, FakeClient(auth_result=True))

    assert sink.reachable is None
    assert sink.check_connectivity() is True
    assert sink.reachable is True


def test_mask_hook_deletes_content_attributes_and_keeps_safe_attributes():
    params = SimpleNamespace(
        spans={
            "span": SimpleNamespace(
                attributes={
                    "gen_ai.prompt.0.content": RAW_PROMPT,
                    "langfuse.observation.output": RAW_ANSWER,
                    "safe.status": "success",
                }
            )
        }
    )

    result = mask_langfuse_otel_spans(params=params)
    deleted = set(result.span_patches["span"].delete_attributes)

    assert "gen_ai.prompt.0.content" in deleted
    assert "langfuse.observation.output" in deleted
    assert "safe.status" not in deleted


def test_flush_and_shutdown_are_controlled_operator_paths(settings):
    client = FakeClient()
    sink = LangfuseSink(settings, client)

    assert sink.flush() is True
    sink.shutdown()

    assert client.flush_calls == 1
    assert client.shutdown_calls == 1


@pytest.mark.asyncio
async def test_langfuse_export_failure_does_not_change_chat_response(
    monkeypatch, tmp_path, settings
):
    from httpx import ASGITransport, AsyncClient

    from src.api import app as app_module

    class FailingObservation(FakeObservation):
        def start_observation(self, **_kwargs: Any) -> FakeObservation:
            raise ConnectionError("Langfuse unavailable")

    class FailingClient(FakeClient):
        def start_observation(self, **kwargs: Any) -> FakeObservation:
            root = FailingObservation(**kwargs)
            self.roots.append(root)
            return root

    sink = LangfuseSink(settings, FailingClient())
    monkeypatch.setenv("RELEASE_READINESS_TEST_MODE", "deterministic")
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("OBSERVABILITY_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_BASE_URL", settings.base_url)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", settings.public_key)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", settings.secret_key)
    monkeypatch.setattr(app_module, "begin_langfuse_request", sink.begin_request)

    async with AsyncClient(
        transport=ASGITransport(app=app_module.app), base_url="http://test"
    ) as client:
        response = await client.post("/chat", json={"message": "synthetic fixture"})

    assert response.status_code == 200
    assert response.json()["answer"]
    assert sink.last_error_category == "unavailable"
