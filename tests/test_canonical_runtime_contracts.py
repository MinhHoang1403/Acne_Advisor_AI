from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from src.agent import answer_formatting
from src.agent.answer_formatting import (
    finalize_answer_presentation,
    infer_response_profile,
)
from src.agent.nodes import workflow
from src.agent.nodes.preparation import prepare_request_node
from src.agent.nodes.quality import answer_quality_node
from src.agent.prompts.medical_answer import (
    build_medical_prompt,
    build_medical_system_instruction,
)
from src.agent.source_presentation import (
    FILE_SOURCE_DISPLAY_NAMES,
    build_source_allowlist,
    build_source_metadata,
    canonical_source_id,
    validate_answer_source_mentions,
)
from src.quality.answer_verifier import verify_answer_quality
from src.retrieval.context_packer import pack_context, packed_context_to_response_contexts
from src.retrieval.contracts import NormalizedQuery, RetrievedCandidate
from src.retrieval.service import EvidenceRetriever


class _WorkflowTool:
    payloads: list[dict[str, Any]] = []

    @classmethod
    async def ainvoke(cls, payload: dict[str, Any]) -> dict[str, Any]:
        cls.payloads.append(payload)
        candidate_id = f"candidate-{len(cls.payloads)}"
        return {
            "vector_contexts": [
                {
                    "id": candidate_id,
                    "text": "Evidence",
                    "source_id": "stable-source",
                }
            ],
            "sources": ["stable-source"],
            "metadata": {
                "retrieval_status": "ok",
                "retrieval_trace": {
                    "query": payload["query"],
                    "rerank_query": payload["rerank_query"],
                    "selected_ids": [candidate_id],
                },
                "retained_retrieval_candidates": [
                    *payload["retained_retrieval_candidates"],
                    {"candidate_id": candidate_id},
                ],
                "packed_context": {
                    "items": [
                        {
                            "item_id": candidate_id,
                            "payload": {"source_id": "stable-source"},
                        }
                    ]
                },
            },
        }


@pytest.mark.asyncio
async def test_canonical_query_identity_matches_normal_and_retry_tool_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _WorkflowTool.payloads = []
    monkeypatch.setattr(workflow, "retrieve_evidence", _WorkflowTool())

    first = await workflow.retrieve_node(
        {
            "request_id": "request-1",
            "normalized_question": "Current normalized question",
            "agent_decision": {
                "action": "retrieve",
                "retrieval_query": "self-contained overall query",
                "reason_code": "needs_evidence",
            },
            "retrieval_attempt": 0,
            "retry_history": [],
        }
    )
    second = await workflow.retrieve_node(
        {
            "request_id": "request-1",
            "normalized_question": "Current normalized question",
            "agent_decision": {
                "action": "retry",
                "retrieval_query": "targeted missing evidence",
                "reason_code": "evidence_gap",
            },
            **first,
        }
    )

    assert first["query_identity"] == {
        "current_question": "Current normalized question",
        "acquisition_query": _WorkflowTool.payloads[0]["query"],
        "overall_rerank_query": _WorkflowTool.payloads[0]["rerank_query"],
    }
    assert second["query_identity"] == {
        "current_question": "Current normalized question",
        "acquisition_query": _WorkflowTool.payloads[1]["query"],
        "overall_rerank_query": _WorkflowTool.payloads[1]["rerank_query"],
    }
    assert second["query_identity"]["acquisition_query"] == "targeted missing evidence"
    assert second["query_identity"]["overall_rerank_query"] == ("self-contained overall query")

    attempt = second["retrieval_attempt_traces"][-1]
    assert attempt["request_id"] == "request-1"
    assert attempt["attempt_index"] == 2
    assert attempt["acquisition_query"] == attempt["retrieval_query"]
    assert attempt["overall_rerank_query"] == attempt["rerank_query"]
    assert attempt["legacy_field_mappings"] == {
        "retrieval_query": "acquisition_query",
        "rerank_query": "overall_rerank_query",
        "retry_history.query": "acquisition_query",
    }
    assert attempt["packed_evidence"] == [
        {
            "item_id": "candidate-2",
            "source_id": "stable-source",
            "section": None,
        }
    ]


@pytest.mark.asyncio
async def test_service_query_identity_matches_dense_bm25_and_reranker_calls() -> None:
    dense_queries: list[str] = []
    sparse_queries: list[str] = []

    class Store:
        async def search_sparse(self, query: str, top_k: int) -> list[dict[str, Any]]:
            sparse_queries.append(query)
            return []

        async def close(self) -> None:
            return None

    class Scorer:
        model_name = "recording-reranker"
        query: str | None = None

        async def score(
            self,
            query: str,
            candidates: Sequence[RetrievedCandidate],
        ) -> Sequence[float]:
            self.query = query
            return [1.0 for _ in candidates]

    class Retriever(EvidenceRetriever):
        async def _dense_search(self, query: str, limit: int) -> list[dict[str, Any]]:
            dense_queries.append(query)
            return [
                {
                    "id": "candidate-1",
                    "score": 1.0,
                    "text": "Evidence",
                    "chunk_id": "candidate-1",
                    "source_id": "stable-source",
                }
            ]

    scorer = Scorer()
    retriever = Retriever(Store(), reranker=scorer, reranker_enabled=True)
    result = await retriever.retrieve(
        "  acquisition   query  ",
        rerank_query="  overall   rerank   query  ",
    )

    assert dense_queries == ["acquisition query"]
    assert sparse_queries == ["acquisition query"]
    assert scorer.query == "overall rerank query"
    assert result.metadata["retrieval_trace"]["query_identity"] == {
        "acquisition_query": dense_queries[0],
        "overall_rerank_query": scorer.query,
    }


@pytest.mark.asyncio
async def test_request_shape_is_parsed_once_and_reused_by_runtime_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = answer_formatting.parse_requested_structure

    def counted(question: str):
        nonlocal calls
        calls += 1
        return original(question)

    monkeypatch.setattr(answer_formatting, "parse_requested_structure", counted)
    question = "So sánh adapalene và benzoyl peroxide bằng bảng 2 cột"
    prepared = await prepare_request_node({"user_question": question, "conversation_history": []})
    shape = prepared["request_shape"]

    assert calls == 1
    assert shape["wants_table"] is True
    assert shape["exact_column_count"] == 2
    assert shape["response_profile"] == "comparison"
    assert infer_response_profile(question, request_shape=shape) == "comparison"

    build_medical_system_instruction(question, request_shape=shape)
    answer = "| Thuốc | Điểm chính |\n|---|---|\n| Adapalene | A |\n| Benzoyl peroxide | B |"
    presented = finalize_answer_presentation(
        answer,
        user_question=question,
        request_shape=shape,
    )
    verify_answer_quality(
        query=question,
        answer=presented,
        request_shape=shape,
    )

    assert calls == 1


def test_source_id_remains_machine_identity_across_runtime_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_id = "qd_4416_cut.pdf"
    candidate = RetrievedCandidate(
        candidate_id="chunk-1",
        collection="acne_knowledge",
        text="Evidence text.",
        rank=1,
        payload={
            "chunk_id": "chunk-1",
            "source_id": source_id,
            "source_url": "https://example.test/source",
        },
    )
    packed = pack_context(
        NormalizedQuery(original_query="question", normalized_text="question"),
        [candidate],
    )
    contexts = packed_context_to_response_contexts(packed)
    allowlist = build_source_allowlist([source_id], contexts)
    prompt = build_medical_prompt(
        question="question",
        contexts=contexts,
        available_sources=allowlist,
        packed_context_text=packed.context_text,
    )
    report = verify_answer_quality(
        query="question",
        answer="Answer.",
        packed_context=packed,
        final_source_ids=[source_id],
    )

    assert packed.items[0].payload["source_id"] == source_id
    assert contexts[0]["source_id"] == source_id
    assert allowlist[0]["source_id"] == source_id
    assert f"source_id={source_id};" in prompt
    assert report.metadata["evidence_locality"]["items"][0]["source_id"] == source_id
    assert report.metadata["evidence_locality"]["final_cited_source_ids"] == [source_id]

    monkeypatch.setitem(FILE_SOURCE_DISPLAY_NAMES, source_id, "Changed display label")
    changed = build_source_metadata([source_id], contexts)
    assert changed[0]["display_name"] == "Changed display label"
    assert changed[0]["source_id"] == source_id
    assert canonical_source_id(changed[0]) == source_id
    assert canonical_source_id({"display_name": "Presentation only"}) == ""

    validation = validate_answer_source_mentions(
        "See invented-guide.pdf and qd_4416_cut.pdf.",
        allowlist,
    )
    assert "invented-guide.pdf" not in validation.answer
    assert source_id in validation.answer
    assert validation.allowlist_source_ids == (source_id,)


@pytest.mark.asyncio
async def test_canonical_additive_fields_do_not_change_exact_cache_lookup_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.agent.nodes import cache as cache_node

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def fake_get_exact_cache(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))
        return None

    monkeypatch.setattr(cache_node, "get_exact_cache", fake_get_exact_cache)
    base_state = {
        "normalized_question": "same normalized question",
        "conversation_context": {"message_count": 0},
        "llm_provider": "gemini",
        "llm_model": "same-model",
        "pipeline_manifest": {"contract": "same"},
        "pipeline_fingerprint": "same-fingerprint",
    }

    before = await cache_node.cache_lookup_node(base_state)
    after = await cache_node.cache_lookup_node(
        {
            **base_state,
            "request_shape": {
                "wants_table": False,
                "required_columns": [],
                "exact_column_count": None,
                "exact_item_count": None,
                "style_constraints": [],
                "response_profile": "routine",
            },
            "query_identity": {
                "current_question": "same normalized question",
                "acquisition_query": "same normalized question",
                "overall_rerank_query": "same normalized question",
            },
        }
    )

    assert before == after
    assert calls[0] == calls[1]


@pytest.mark.asyncio
async def test_structural_and_evidence_availability_aliases_preserve_legacy_fields() -> None:
    availability = await workflow.assess_evidence_node(
        {
            "vector_contexts": [{"text": "Evidence", "source_id": "stable-source"}],
            "retrieval_attempt": 1,
        }
    )
    assert availability["evidence_availability"] == availability["evidence_assessment"]
    assert availability["evidence_availability"]["assessment_kind"] == (
        "provenance_complete_evidence_presence"
    )

    prepared = await prepare_request_node(
        {"user_question": "Mụn là gì?", "conversation_history": []}
    )
    verification = await answer_quality_node(
        {
            **prepared,
            "final_answer": "Mụn là một tình trạng da.",
            "sources": [],
            "source_validation": {},
        }
    )
    assert verification["structural_verification_report"] == (verification["answer_quality_report"])
    assert (
        verification["structural_verification_report"]["metadata"]["medical_semantic_verification"]
        is False
    )


@pytest.mark.asyncio
async def test_request_shape_does_not_change_deterministic_safety_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def forbidden_cache(_state: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("safety override must still run before exact cache")

    monkeypatch.setattr(workflow, "cache_lookup_node", forbidden_cache)
    prepared = await prepare_request_node(
        {
            "user_question": "Sau thuốc tôi khó thở và sưng lưỡi.",
            "conversation_history": [],
        }
    )
    guarded = await workflow.guard_node({**prepared, "bypass_cache": False})

    assert prepared["request_shape"]["response_profile"] == "routine"
    assert guarded["safety_override"] is True
    assert guarded["safety_severity"] == "emergency"
    assert (await workflow.decide_node(guarded))["next_action"] == "finalize"
