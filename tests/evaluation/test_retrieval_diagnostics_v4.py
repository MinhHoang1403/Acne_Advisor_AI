from __future__ import annotations

import copy
import json

import pytest

from evaluation.deterministic import contains_concept, deterministic_result, summarize_metrics
from evaluation.retrieval_diagnostics import (
    build_case_retrieval_diagnostics,
    summarize_retrieval_diagnostics,
    write_retrieval_diagnostic_report,
)
from src.agent.nodes import retrieve as retrieve_node
from src.retrieval.contracts import (
    ContextItem,
    NormalizedQuery,
    PackedContext,
    QueryExpansion,
    RerankScoreBreakdown,
    RerankTrace,
    RerankedCandidate,
    RetrievedCandidate,
)
from src.retrieval.diagnostics import build_retrieval_diagnostic_trace


def _case(**overrides: object) -> dict[str, object]:
    case: dict[str, object] = {
        "id": "diagnostic-case",
        "category": "core_knowledge",
        "question": "alpha la gi?",
        "expected_concepts": ["alpha"],
        "accepted_sources": [],
        "critical_case": False,
    }
    case.update(overrides)
    return case


def _stage(present: bool | None) -> dict[str, object]:
    return {
        "present": present,
        "rank": 1 if present else None,
        "candidate_ids": ["candidate-1"] if present else [],
    }


def _runtime_trace(
    *,
    concept: str = "alpha",
    candidate: bool | None = False,
    fusion: bool | None = False,
    rerank: bool | None = False,
    packed: bool | None = False,
    prompt: bool | None = False,
    original_query: str = "alpha la gi?",
    retrieval_query: str | None = None,
    normalized_query: str | None = None,
    source_present: bool | None = None,
    parsed_present: bool | None = None,
    chunk_present: bool | None = None,
    indexed_present: bool | None = None,
    source_name: str | None = None,
) -> dict[str, object]:
    retrieval_query = retrieval_query if retrieval_query is not None else original_query
    normalized_query = normalized_query if normalized_query is not None else retrieval_query
    raw_candidates = []
    if source_name:
        raw_candidates.append(
            {
                "candidate_id": "candidate-1",
                "channel": "dense",
                "source_name": source_name,
            }
        )
    return {
        "query_trace": {
            "original_query": original_query,
            "retrieval_query": retrieval_query,
            "normalized_query": normalized_query,
            "expanded_aliases": [],
        },
        "raw_candidate_trace": raw_candidates,
        "concept_traces": [
            {
                "concept": concept,
                "source_present": source_present,
                "parsed_present": parsed_present,
                "chunk_present": chunk_present,
                "indexed_present": indexed_present,
                "retrieval": {
                    "dense": _stage(candidate),
                    "sparse": _stage(False),
                    "entity": _stage(False),
                    "graph": _stage(None),
                    "candidate": _stage(candidate),
                },
                "fusion": _stage(fusion),
                "rerank": _stage(rerank),
                "packed_context": _stage(packed),
                "prompt_evidence": _stage(prompt),
            }
        ],
        "prompt_evidence_trace": {"concept_presence": {concept: prompt}},
    }


def _diagnostics(
    *,
    trace: dict[str, object],
    case: dict[str, object] | None = None,
    answer: str = "",
    verifier: dict[str, object] | None = None,
) -> dict[str, object]:
    return build_case_retrieval_diagnostics(
        result={
            "answer": answer,
            "retrieval_diagnostics": trace,
            "answer_quality_report": verifier or {},
        },
        case=case or _case(),
        concept_matcher=contains_concept,
    )


def _loss_stage(diagnostics: dict[str, object]) -> str:
    concepts = diagnostics["concept_diagnostics"]
    assert isinstance(concepts, list) and concepts
    return str(concepts[0]["loss_stage"])


def test_concept_retained_through_answer_has_no_loss() -> None:
    diagnostics = _diagnostics(
        trace=_runtime_trace(candidate=True, fusion=True, rerank=True, packed=True, prompt=True),
        answer="alpha duoc giai thich trong cau tra loi.",
    )

    assert _loss_stage(diagnostics) == "NOT_APPLICABLE"


def test_missing_raw_candidate_is_retrieval_miss() -> None:
    assert _loss_stage(_diagnostics(trace=_runtime_trace())) == "RETRIEVAL_MISS"


def test_candidate_lost_before_fusion_is_fusion_miss() -> None:
    assert _loss_stage(_diagnostics(trace=_runtime_trace(candidate=True))) == "FUSION_MISS"


def test_fusion_candidate_lost_before_rerank_is_rerank_miss() -> None:
    assert _loss_stage(_diagnostics(trace=_runtime_trace(candidate=True, fusion=True))) == "RERANK_MISS"


def test_reranked_candidate_not_packed_is_context_packing_miss() -> None:
    assert _loss_stage(
        _diagnostics(trace=_runtime_trace(candidate=True, fusion=True, rerank=True))
    ) == "CONTEXT_PACKING_MISS"


def test_packed_evidence_missing_from_answer_is_generation_miss() -> None:
    assert _loss_stage(
        _diagnostics(
            trace=_runtime_trace(candidate=True, fusion=True, rerank=True, packed=True, prompt=True)
        )
    ) == "GENERATION_MISS"


def test_query_rewrite_dropping_concept_is_query_understanding_miss() -> None:
    diagnostics = _diagnostics(
        trace=_runtime_trace(
            original_query="alpha la gi?",
            retrieval_query="thuoc tri mun la gi?",
            normalized_query="thuoc tri mun",
        )
    )

    assert _loss_stage(diagnostics) == "QUERY_UNDERSTANDING_MISS"


def test_critical_concepts_are_tracked_separately() -> None:
    case = _case(category="pregnancy_lactation", critical_case=True)
    diagnostics = _diagnostics(
        case=case,
        trace=_runtime_trace(candidate=True, fusion=True, rerank=True, packed=True, prompt=True),
        answer="alpha duoc giai thich trong cau tra loi.",
    )
    summary = summarize_retrieval_diagnostics(
        [{"case_id": case["id"], "category": case["category"], "retrieval_diagnostics": diagnostics}]
    )

    assert diagnostics["concept_diagnostics"][0]["critical"] is True
    assert summary["critical_concepts"]["total"] == 1
    assert summary["critical_concepts"]["answer_present"] == 1


def test_critical_verifier_miss_is_distinguished_from_generation_miss() -> None:
    diagnostics = _diagnostics(
        case=_case(category="pregnancy_lactation", critical_case=True),
        trace=_runtime_trace(candidate=True, fusion=True, rerank=True, packed=True, prompt=True),
        verifier={"required_facts": ["alpha"], "missing_facts": [], "issues": []},
    )

    assert _loss_stage(diagnostics) == "VERIFIER_MISS"


def test_legacy_context_metric_remains_readable_and_pack_metric_is_real() -> None:
    case = {
        **_case(accepted_sources=["source.pdf"]),
        "expected_entities": [],
        "expected_behavior": "cautious_answer",
        "acceptable_origins": ["llm_generated"],
        "expected_safety_level": "caution",
        "forbidden_claims": [],
        "source_required": False,
        "format_contract": {"type": "short_answer"},
        "naturalness_applicable": False,
        "notes": "",
    }
    raw = {
        "ok": True,
        "latency_ms": 1.0,
        "requested_provider": "ollama",
        "requested_model": "qwen3:8b",
        "result": {
            "answer": "alpha duoc giai thich trong cau tra loi.",
            "actual_provider": "ollama",
            "actual_model": "qwen3:8b",
            "fallback_applied": False,
            "fallback_type": "none",
            "medical_severity": "caution",
            "sources": ["source.pdf"],
            "retrieval_diagnostics": _runtime_trace(
                candidate=True,
                fusion=True,
                rerank=True,
                packed=False,
                prompt=False,
                source_name="source.pdf",
            ),
        },
    }

    metrics = summarize_metrics([deterministic_result(raw, case, "ollama", "qwen3:8b")])

    assert metrics["retrieval_and_grounding"]["context_evidence_retention"]["semantics"] == "legacy_source_hit_proxy_v3"
    assert metrics["retrieval_and_grounding"]["context_evidence_retention"]["value"] == 100.0
    assert metrics["retrieval_diagnostics"]["metrics"]["packed_context_recall"] == {
        "numerator": 0,
        "denominator": 1,
        "value": 0.0,
    }


def test_loss_stage_aggregation_counts_each_stage() -> None:
    first = _diagnostics(trace=_runtime_trace())
    second = _diagnostics(trace=_runtime_trace(candidate=True, fusion=True))
    summary = summarize_retrieval_diagnostics(
        [
            {"case_id": "case-retrieval", "category": "core_knowledge", "retrieval_diagnostics": first},
            {"case_id": "case-rerank", "category": "comparison", "retrieval_diagnostics": second},
        ]
    )

    assert summary["loss_stage_counts"]["RETRIEVAL_MISS"] == 1
    assert summary["loss_stage_counts"]["RERANK_MISS"] == 1
    assert summary["loss_stage_percentages"]["RETRIEVAL_MISS"] == 50.0


def test_affected_case_count_is_not_confused_with_concept_count() -> None:
    trace = _runtime_trace()
    trace["concept_traces"] = [
        trace["concept_traces"][0],
        {
            **trace["concept_traces"][0],
            "concept": "beta",
        },
    ]
    trace["prompt_evidence_trace"] = {"concept_presence": {"alpha": False, "beta": False}}
    case = _case(expected_concepts=["alpha", "beta"])
    diagnostics = _diagnostics(trace=trace, case=case)
    summary = summarize_retrieval_diagnostics(
        [{"case_id": "single-case", "category": "core_knowledge", "retrieval_diagnostics": diagnostics}]
    )

    assert summary["loss_stage_counts"]["RETRIEVAL_MISS"] == 2
    assert summary["loss_stage_cases_affected"]["RETRIEVAL_MISS"] == 1


def test_evaluation_label_issue_is_reported_without_mutating_case() -> None:
    case = _case(diagnostic_label_issue=True)
    original_case = copy.deepcopy(case)

    diagnostics = _diagnostics(trace=_runtime_trace(), case=case)

    assert _loss_stage(diagnostics) == "EVALUATION_LABEL_ISSUE"
    assert case == original_case


def test_diagnostic_data_serializes_stably_and_report_is_human_readable(tmp_path) -> None:
    diagnostics = _diagnostics(trace=_runtime_trace(candidate=True, fusion=True))
    rendered = json.dumps(diagnostics, ensure_ascii=False, sort_keys=True)
    summary = summarize_retrieval_diagnostics(
        [{"case_id": "stable-case", "category": "core_knowledge", "retrieval_diagnostics": diagnostics}]
    )
    report_path = write_retrieval_diagnostic_report(
        tmp_path / "RETRIEVAL_DIAGNOSTICS.md",
        summary,
        [{"case_id": "stable-case", "retrieval_diagnostics": diagnostics}],
    )

    assert rendered == json.dumps(diagnostics, ensure_ascii=False, sort_keys=True)
    assert json.loads(rendered)["version"] == "retrieval_diagnostics_v1"
    assert "Loss stage: `RERANK_MISS`" in report_path.read_text(encoding="utf-8")


def test_diagnostic_capture_does_not_mutate_retrieval_candidates() -> None:
    query = NormalizedQuery(original_query="alpha la gi?", normalized_text="alpha la gi", intent="general")
    expansion = QueryExpansion(original_query=query.original_query, normalized_query=query)
    candidate = RetrievedCandidate(
        candidate_id="chunk-1",
        source="chunk",
        collection="acne_chunks_v1",
        text="alpha evidence",
        score=0.8,
        fused_score=0.8,
        payload={"chunk_id": "chunk-1", "source_file": "source.pdf"},
        rank=1,
    )
    item = ContextItem(
        item_id="chunk-1",
        source="chunk",
        role="evidence",
        text="alpha evidence",
        payload={"chunk_id": "chunk-1", "source_file": "source.pdf"},
        score=0.8,
        fused_score=0.8,
        rank=1,
        reason="fixture",
    )
    packed = PackedContext(
        original_query=query.original_query,
        intent=query.intent,
        items=[item],
        context_text="alpha evidence",
        chunk_items_count=1,
        debug={"pack_trace": {"selected_chunk_ids": ["chunk-1"], "dropped_candidates": []}},
    )
    rerank = RerankTrace(
        provider="fixture",
        enabled=True,
        input_count=1,
        output_count=1,
        top_n=1,
        ranked_candidates=[
            RerankedCandidate(
                candidate=candidate,
                rerank_score=0.9,
                rerank_rank=1,
                score_breakdown=RerankScoreBreakdown(final_score=0.9),
            )
        ],
    )
    dense_results = [{"id": "chunk-1", "text": "alpha evidence", "score": 0.8, "source_file": "source.pdf"}]
    before_candidate = candidate.model_dump(mode="json")
    before_dense = copy.deepcopy(dense_results)

    trace = build_retrieval_diagnostic_trace(
        expected_concepts=["alpha"],
        critical=False,
        original_query=query.original_query,
        retrieval_query=query.original_query,
        normalized_query=query,
        expansion=expansion,
        dense_results=dense_results,
        sparse_results=[],
        entity_candidates=[],
        fused_results=dense_results,
        chunk_candidates=[candidate],
        merged_candidates=[candidate],
        reranked_candidates=[candidate],
        packed_context=packed,
        rerank_trace=rerank,
        context_max_items=5,
        context_max_chars=4200,
        warnings=[],
        concept_matcher=contains_concept,
    ).model_dump(mode="json")

    assert candidate.model_dump(mode="json") == before_candidate
    assert dense_results == before_dense
    assert trace["fusion_trace"]["rrf_output_candidate_ids"] == ["chunk-1"]
    assert trace["rerank_trace"]["ranked_candidates"][0]["rerank_score"] == 0.9
    assert trace["pack_trace"]["actual_context_char_count"] == len("alpha evidence")


@pytest.mark.asyncio
async def test_default_retrieval_path_preserves_existing_call_shape(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class Result:
        vector_contexts = [{"text": "alpha evidence", "source_file": "source.pdf"}]
        graph_facts: list[object] = []
        sources = ["source.pdf"]
        metadata = {"retrieval_trace": {}, "packed_context": {"items": [], "context_text": ""}}

    class FakeRetriever:
        async def retrieve(self, query: str, **kwargs: object) -> Result:
            calls.append((query, kwargs))
            return Result()

        async def close(self) -> None:
            return None

    monkeypatch.setattr(retrieve_node, "HybridRetriever", FakeRetriever)
    baseline = await retrieve_node.retrieve_context_node({"standalone_question": "alpha la gi?"})
    diagnostic = await retrieve_node.retrieve_context_node(
        {
            "user_question": "alpha la gi?",
            "standalone_question": "alpha la gi?",
            "evaluation_mode": True,
            "evaluation_expected_concepts": ["alpha"],
            "evaluation_critical_case": False,
            "evaluation_concept_matcher": contains_concept,
        }
    )

    assert calls[0] == ("alpha la gi?", {"top_k": 5})
    assert calls[1][0] == "alpha la gi?"
    assert calls[1][1]["top_k"] == 5
    assert calls[1][1]["diagnostic_expected_concepts"] == ["alpha"]
    assert baseline["vector_contexts"] == diagnostic["vector_contexts"]
    assert baseline["retrieval_diagnostics"] is None
