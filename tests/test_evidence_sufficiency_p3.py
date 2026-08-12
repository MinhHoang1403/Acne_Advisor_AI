from __future__ import annotations

import json

import pytest

from src.agent.nodes import retrieve as retrieve_node
from src.agent.nodes.evidence_sufficiency import (
    assess_evidence_sufficiency_node,
    build_evidence_retry_plan_node,
    evidence_abstention_node,
    route_after_evidence_sufficiency,
)
from src.observability.versioning import (
    build_pipeline_version_manifest,
    compute_pipeline_fingerprint,
)
from src.retrieval.evidence_sufficiency import (
    EvidenceAbstentionType,
    EvidenceSufficiencyStatus,
    P3ExecutionTrace,
    P3TraceEventType,
    RetryEligibility,
    assess_evidence_sufficiency,
    build_evidence_abstention,
    build_retry_plan,
    p3_max_attempts_from_env,
)


def _item(
    candidate_id: str,
    *,
    roles: tuple[str, ...] = ("primary", "source_traceability"),
    source_path: str | None = "clinical/source.pdf",
    document_id: str | None = "doc-1",
    chunk_id: str | None = None,
    critical: bool = False,
) -> dict:
    return {
        "evidence": {
            "candidate": {
                "candidate": {
                    "candidate_id": candidate_id,
                    "provenance": {
                        "point_id": f"point-{candidate_id}",
                        "chunk_id": chunk_id or f"chunk-{candidate_id}",
                        "document_id": document_id,
                        "source_path": source_path,
                    },
                    "text": "Source-backed acne evidence.",
                    "scores": {},
                    "metadata_features": [],
                },
                "rrf_rank": 1,
                "scores": {},
            },
            "input_rank": 1,
            "output_rank": 1,
            "scores": {},
            "fallback_used": False,
        },
        "roles": list(roles),
        "selection_reason": "fixture",
        "critical": critical,
    }


def _selector(
    items: list[dict],
    *,
    required_roles: tuple[str, ...] = ("primary", "source_traceability"),
    critical_flags: tuple[str, ...] = (),
) -> dict:
    covered = sorted({role for item in items for role in item["roles"]})
    missing = [role for role in required_roles if role not in covered]
    return {
        "selected_evidence": items,
        "status": "INSUFFICIENT" if missing else "SUFFICIENT",
        "missing_roles": missing,
        "satisfied_roles": covered,
        "requirements": {
            "required_roles": list(required_roles),
            "critical_safety_flags": list(critical_flags),
            "graph_required_roles": [],
        },
        "entity_signal_count": 0,
        "graph_signal_count": 0,
    }


def _packer(ids: list[str], *, status: str = "SUFFICIENT") -> dict:
    return {
        "selected_evidence_ids": ids,
        "rendered_blocks": ["evidence"] * len(ids),
        "context_text": "evidence" if ids else "",
        "clipped_evidence_ids": [],
        "omitted_evidence_ids": [],
        "drops": [],
        "character_count": 8 if ids else 0,
        "max_characters": 4200,
        "token_count": 2 if ids else 0,
        "token_count_mode": "approximate_chars_div_4",
        "max_tokens": 1050,
        "max_items": 5,
        "used_items": len(ids),
        "source_paths": ["clinical/source.pdf"] if ids else [],
        "critical_evidence_ids": [],
        "critical_evidence_preserved": status != "CRITICAL_EVIDENCE_OVERFLOW",
        "status": status,
    }


def _assess(
    items: list[dict],
    packed_ids: list[str],
    *,
    required_roles: tuple[str, ...] = ("primary", "source_traceability"),
    critical_flags: tuple[str, ...] = (),
    packer_status: str = "SUFFICIENT",
    retrieval_status: str = "success",
    in_domain: bool = True,
    attempt: int = 0,
):
    return assess_evidence_sufficiency(
        evidence_selector=_selector(
            items,
            required_roles=required_roles,
            critical_flags=critical_flags,
        ),
        evidence_packer=_packer(packed_ids, status=packer_status),
        retrieval_status=retrieval_status,
        is_in_domain=in_domain,
        attempt_index=attempt,
        trace_id="trace-p3",
    )


def test_sufficient_source_backed_evidence_passes_pre_and_post_pack():
    result = _assess([_item("e1")], ["e1"])

    assert result.pre_pack.status == EvidenceSufficiencyStatus.SUFFICIENT
    assert result.post_pack.status == EvidenceSufficiencyStatus.SUFFICIENT
    assert result.final.provenance_valid is True
    assert result.final.retry_eligibility == RetryEligibility.NON_RETRYABLE


def test_entity_and_graph_signals_cannot_satisfy_source_evidence():
    selector = _selector([])
    selector["entity_signal_count"] = 1
    selector["graph_signal_count"] = 1
    result = assess_evidence_sufficiency(
        evidence_selector=selector,
        evidence_packer=_packer([]),
        retrieval_status="no_evidence",
        is_in_domain=True,
        attempt_index=0,
        trace_id="trace-p3",
    )

    assert result.final.status == EvidenceSufficiencyStatus.INSUFFICIENT
    assert result.final.evidence_ids == ()
    assert result.final.retry_eligibility == RetryEligibility.RETRYABLE


def test_invalid_provenance_never_counts_as_sufficient():
    invalid = _item("e1", source_path=None, document_id=None, chunk_id="chunk-e1")
    result = _assess([invalid], ["e1"])
    abstention = build_evidence_abstention(result.final)

    assert result.final.status == EvidenceSufficiencyStatus.INSUFFICIENT
    assert result.final.provenance_valid is False
    assert "SOURCE_PROVENANCE_FAILURE" in result.final.reasons
    assert abstention.abstention_type == EvidenceAbstentionType.SOURCE_PROVENANCE_FAILURE


def test_critical_source_role_missing_is_distinct_and_retryable_once():
    result = _assess(
        [_item("e1")],
        ["e1"],
        required_roles=("primary", "source_traceability", "safety"),
        critical_flags=("pregnancy",),
    )

    assert result.final.status == EvidenceSufficiencyStatus.CRITICAL_EVIDENCE_MISSING
    assert result.final.critical_missing_roles == ("safety",)
    assert result.final.retry_eligibility == RetryEligibility.RETRYABLE

    second = _assess(
        [_item("e1")],
        ["e1"],
        required_roles=("primary", "source_traceability", "safety"),
        critical_flags=("pregnancy",),
        attempt=1,
    )
    assert second.final.retry_eligibility == RetryEligibility.NON_RETRYABLE
    assert build_evidence_abstention(second.final).attempts == 2


def test_safety_role_without_explicit_critical_evidence_is_not_enough():
    result = _assess(
        [_item("e1", roles=("primary", "source_traceability", "safety"))],
        ["e1"],
        required_roles=("primary", "source_traceability", "safety"),
        critical_flags=("pregnancy",),
    )
    assert result.final.status == EvidenceSufficiencyStatus.CRITICAL_EVIDENCE_MISSING
    assert result.final.critical_missing_roles == ("critical",)


def test_critical_packer_overflow_is_non_retryable():
    item = _item(
        "safety-1",
        roles=("primary", "source_traceability", "safety", "critical"),
        critical=True,
    )
    result = _assess(
        [item],
        [],
        required_roles=("primary", "source_traceability", "safety"),
        critical_flags=("pregnancy",),
        packer_status="CRITICAL_EVIDENCE_OVERFLOW",
    )

    assert result.pre_pack.status == EvidenceSufficiencyStatus.SUFFICIENT
    assert result.post_pack.status == EvidenceSufficiencyStatus.CRITICAL_EVIDENCE_MISSING
    assert result.final.retry_eligibility == RetryEligibility.NON_RETRYABLE
    assert "PACKER_CRITICAL_EVIDENCE_OVERFLOW" in result.final.reasons


@pytest.mark.parametrize(
    ("retrieval_status", "in_domain", "expected_type"),
    [
        ("recoverable_error", True, EvidenceAbstentionType.RETRIEVAL_PROVIDER_FAILURE),
        ("success", False, EvidenceAbstentionType.OUT_OF_SCOPE),
    ],
)
def test_non_retryable_provider_and_scope_failures(retrieval_status, in_domain, expected_type):
    result = _assess(
        [],
        [],
        retrieval_status=retrieval_status,
        in_domain=in_domain,
    )
    assert result.final.retry_eligibility == RetryEligibility.NON_RETRYABLE
    assert build_evidence_abstention(result.final).abstention_type == expected_type


def test_retry_plan_is_different_bounded_and_uses_structural_hints():
    result = _assess([], [])
    plan = build_retry_plan(
        original_query="Tazorac có phải kháng sinh không?",
        assessment=result.final,
        retrieval_trace_v5={
            "query_observation": {"normalized_entity_ids": ["tazarotene"]},
            "entity_signals": [
                {
                    "canonical_name": "Tazarotene",
                    "matched_terms": ["Tazorac"],
                }
            ],
            "graph_signals": [
                {
                    "relation_path": ["BELONGS_TO_CLASS"],
                    "target_entity_id": "topical_retinoid",
                }
            ],
        },
    )

    assert plan.query_variant != "Tazorac có phải kháng sinh không?"
    assert plan.original_query_hash != plan.retry_query_hash
    assert plan.attempt_index == 1
    assert "Tazarotene" in plan.entity_expansion
    assert "BELONGS_TO_CLASS" in plan.graph_hints
    assert len(plan.query_variant) <= 1000


def test_p3_max_attempts_is_hard_clamped(monkeypatch):
    monkeypatch.setenv("P3_MAX_RETRIEVAL_ATTEMPTS", "99")
    assert p3_max_attempts_from_env() == 2
    monkeypatch.setenv("P3_MAX_RETRIEVAL_ATTEMPTS", "0")
    assert p3_max_attempts_from_env() == 1


def test_pipeline_fingerprint_includes_p3_without_bumping_cache_version():
    disabled = build_pipeline_version_manifest(
        {
            "P3_EVIDENCE_SUFFICIENCY_ENABLED": "false",
            "CACHE_ANSWER_VERSION": "v5",
        }
    )
    enabled = build_pipeline_version_manifest(
        {
            "P3_EVIDENCE_SUFFICIENCY_ENABLED": "true",
            "CACHE_ANSWER_VERSION": "v5",
        }
    )

    assert disabled["answer_cache_version"] == enabled["answer_cache_version"] == "v5"
    assert compute_pipeline_fingerprint(disabled) != compute_pipeline_fingerprint(enabled)
    assert enabled["p3_max_retrieval_attempts"] == 2


@pytest.mark.asyncio
async def test_nodes_route_first_pass_retry_then_second_pass_abstention():
    base = {
        "pipeline_manifest": {
            "retrieval_pipeline_version": "v5",
            "p3_evidence_sufficiency_enabled": True,
        },
        "standalone_question": "Tài liệu có đủ không?",
        "retrieval_attempt": 0,
        "retrieval_status": "no_evidence",
        "is_in_domain": True,
        "evidence_selector": _selector([]),
        "evidence_packer": _packer([]),
        "retrieval_trace_v5": {"trace_id": "trace-p3", "query_observation": {}},
        "retry_history": [],
        "performance_timings": {},
        "p3_trace": None,
    }
    first = await assess_evidence_sufficiency_node(base)
    first_state = {**base, **first}
    assert route_after_evidence_sufficiency(first_state) == "build_retry_plan"

    planned = await build_evidence_retry_plan_node(first_state)
    assert planned["retrieval_attempt"] == 1
    second_state = {**first_state, **planned, "retrieval_status": "no_evidence"}
    second = await assess_evidence_sufficiency_node(second_state)
    second_state.update(second)
    assert route_after_evidence_sufficiency(second_state) == "evidence_abstention"

    abstained = await evidence_abstention_node(second_state)
    assert abstained["fallback_applied"] is True
    assert abstained["fallback_cache_eligible"] is False
    assert abstained["abstention"]["attempts"] == 2
    events = [event["event_type"] for event in abstained["p3_trace"]["events"]]
    assert events == [
        P3TraceEventType.SUFFICIENCY_ASSESSED.value,
        P3TraceEventType.RETRY_PLANNED.value,
        P3TraceEventType.RETRY_STARTED.value,
        P3TraceEventType.RETRY_COMPLETED.value,
        P3TraceEventType.SUFFICIENCY_ASSESSED.value,
        P3TraceEventType.RETRY_SKIPPED.value,
        P3TraceEventType.ABSTENTION_TRIGGERED.value,
    ]
    assert json.loads(json.dumps(abstained["p3_trace"]))
    P3ExecutionTrace.model_validate(abstained["p3_trace"])


@pytest.mark.asyncio
async def test_nodes_do_not_retry_sufficient_first_pass():
    item = _item("e1")
    state = {
        "pipeline_manifest": {
            "retrieval_pipeline_version": "v5",
            "p3_evidence_sufficiency_enabled": True,
        },
        "standalone_question": "Mụn đầu đen hình thành thế nào?",
        "retrieval_attempt": 0,
        "retrieval_status": "success",
        "is_in_domain": True,
        "evidence_selector": _selector([item]),
        "evidence_packer": _packer(["e1"]),
        "retrieval_trace_v5": {"trace_id": "trace-p3"},
        "retry_history": [],
        "performance_timings": {},
        "p3_trace": None,
    }
    result = await assess_evidence_sufficiency_node(state)
    assert route_after_evidence_sufficiency({**state, **result}) == "fallback_decision"
    assert len(result["retry_history"]) == 1
    assert [event["event_type"] for event in result["p3_trace"]["events"]] == [
        P3TraceEventType.SUFFICIENCY_ASSESSED.value,
        P3TraceEventType.RETRY_SKIPPED.value,
    ]


@pytest.mark.asyncio
async def test_v4_rollback_bypasses_p3_policy():
    state = {
        "pipeline_manifest": {
            "retrieval_pipeline_version": "v4",
            "p3_evidence_sufficiency_enabled": True,
        },
        "retrieval_attempt": 0,
        "p3_trace": None,
    }
    result = await assess_evidence_sufficiency_node(state)
    assert result["p3_active"] is False
    assert route_after_evidence_sufficiency({**state, **result}) == "fallback_decision"


@pytest.mark.asyncio
async def test_retry_retrieval_uses_query_variant_and_preserves_attempt_namespace(monkeypatch):
    captured = []

    class Result:
        vector_contexts = [{"text": "evidence", "source_file": "source.pdf"}]
        graph_facts = []
        sources = ["source.pdf"]
        metadata = {
            "retrieval_trace": {"timings_ms": {"total": 4.0}},
            "retrieval_v5": {"trace": {"trace_id": "retry-trace"}},
            "evidence_selector": _selector([_item("retry-e1")]),
            "evidence_packer": _packer(["retry-e1"]),
            "packed_context": {"items": [{"text": "evidence"}], "context_text": "evidence"},
        }

    class FakeRetriever:
        async def retrieve(self, query, top_k):
            captured.append((query, top_k))
            return Result()

        async def close(self):
            return None

    monkeypatch.setattr(retrieve_node, "HybridRetriever", FakeRetriever)
    result = await retrieve_node.retrieve_context_node(
        {
            "standalone_question": "original query",
            "retrieval_attempt": 1,
            "retry_plan": {"query_variant": "different retry query"},
            "performance_timings": {"retrieval_attempt_0_total": 3.0},
        }
    )

    assert captured == [("different retry query", 5)]
    assert result["performance_timings"] == {
        "retrieval_attempt_0_total": 3.0,
        "retrieval_attempt_1_total": 4.0,
    }
    assert result["evidence_selector"] is not None
    assert result["evidence_packer"] is not None
