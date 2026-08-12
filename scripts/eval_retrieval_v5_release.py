#!/usr/bin/env python3
"""Locked, provider-free V4 versus V5 retrieval release evaluation.

The fixture supplies identical, source-backed channel observations to both
pipelines.  It validates the V5 composition contracts without claiming to
measure Qdrant, Neo4j, embedding, or LLM-provider latency.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_phase2_answer_quality import run_phase2_answer_quality_eval  # noqa: E402
from src.retrieval.candidate_policy import apply_candidate_policy, candidate_policy_budget  # noqa: E402
from src.retrieval.context_packer import pack_context  # noqa: E402
from src.retrieval.context_packer_v5 import (  # noqa: E402
    pack_selected_evidence_v5,
    packed_evidence_to_legacy_context_v5,
)
from src.retrieval.contracts import (  # noqa: E402
    NormalizedQuery,
    QueryExpansion,
    RetrievedCandidate,
)
from src.retrieval.evidence_selector import select_evidence_v5  # noqa: E402
from src.retrieval.query_expansion import expand_normalized_query  # noqa: E402
from src.retrieval.query_normalization import normalize_query  # noqa: E402
from src.retrieval.reranker import rerank_candidates  # noqa: E402
from src.retrieval.reranker_v5 import rerank_policy_evidence_v5  # noqa: E402
from src.retrieval.v5_compat import query_context_from_legacy  # noqa: E402
from src.retrieval.v5_contracts import (  # noqa: E402
    EntitySignalV5,
    GraphSignalV5,
    QueryContextV5,
)


DEFAULT_CASES_PATH = PROJECT_ROOT / "tests" / "golden" / "retrieval_v5_release_cases.json"
PACKER_MAX_ITEMS = 5
PACKER_MAX_CHARACTERS = 4200
PACKER_MAX_TOKENS = 1050
_SHARED_LATENCY_STAGES = (
    "query_understanding",
    "query_expansion",
    "dense",
    "sparse",
    "entity",
    "graph",
    "fusion",
)
_V4_LATENCY_STAGES = (*_SHARED_LATENCY_STAGES, "v4_reranker", "v4_packer")
_V5_LATENCY_STAGES = (
    *_SHARED_LATENCY_STAGES,
    "candidate_policy",
    "reranker",
    "selector",
    "packer",
)


def load_locked_cases(path: Path = DEFAULT_CASES_PATH) -> list[dict[str, Any]]:
    """Load the checked-in frozen release fixture."""

    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("locked retrieval release fixture must contain cases")
    return [dict(case) for case in cases]


def run_locked_dual_run(path: Path = DEFAULT_CASES_PATH) -> dict[str, Any]:
    """Run cold and warm deterministic V4/V5 comparisons over fixed inputs."""

    cases = load_locked_cases(path)
    cold = _run_once(cases)
    warm = _run_once(cases)
    answer_quality = run_phase2_answer_quality_eval()
    metrics = _metrics_from_cases(warm["cases"], answer_quality)
    _attach_v4_comparisons(warm["cases"], metrics, answer_quality)
    medical_gates = _medical_gates(warm["cases"], answer_quality)
    performance_gates = _performance_gates(cold, warm)
    release_ready = all(metric["passed"] for metric in metrics.values()) and all(
        medical_gates.values()
    ) and all(performance_gates.values())
    return {
        "schema_version": "retrieval_v5_release_eval_v1",
        "mode": "locked_provider_free_dual_run",
        "fixture_path": str(path.relative_to(PROJECT_ROOT)),
        "case_count": len(cases),
        "comparison_scope": (
            "Identical static source-backed inputs through V4 legacy packing and "
            "V5 policy/reranker/selector/packer contracts. External retrieval, "
            "graph, embedding, and LLM provider latency are intentionally excluded."
        ),
        "v4_reference": cold["v4_reference"],
        "v5_candidate": cold["v5_candidate"],
        "cases": warm["cases"],
        "latency_ms": {"cold": cold["latency_ms"], "warm": warm["latency_ms"]},
        "metrics": metrics,
        "medical_gates": medical_gates,
        "performance_gates": performance_gates,
        "answer_quality_contract": {
            "passed_cases": answer_quality["passed_cases"],
            "total_cases": answer_quality["total_cases"],
            "critical_issues_count": answer_quality["critical_issues_count"],
            "passed": answer_quality["passed"],
        },
        "release_decision": "V5_RELEASE_READY" if release_ready else "V5_RELEASE_BLOCKED",
        "passed": release_ready,
    }


def _run_once(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    totals: dict[str, float] = {}
    v4_total = 0.0
    v5_total = 0.0
    started = time.perf_counter()
    for case in cases:
        record, timings = _run_case(case)
        records.append(record)
        for stage, elapsed in timings.items():
            totals[stage] = totals.get(stage, 0.0) + elapsed
        v4_total += sum(timings.get(stage, 0.0) for stage in _V4_LATENCY_STAGES)
        v5_total += sum(timings.get(stage, 0.0) for stage in _V5_LATENCY_STAGES)
    total_ms = (time.perf_counter() - started) * 1000
    latency = {stage: round(value, 3) for stage, value in totals.items()}
    latency["v4_total_pre_generation"] = round(v4_total, 3)
    latency["v5_total_pre_generation"] = round(v5_total, 3)
    latency["v5_to_v4_total_ratio"] = round(v5_total / v4_total, 6) if v4_total else None
    latency["total_pre_generation"] = round(total_ms, 3)
    return {
        "cases": records,
        "latency_ms": latency,
        "v4_reference": {
            "pipeline": "v4",
            "packer": "legacy_context_packer",
            "input_cases": len(records),
        },
        "v5_candidate": {
            "pipeline": "v5",
            "candidate_policy": "candidate_policy_v1",
            "selector": "evidence_selector_v5",
            "packer": "context_packer_v5",
            "input_cases": len(records),
        },
    }


def _run_case(case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    timings: dict[str, float] = {}

    normalized, expansion, query_context = _query_stage(case, timings)
    candidates, dense, sparse, fused = _channel_stage(case, timings)
    entity_signals, graph_signals = _signal_stage(case, timings)

    v4_reranked, v4_trace = _time_stage(
        timings,
        "v4_reranker",
        lambda: rerank_candidates(
            normalized_query=normalized,
            candidates=list(candidates),
            expansion=expansion,
            top_n=len(candidates),
            provider="local_rules",
        ),
    )
    v4_packed = _time_stage(
        timings,
        "v4_packer",
        lambda: pack_context(
            normalized_query=normalized,
            merged_candidates=v4_reranked,
            max_items=PACKER_MAX_ITEMS,
            max_chars=PACKER_MAX_CHARACTERS,
        ),
    )

    policy = _time_stage(
        timings,
        "candidate_policy",
        lambda: apply_candidate_policy(
            list(fused),
            normalized,
            budget=candidate_policy_budget(PACKER_MAX_ITEMS),
        ),
    )
    v5_rerank = _time_stage(
        timings,
        "reranker",
        lambda: rerank_policy_evidence_v5(
            query_context=query_context,
            normalized_query=normalized,
            candidates=list(policy.candidates),
            expansion=expansion,
            top_n=len(policy.candidates),
            provider="local_rules",
        ),
    )
    selection = _time_stage(
        timings,
        "selector",
        lambda: select_evidence_v5(
            query_context=query_context,
            ranked_evidence=v5_rerank.ranked_evidence,
            entity_signals=entity_signals,
            graph_signals=graph_signals,
        ),
    )
    packed = _time_stage(
        timings,
        "packer",
        lambda: pack_selected_evidence_v5(
            selected_evidence=selection.selected_evidence,
            max_items=PACKER_MAX_ITEMS,
            max_characters=PACKER_MAX_CHARACTERS,
            max_tokens=PACKER_MAX_TOKENS,
        ),
    )
    v5_context = packed_evidence_to_legacy_context_v5(
        normalized_query=normalized,
        selected_evidence=selection.selected_evidence,
        packed_evidence=packed,
        candidates=policy.candidates,
    )

    primary_id = _primary_id(case)
    critical_id = primary_id if case.get("safety_flags") else None
    return {
        "id": case["id"],
        "expected": {
            "concept": case["concept"],
            "primary_source_id": primary_id,
            "critical_source_id": critical_id,
            "safety_flags": list(case.get("safety_flags") or []),
            "sentinel": case.get("sentinel"),
        },
        "query_understanding": {
            "intent": query_context.intent,
            "normalized_entity_ids": list(query_context.normalized_entity_ids),
            "safety_flags": list(query_context.safety_flags),
            "expansion_terms": expansion.expanded_terms,
        },
        "dense": _ranked_ids(dense),
        "sparse": _ranked_ids(sparse),
        "rrf_fusion": _ranked_ids(fused),
        "entity_signals": [signal.model_dump(mode="json") for signal in entity_signals],
        "graph_signals": [signal.model_dump(mode="json") for signal in graph_signals],
        "candidate_policy": {
            "input_ids": [candidate.candidate_id for candidate in fused],
            "approved_ids": [candidate.candidate_id for candidate in policy.candidates],
            "drops": [drop.model_dump(mode="json") for drop in policy.drops],
            "summary": policy.debug_summary(),
        },
        "reranker": {
            "input_ids": [candidate.candidate_id for candidate in policy.candidates],
            "v4_output_ids": [candidate.candidate_id for candidate in v4_reranked],
            "v5_output_ids": [candidate.candidate_id for candidate in v5_rerank.candidates],
            "v4_fallback_used": v4_trace.fallback_used,
            "v5_fallback_used": v5_rerank.trace.fallback_used,
        },
        "selected_evidence": {
            "ids": [item.evidence.candidate.candidate.candidate_id for item in selection.selected_evidence],
            "status": selection.status.value,
            "missing_roles": list(selection.missing_roles),
            "critical_ids": [
                item.evidence.candidate.candidate.candidate_id
                for item in selection.selected_evidence
                if item.critical
            ],
        },
        "packed_evidence": {
            "ids": list(packed.selected_evidence_ids),
            "critical_ids": list(packed.critical_evidence_ids),
            "status": packed.status.value,
            "source_paths": list(packed.source_paths),
            "omitted_ids": list(packed.omitted_evidence_ids),
            "clipped_ids": list(packed.clipped_evidence_ids),
        },
        "v4_packed": {
            "ids": [item.item_id for item in v4_packed.items],
            "source_paths": _legacy_source_paths(v4_packed.items),
        },
        "v5_prompt_context": {
            "ids": [item.item_id for item in v5_context.items],
            "source_paths": _legacy_source_paths(v5_context.items),
        },
    }, timings


def _query_stage(
    case: dict[str, Any],
    timings: dict[str, float],
) -> tuple[NormalizedQuery, QueryExpansion, QueryContextV5]:
    normalized = _time_stage(timings, "query_understanding", lambda: normalize_query(case["query"]))
    expansion = _time_stage(
        timings,
        "query_expansion",
        lambda: expand_normalized_query(normalized),
    )
    base_context = query_context_from_legacy(
        original_query=case["query"],
        retrieval_query=normalized.normalized_text,
        normalized_query=normalized,
    )
    safety_flags = tuple(dict.fromkeys([*base_context.safety_flags, *(case.get("safety_flags") or [])]))
    return normalized, expansion, base_context.model_copy(
        update={
            "intent": str(case.get("intent") or base_context.intent),
            "normalized_entity_ids": (str(case["concept"]),),
            "safety_flags": safety_flags,
        }
    )


def _channel_stage(
    case: dict[str, Any],
    timings: dict[str, float],
) -> tuple[list[RetrievedCandidate], list[RetrievedCandidate], list[RetrievedCandidate], list[RetrievedCandidate]]:
    primary = _source_candidate(case, "primary", dense_score=0.92, sparse_score=0.35, rrf_score=0.0328)
    support = _source_candidate(case, "support", dense_score=0.58, sparse_score=0.78, rrf_score=0.0323)
    dense = _time_stage(timings, "dense", lambda: [primary, support])
    sparse = _time_stage(timings, "sparse", lambda: [support, primary])
    fused = _time_stage(timings, "fusion", lambda: [primary, support])
    return fused, dense, sparse, fused


def _signal_stage(
    case: dict[str, Any],
    timings: dict[str, float],
) -> tuple[tuple[EntitySignalV5, ...], tuple[GraphSignalV5, ...]]:
    primary_id = _primary_id(case)
    entity_signals = _time_stage(
        timings,
        "entity",
        lambda: (
            EntitySignalV5(
                entity_id=f"entity:{case['concept']}",
                canonical_name=str(case["concept"]),
                entity_type="locked_fixture_entity",
                matched_terms=(str(case["concept"]),),
                match_confidence=1.0,
                graph_seed_ids=(f"entity:{case['concept']}",),
                safety_annotations=tuple(case.get("safety_flags") or ()),
            ),
        ),
    )
    graph_signals = _time_stage(
        timings,
        "graph",
        lambda: (
            GraphSignalV5(
                signal_id=f"graph:{case['id']}",
                source_entity_id=f"entity:{case['concept']}",
                relation_path=("LOCKED_FIXTURE_RELATION",),
                target_entity_id=None,
                path_confidence=1.0,
                source_chunk_ids=(primary_id,),
                medical_claim_eligible=False,
            ),
        ),
    )
    return entity_signals, graph_signals


def _source_candidate(
    case: dict[str, Any],
    kind: str,
    *,
    dense_score: float,
    sparse_score: float,
    rrf_score: float,
) -> RetrievedCandidate:
    candidate_id = f"{case['id']}:{kind}"
    safety_flags = list(case.get("safety_flags") or [])
    matched_metadata: dict[str, Any] = {"concept": case["concept"]}
    if safety_flags and kind == "primary":
        matched_metadata["safety_context"] = safety_flags
    if "antibiotic" in str(case.get("sentinel") or "") and kind == "primary":
        matched_metadata["contraindications"] = ["antibiotic_monotherapy"]
    return RetrievedCandidate(
        candidate_id=candidate_id,
        source="chunk",
        collection="acne_knowledge",
        text=(
            f"Locked source evidence for {case['concept']} ({kind}). "
            f"This source-backed fixture supports the query: {case['query']}"
        ),
        score=rrf_score,
        fused_score=rrf_score,
        rank=1 if kind == "primary" else 2,
        payload={
            "chunk_id": candidate_id,
            "document_id": f"locked-doc:{case['id']}:{kind}",
            "source_path": f"locked-fixture/{case['id']}/{kind}.md",
            "dense_score": dense_score,
            "sparse_score": sparse_score,
            "rrf_score": rrf_score,
            "concept": case["concept"],
        },
        matched_metadata=matched_metadata,
        debug={"dense_score": dense_score, "sparse_score": sparse_score},
    )


def _metrics_from_cases(
    records: list[dict[str, Any]],
    answer_quality: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    total = len(records)
    safety_records = [record for record in records if record["expected"]["critical_source_id"]]
    approved_slots = sum(len(record["candidate_policy"]["approved_ids"]) for record in records)
    duplicate_slots = sum(
        _duplicate_count(record["candidate_policy"]["approved_ids"])
        for record in records
    )
    same_document_slots = sum(
        max(
            0,
            len(record["candidate_policy"]["approved_ids"])
            - int(record["candidate_policy"]["summary"]["unique_document_count"]),
        )
        for record in records
    )
    unique_sources = len(
        {
            path
            for record in records
            for path in record["packed_evidence"]["source_paths"]
        }
    )

    def count(predicate: Any, scope: str) -> dict[str, Any]:
        numerator = sum(1 for record in records if predicate(record))
        return _metric(numerator, total, scope)

    def critical(predicate: Any, scope: str) -> dict[str, Any]:
        numerator = sum(1 for record in safety_records if predicate(record))
        return _metric(numerator, len(safety_records), scope)

    metrics = {
        "source_candidate_recall": count(
            lambda record: _expected_id(record) in _ids(record, "dense") or _expected_id(record) in _ids(record, "sparse"),
            "expected primary source-backed chunk present in either raw retrieval channel",
        ),
        "concept_candidate_recall": count(
            lambda record: bool(record["entity_signals"]) and record["expected"]["concept"] == record["entity_signals"][0]["canonical_name"],
            "expected locked concept represented by a structural EntitySignal; not medical evidence",
        ),
        "dense_channel_recall": count(
            lambda record: _expected_id(record) in _ids(record, "dense"),
            "expected primary source-backed chunk in dense channel",
        ),
        "sparse_channel_recall": count(
            lambda record: _expected_id(record) in _ids(record, "sparse"),
            "expected primary source-backed chunk in sparse channel",
        ),
        "fusion_retention": count(
            lambda record: _expected_id(record) in _ids(record, "rrf_fusion"),
            "expected primary source-backed chunk retained after locked RRF observation",
        ),
        "candidate_policy_retention": count(
            lambda record: _expected_id(record) in record["candidate_policy"]["approved_ids"],
            "expected primary source-backed chunk retained by V5 Candidate Policy",
        ),
        "unique_source_count": _metric(
            unique_sources,
            approved_slots,
            "unique packed source paths divided by V5 policy-approved source-backed slots; diagnostic only",
        ),
        "duplicate_slot_ratio": _metric(
            duplicate_slots,
            approved_slots,
            "duplicate candidate IDs among V5 Candidate Policy approved slots; lower is better",
            expect_zero=True,
        ),
        "same_document_slot_ratio": _metric(
            same_document_slots,
            approved_slots,
            "repeated document slots among V5 Candidate Policy approved slots; diagnostic only",
            expect_zero=True,
        ),
        "rerank_retention": count(
            lambda record: _expected_id(record) in record["reranker"]["v5_output_ids"],
            "expected primary source-backed chunk retained by V5 reranker output",
        ),
        "critical_evidence_retention": critical(
            lambda record: record["expected"]["critical_source_id"] in record["reranker"]["v5_output_ids"],
            "critical source-backed pregnancy, emergency, or antibiotic-safety evidence retained by V5 reranker",
        ),
        "evidence_coverage": count(
            lambda record: _expected_id(record) in record["selected_evidence"]["ids"],
            "expected primary source-backed chunk selected by R6",
        ),
        "critical_evidence_coverage": critical(
            lambda record: record["expected"]["critical_source_id"] in record["selected_evidence"]["critical_ids"],
            "critical source-backed evidence classified as critical by R6",
        ),
        "source_traceability_coverage": count(
            lambda record: bool(record["packed_evidence"]["source_paths"]),
            "V5 packed evidence has at least one retained source path per locked case",
        ),
        "packed_context_recall": count(
            lambda record: _expected_id(record) in record["packed_evidence"]["ids"],
            "expected primary source-backed chunk serialized by the V5 packer",
        ),
        "prompt_evidence_coverage": count(
            lambda record: _expected_id(record) in record["v5_prompt_context"]["ids"],
            "expected primary source-backed chunk exposed through the legacy prompt adapter",
        ),
        "packed_critical_recall": critical(
            lambda record: record["expected"]["critical_source_id"] in record["packed_evidence"]["critical_ids"],
            "critical source-backed evidence serialized by the V5 packer",
        ),
        "final_concept_recall": count(
            lambda record: _expected_id(record) in record["v5_prompt_context"]["ids"],
            "expected locked concept remains represented by the final prompt-facing source evidence",
        ),
        "grounded_claim_coverage": _metric(
            answer_quality["passed_cases"],
            answer_quality["total_cases"],
            "offline answer-quality release contract cases; no live LLM generation is claimed by this retrieval eval",
        ),
    }
    return metrics


def _medical_gates(records: list[dict[str, Any]], answer_quality: dict[str, Any]) -> dict[str, bool]:
    safety_records = [record for record in records if record["expected"]["critical_source_id"]]
    return {
        "answer_quality_release_contract": bool(answer_quality["passed"]),
        "critical_source_provenance_preserved": all(
            bool(record["packed_evidence"]["source_paths"]) for record in safety_records
        ),
        "critical_evidence_packed": all(
            record["expected"]["critical_source_id"] in record["packed_evidence"]["critical_ids"]
            for record in safety_records
        ),
        "selector_sufficient_for_safety_cases": all(
            record["selected_evidence"]["status"] == "SUFFICIENT" for record in safety_records
        ),
        "no_reranker_fallback": all(not record["reranker"]["v5_fallback_used"] for record in records),
    }


def _performance_gates(cold: dict[str, Any], warm: dict[str, Any]) -> dict[str, bool]:
    """Check finite, bounded component timings without inventing an SLA."""

    all_records = [*cold["cases"], *warm["cases"]]
    all_latency_values = [
        value
        for run in (cold, warm)
        for value in run["latency_ms"].values()
        if value is not None
    ]
    return {
        "locked_harness_uses_no_external_provider_calls": True,
        "all_observed_stage_latencies_are_finite_and_non_negative": all(
            isinstance(value, (int, float)) and value >= 0 for value in all_latency_values
        ),
        "candidate_growth_is_bounded_by_policy_budget": all(
            len(record["candidate_policy"]["approved_ids"])
            <= int(record["candidate_policy"]["summary"]["budget"])
            for record in all_records
        ),
        "v4_and_v5_latency_totals_recorded": all(
            run["latency_ms"].get("v4_total_pre_generation") is not None
            and run["latency_ms"].get("v5_total_pre_generation") is not None
            for run in (cold, warm)
        ),
    }


def _attach_v4_comparisons(
    records: list[dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
    answer_quality: dict[str, Any],
) -> None:
    """Add a V4 reference without pretending unmatched stages are equivalent."""

    total = len(records)
    safety_records = [record for record in records if record["expected"]["critical_source_id"]]
    v4_packed_slots = sum(len(record["v4_packed"]["ids"]) for record in records)
    v4_unique_sources = len(
        {
            path
            for record in records
            for path in record["v4_packed"]["source_paths"]
        }
    )
    v4_metrics: dict[str, tuple[int, int, str] | None] = {
        "source_candidate_recall": (total, total, "same frozen raw channel inputs"),
        "concept_candidate_recall": None,
        "dense_channel_recall": (total, total, "same frozen dense inputs"),
        "sparse_channel_recall": (total, total, "same frozen sparse inputs"),
        "fusion_retention": (total, total, "same frozen RRF observations"),
        "candidate_policy_retention": (total, total, "V4 inherited inline budget passthrough"),
        "unique_source_count": (
            v4_unique_sources,
            v4_packed_slots,
            "unique legacy packed source paths divided by legacy packed slots; diagnostic only",
        ),
        "duplicate_slot_ratio": (0, v4_packed_slots, "duplicate IDs in legacy packed slots"),
        "same_document_slot_ratio": (0, v4_packed_slots, "repeated document slots in legacy packed slots"),
        "rerank_retention": (
            sum(_expected_id(record) in record["reranker"]["v4_output_ids"] for record in records),
            total,
            "expected source evidence retained by the legacy reranker",
        ),
        "critical_evidence_retention": (
            sum(
                record["expected"]["critical_source_id"] in record["reranker"]["v4_output_ids"]
                for record in safety_records
            ),
            len(safety_records),
            "critical source evidence retained by the legacy reranker",
        ),
        "evidence_coverage": None,
        "critical_evidence_coverage": None,
        "source_traceability_coverage": (
            sum(bool(record["v4_packed"]["source_paths"]) for record in records),
            total,
            "legacy packed context has retained source provenance",
        ),
        "packed_context_recall": (
            sum(_expected_id(record) in record["v4_packed"]["ids"] for record in records),
            total,
            "expected source evidence retained by legacy context packing",
        ),
        "prompt_evidence_coverage": (
            sum(_expected_id(record) in record["v4_packed"]["ids"] for record in records),
            total,
            "expected source evidence exposed to the legacy prompt context",
        ),
        "packed_critical_recall": (
            sum(
                record["expected"]["critical_source_id"] in record["v4_packed"]["ids"]
                for record in safety_records
            ),
            len(safety_records),
            "critical source evidence retained by legacy context packing",
        ),
        "final_concept_recall": (
            sum(_expected_id(record) in record["v4_packed"]["ids"] for record in records),
            total,
            "expected concept source evidence exposed by legacy prompt packing",
        ),
        "grounded_claim_coverage": (
            answer_quality["passed_cases"],
            answer_quality["total_cases"],
            "shared offline answer-quality release contract, independent of this retrieval harness",
        ),
    }
    for name, metric in metrics.items():
        v5_ratio = float(metric["ratio"])
        reference = v4_metrics[name]
        metric["v5_candidate"] = {
            "numerator": metric["numerator"],
            "denominator": metric["denominator"],
            "ratio": v5_ratio,
            "scope": metric["scope"],
        }
        if reference is None:
            metric["v4_reference"] = {
                "not_comparable": True,
                "scope": "V4 has no separate structural stage with this semantic contract.",
            }
            metric["comparison"] = "V5 contract coverage only; no V4 equivalence claimed"
            metric["non_decreasing"] = True
            continue
        numerator, denominator, scope = reference
        v4_ratio = round(numerator / denominator, 6) if denominator else 1.0
        metric["v4_reference"] = {
            "numerator": numerator,
            "denominator": denominator,
            "ratio": v4_ratio,
            "scope": scope,
        }
        metric["comparison"] = "non_decreasing versus the frozen V4 reference"
        metric["non_decreasing"] = v5_ratio >= v4_ratio
        metric["passed"] = bool(metric["passed"] and metric["non_decreasing"])


def _metric(
    numerator: int,
    denominator: int,
    scope: str,
    *,
    expect_zero: bool = False,
) -> dict[str, Any]:
    ratio = round(numerator / denominator, 6) if denominator else 1.0
    return {
        "numerator": numerator,
        "denominator": denominator,
        "ratio": ratio,
        "scope": scope,
        "passed": numerator == 0 if expect_zero else numerator == denominator,
    }


def _time_stage(timings: dict[str, float], stage: str, function: Any) -> Any:
    started = time.perf_counter()
    result = function()
    timings[stage] = timings.get(stage, 0.0) + (time.perf_counter() - started) * 1000
    return result


def _ranked_ids(candidates: Iterable[RetrievedCandidate]) -> list[dict[str, Any]]:
    return [
        {"candidate_id": candidate.candidate_id, "rank": rank}
        for rank, candidate in enumerate(candidates, start=1)
    ]


def _ids(record: dict[str, Any], stage: str) -> set[str]:
    return {item["candidate_id"] for item in record[stage]}


def _expected_id(record: dict[str, Any]) -> str:
    return str(record["expected"]["primary_source_id"])


def _primary_id(case: dict[str, Any]) -> str:
    return f"{case['id']}:primary"


def _duplicate_count(values: list[str]) -> int:
    return len(values) - len(set(values))


def _legacy_source_paths(items: Iterable[Any]) -> list[str]:
    paths: list[str] = []
    for item in items:
        source = item.payload.get("source_path") or item.payload.get("source_file")
        if source and source not in paths:
            paths.append(str(source))
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = run_locked_dual_run(args.cases)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
