"""Evaluation-only knowledge-loss classification for System V4 P1."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from src.retrieval.diagnostics import KnowledgeLossStage

from .models import RETRIEVAL_DIAGNOSTICS_VERSION

ConceptMatcher = Callable[[str, str], bool]
_CRITICAL_CATEGORIES = {"urgent_emergency", "pregnancy_lactation", "antibiotic_stewardship"}


def build_case_retrieval_diagnostics(
    *,
    result: dict[str, Any],
    case: dict[str, Any],
    concept_matcher: ConceptMatcher,
) -> dict[str, Any]:
    """Classify concept loss from evaluation-only retrieval trace metadata."""

    trace = result.get("retrieval_diagnostics")
    trace_data = trace if isinstance(trace, dict) else {}
    concept_by_name = {
        str(item.get("concept") or ""): item
        for item in trace_data.get("concept_traces") or []
        if isinstance(item, dict) and item.get("concept")
    }
    expected_concepts = _dedupe(case.get("expected_concepts") or [])
    critical = bool(case.get("critical_case")) or str(case.get("category")) in _CRITICAL_CATEGORIES
    prompt_trace = trace_data.get("prompt_evidence_trace")
    prompt_presence = (
        prompt_trace.get("concept_presence")
        if isinstance(prompt_trace, dict) and isinstance(prompt_trace.get("concept_presence"), dict)
        else {}
    )
    query_trace = trace_data.get("query_trace") if isinstance(trace_data.get("query_trace"), dict) else {}
    answer = str(result.get("answer") or "")
    verifier = result.get("answer_quality_report") if isinstance(result.get("answer_quality_report"), dict) else {}
    concepts = [
        _concept_diagnostic(
            concept=concept,
            runtime_trace=concept_by_name.get(concept),
            critical=critical,
            answer=answer,
            query_trace=query_trace,
            prompt_present=prompt_presence.get(concept),
            expected_sources=list(case.get("accepted_sources") or []),
            raw_candidate_trace=list(trace_data.get("raw_candidate_trace") or []),
            verifier=verifier,
            label_issue_hint=bool(case.get("diagnostic_label_issue")),
            concept_matcher=concept_matcher,
        )
        for concept in expected_concepts
    ]
    loss_counts = Counter(str(item["loss_stage"]) for item in concepts)
    return {
        "version": RETRIEVAL_DIAGNOSTICS_VERSION,
        "available": bool(trace_data),
        "query_trace": query_trace,
        "candidate_trace_summary": {
            "raw_candidate_count": len(trace_data.get("raw_candidate_trace") or []),
            "channels": _channel_counts(trace_data.get("raw_candidate_trace") or []),
        },
        "fusion_trace": trace_data.get("fusion_trace") or {},
        "rerank_trace": trace_data.get("rerank_trace") or {},
        "pack_trace": trace_data.get("pack_trace") or {},
        "prompt_evidence_trace": prompt_trace or {},
        "concept_diagnostics": concepts,
        "loss_stage_summary": {
            "total_expected_concepts": len(concepts),
            "concepts_found_in_final_answer": sum(bool(item["final_answer"]["present"]) for item in concepts),
            "missing_concepts": sum(not bool(item["final_answer"]["present"]) for item in concepts),
            "loss_stage_counts": _loss_stage_counts(loss_counts),
        },
    }


def summarize_retrieval_diagnostics(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate loss stages and retention metrics without mixing denominators."""

    rows = list(results)
    concept_rows: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        diagnostics = row.get("retrieval_diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        case_id = str(row.get("case_id") or "")
        for concept in diagnostics.get("concept_diagnostics") or []:
            if isinstance(concept, dict):
                concept_rows.append((case_id, concept))

    stage_counts = Counter(str(item.get("loss_stage") or KnowledgeLossStage.UNKNOWN.value) for _, item in concept_rows)
    cases_by_stage: dict[str, set[str]] = defaultdict(set)
    for case_id, item in concept_rows:
        cases_by_stage[str(item.get("loss_stage") or KnowledgeLossStage.UNKNOWN.value)].add(case_id)

    candidate_rows = [item for _, item in concept_rows if _present(item, "retrieval", "candidate")]
    fusion_rows = [item for _, item in concept_rows if _present(item, "fusion")]
    rerank_rows = [item for _, item in concept_rows if _present(item, "rerank")]
    packed_rows = [item for _, item in concept_rows if _known(item, "packed_context")]
    prompt_rows = [item for _, item in concept_rows if _known(item, "prompt_evidence")]
    source_rows = [item for _, item in concept_rows if item.get("source_candidate_present") is not None]
    critical_rows = [item for _, item in concept_rows if bool(item.get("critical"))]
    return {
        "version": RETRIEVAL_DIAGNOSTICS_VERSION,
        "available_case_count": sum(
            bool((row.get("retrieval_diagnostics") or {}).get("available"))
            for row in rows
            if isinstance(row.get("retrieval_diagnostics"), dict)
        ),
        "total_expected_concepts": len(concept_rows),
        "concepts_found_in_final_answer": sum(
            bool(item.get("final_answer", {}).get("present")) for _, item in concept_rows
        ),
        "missing_concepts": sum(
            not bool(item.get("final_answer", {}).get("present")) for _, item in concept_rows
        ),
        "loss_stage_counts": _loss_stage_counts(stage_counts),
        "loss_stage_percentages": _loss_stage_percentages(stage_counts, len(concept_rows)),
        "loss_stage_cases_affected": {
            stage.value: len(cases_by_stage[stage.value]) for stage in KnowledgeLossStage
        },
        "by_category": _by_category(rows),
        "critical_concepts": {
            "total": len(critical_rows),
            "candidate_retained": sum(_present(item, "retrieval", "candidate") for item in critical_rows),
            "rerank_retained": sum(_present(item, "rerank") for item in critical_rows),
            "packed_retained": sum(_present(item, "packed_context") for item in critical_rows),
            "answer_present": sum(bool(item.get("final_answer", {}).get("present")) for item in critical_rows),
        },
        "metrics": {
            "concept_recall": _rate(
                sum(bool(item.get("final_answer", {}).get("present")) for _, item in concept_rows),
                len(concept_rows),
            ),
            "source_candidate_recall": _rate(
                sum(bool(item.get("source_candidate_present")) for item in source_rows), len(source_rows)
            ),
            "fusion_retention": _rate(sum(_present(item, "fusion") for item in candidate_rows), len(candidate_rows)),
            "rerank_retention": _rate(sum(_present(item, "rerank") for item in fusion_rows), len(fusion_rows)),
            "packed_context_recall": _rate(sum(_present(item, "packed_context") for item in packed_rows), len(packed_rows)),
            "prompt_evidence_coverage": _rate(sum(_present(item, "prompt_evidence") for item in prompt_rows), len(prompt_rows)),
            "critical_concept_recall": _rate(
                sum(bool(item.get("final_answer", {}).get("present")) for item in critical_rows),
                len(critical_rows),
            ),
        },
    }


def write_retrieval_diagnostic_report(
    path: Path,
    summary: dict[str, Any],
    results: Iterable[dict[str, Any]],
) -> Path:
    """Write a concise developer-facing report without prompt or chunk dumps."""

    metrics = summary.get("metrics") or {}
    lines = [
        "# Retrieval Knowledge-Loss Diagnostics",
        "",
        f"- Expected concepts: {summary.get('total_expected_concepts', 0)}",
        f"- Found in final answer: {summary.get('concepts_found_in_final_answer', 0)}",
        f"- Missing concepts: {summary.get('missing_concepts', 0)}",
        "",
        "## Metrics",
    ]
    for name in (
        "source_candidate_recall",
        "fusion_retention",
        "rerank_retention",
        "packed_context_recall",
        "prompt_evidence_coverage",
        "concept_recall",
        "critical_concept_recall",
    ):
        metric = metrics.get(name) or {}
        value = metric.get("value")
        display = "N/A" if value is None else f"{value}% ({metric.get('numerator')}/{metric.get('denominator')})"
        lines.append(f"- {name}: {display}")
    lines.extend(["", "## Loss Stages"])
    for stage, count in (summary.get("loss_stage_counts") or {}).items():
        lines.append(f"- {stage}: {count}")
    lines.extend(["", "## Missing Concepts"])
    failures = _missing_concepts_for_report(results)
    if not failures:
        lines.append("- None observed in this diagnostic run.")
    for item in failures[:40]:
        lines.extend(
            [
                f"### Case `{item['case_id']}` - `{item['concept']}`",
                f"- Candidate: {_yes_no(item['candidate'])}; fusion: {_yes_no(item['fusion'])}; rerank: {_yes_no(item['rerank'])}",
                f"- Packed: {_yes_no(item['packed'])}; prompt: {_yes_no(item['prompt'])}; answer: {_yes_no(item['answer'])}",
                f"- Loss stage: `{item['loss_stage']}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _concept_diagnostic(
    *,
    concept: str,
    runtime_trace: dict[str, Any] | None,
    critical: bool,
    answer: str,
    query_trace: dict[str, Any],
    prompt_present: Any,
    expected_sources: list[str],
    raw_candidate_trace: list[Any],
    verifier: dict[str, Any],
    label_issue_hint: bool,
    concept_matcher: ConceptMatcher,
) -> dict[str, Any]:
    trace = runtime_trace or {}
    retrieval = trace.get("retrieval") if isinstance(trace.get("retrieval"), dict) else {}
    answer_present = concept_matcher(answer, concept)
    verifier_expected, verifier_detected = _verifier_state(verifier, concept, concept_matcher)
    source_candidate_present = _source_candidate_present(expected_sources, raw_candidate_trace)
    prompt = _stage(trace.get("prompt_evidence"))
    if prompt_present is not None:
        prompt = {"present": bool(prompt_present), "rank": None, "candidate_ids": []}
    diagnostic = {
        "concept": concept,
        "critical": critical,
        "source_present": trace.get("source_present"),
        "parsed_present": trace.get("parsed_present"),
        "chunk_present": trace.get("chunk_present"),
        "indexed_present": trace.get("indexed_present"),
        "source_candidate_present": source_candidate_present,
        "retrieval": {
            "dense": _stage(retrieval.get("dense")),
            "sparse": _stage(retrieval.get("sparse")),
            "entity": _stage(retrieval.get("entity")),
            "graph": _stage(retrieval.get("graph")),
            "candidate": _stage(retrieval.get("candidate")),
        },
        "fusion": _stage(trace.get("fusion")),
        "rerank": _stage(trace.get("rerank")),
        "packed_context": _stage(trace.get("packed_context")),
        "prompt_evidence": prompt,
        "final_answer": {"present": answer_present},
        "verifier_detected": verifier_detected,
    }
    diagnostic["loss_stage"] = _classify_loss_stage(
        diagnostic=diagnostic,
        query_trace=query_trace,
        verifier_expected=verifier_expected,
        label_issue_hint=label_issue_hint,
        concept_matcher=concept_matcher,
    ).value
    return diagnostic


def _classify_loss_stage(
    *,
    diagnostic: dict[str, Any],
    query_trace: dict[str, Any],
    verifier_expected: bool,
    label_issue_hint: bool,
    concept_matcher: ConceptMatcher,
) -> KnowledgeLossStage:
    if diagnostic["final_answer"]["present"]:
        return KnowledgeLossStage.NOT_APPLICABLE
    if label_issue_hint:
        return KnowledgeLossStage.EVALUATION_LABEL_ISSUE
    if diagnostic["source_present"] is False:
        return KnowledgeLossStage.SOURCE_MISS
    if diagnostic["parsed_present"] is False:
        return KnowledgeLossStage.PARSING_MISS
    if diagnostic["chunk_present"] is False:
        return KnowledgeLossStage.CHUNKING_MISS
    if diagnostic["indexed_present"] is False:
        return KnowledgeLossStage.INDEXING_MISS
    if not query_trace:
        return KnowledgeLossStage.UNKNOWN
    if (
        diagnostic["critical"]
        and verifier_expected
        and not diagnostic["verifier_detected"]
        and diagnostic["prompt_evidence"]["present"]
    ):
        return KnowledgeLossStage.VERIFIER_MISS
    if diagnostic["prompt_evidence"]["present"]:
        return KnowledgeLossStage.GENERATION_MISS
    if diagnostic["packed_context"]["present"]:
        return KnowledgeLossStage.CONTEXT_PACKING_MISS
    if diagnostic["rerank"]["present"]:
        return KnowledgeLossStage.CONTEXT_PACKING_MISS
    if diagnostic["fusion"]["present"]:
        return KnowledgeLossStage.RERANK_MISS
    if diagnostic["retrieval"]["candidate"]["present"]:
        return KnowledgeLossStage.FUSION_MISS
    if _query_dropped_concept(query_trace, diagnostic["concept"], concept_matcher):
        return KnowledgeLossStage.QUERY_UNDERSTANDING_MISS
    return KnowledgeLossStage.RETRIEVAL_MISS


def _query_dropped_concept(query_trace: dict[str, Any], concept: str, matcher: ConceptMatcher) -> bool:
    original = str(query_trace.get("original_query") or "")
    retrieval = str(query_trace.get("retrieval_query") or "")
    normalized = str(query_trace.get("normalized_query") or "")
    aliases = " ".join(str(value) for value in query_trace.get("expanded_aliases") or [])
    return matcher(original, concept) and not any(matcher(value, concept) for value in (retrieval, normalized, aliases))


def _source_candidate_present(expected_sources: list[str], candidates: list[Any]) -> bool | None:
    if not expected_sources:
        return None
    expected = {_source_key(source) for source in expected_sources}
    observed = {_source_key(item.get("source_name")) for item in candidates if isinstance(item, dict) and item.get("source_name")}
    return bool(expected & observed)


def _source_key(value: Any) -> str:
    return str(value or "").replace("\\", "/").split("/")[-1].casefold()


def _verifier_state(verifier: dict[str, Any], concept: str, matcher: ConceptMatcher) -> tuple[bool, bool]:
    required = " ".join(str(value) for value in verifier.get("required_facts") or [])
    missing = " ".join(str(value) for value in verifier.get("missing_facts") or [])
    issue_evidence = " ".join(
        str((issue.get("evidence") or {}).get("fact") or "")
        for issue in verifier.get("issues") or []
        if isinstance(issue, dict)
    )
    expected = matcher(required, concept)
    detected = matcher(missing, concept) or matcher(issue_evidence, concept)
    return expected, detected


def _stage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"present": None, "rank": None, "candidate_ids": []}
    return {
        "present": value.get("present"),
        "rank": value.get("rank"),
        "candidate_ids": list(value.get("candidate_ids") or []),
    }


def _present(item: dict[str, Any], *path: str) -> bool:
    value: Any = item
    for key in path:
        if not isinstance(value, dict):
            return False
        value = value.get(key)
    return bool(value.get("present")) if isinstance(value, dict) else False


def _known(item: dict[str, Any], *path: str) -> bool:
    value: Any = item
    for key in path:
        if not isinstance(value, dict):
            return False
        value = value.get(key)
    return isinstance(value, dict) and value.get("present") is not None


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": round(100 * numerator / denominator, 2) if denominator else None,
    }


def _loss_stage_counts(counts: Counter[str]) -> dict[str, int]:
    return {stage.value: int(counts[stage.value]) for stage in KnowledgeLossStage}


def _loss_stage_percentages(counts: Counter[str], total: int) -> dict[str, float | None]:
    return {
        stage.value: round(100 * counts[stage.value] / total, 2) if total else None
        for stage in KnowledgeLossStage
    }


def _channel_counts(records: Iterable[Any]) -> dict[str, int]:
    counts = Counter(
        str(item.get("channel") or "unknown") for item in records if isinstance(item, dict)
    )
    return dict(sorted(counts.items()))


def _by_category(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        category = str(row.get("category") or "unknown")
        diagnostics = row.get("retrieval_diagnostics") or {}
        for item in diagnostics.get("concept_diagnostics") or []:
            if isinstance(item, dict):
                grouped[category][str(item.get("loss_stage") or KnowledgeLossStage.UNKNOWN.value)] += 1
    return {
        category: _loss_stage_counts(counts)
        for category, counts in sorted(grouped.items())
    }


def _missing_concepts_for_report(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in results:
        diagnostics = row.get("retrieval_diagnostics") or {}
        for item in diagnostics.get("concept_diagnostics") or []:
            if not isinstance(item, dict) or item.get("final_answer", {}).get("present"):
                continue
            output.append(
                {
                    "case_id": str(row.get("case_id") or ""),
                    "concept": str(item.get("concept") or ""),
                    "candidate": _present(item, "retrieval", "candidate"),
                    "fusion": _present(item, "fusion"),
                    "rerank": _present(item, "rerank"),
                    "packed": _present(item, "packed_context"),
                    "prompt": _present(item, "prompt_evidence"),
                    "answer": bool(item.get("final_answer", {}).get("present")),
                    "loss_stage": str(item.get("loss_stage") or KnowledgeLossStage.UNKNOWN.value),
                }
            )
    return output


def _yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def _dedupe(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item.casefold() in seen:
            continue
        seen.add(item.casefold())
        output.append(item)
    return output


__all__ = [
    "RETRIEVAL_DIAGNOSTICS_VERSION",
    "build_case_retrieval_diagnostics",
    "summarize_retrieval_diagnostics",
    "write_retrieval_diagnostic_report",
]
