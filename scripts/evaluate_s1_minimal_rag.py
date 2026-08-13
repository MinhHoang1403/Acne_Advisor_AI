#!/usr/bin/env python3
"""Compare CURRENT_SYSTEM with MINIMAL_RAG_V1 on frozen evaluation inputs."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_retrieval_v5_release import _run_case as run_current_locked_case  # noqa: E402
from src.evaluation.minimal_rag import (  # noqa: E402
    CURRENT_SYSTEM_ID,
    MINIMAL_RAG_SYSTEM_ID,
    MinimalRagService,
)

R8_CASES = PROJECT_ROOT / "tests" / "golden" / "retrieval_v5_release_cases.json"
P45_CASES = PROJECT_ROOT / "tests" / "golden" / "p45_production_shadow_questions.json"
DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    category: str
    language: str
    critical: bool
    positive_evidence_ids: tuple[str, ...]
    dataset: str
    current_fixture: dict[str, Any]


@dataclass(frozen=True)
class SystemRunResult:
    system_id: str
    case_id: str
    evidence_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    answer: str
    citations: tuple[str, ...]
    latency_ms: dict[str, float]
    call_counts: dict[str, int]
    status: str
    error: str | None = None


class StaticVectorStore:
    """Read-only channel observations matching the locked CURRENT_SYSTEM fixture."""

    def __init__(self, case: EvaluationCase) -> None:
        self._case = case

    async def search(self, _query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        return self._candidates("dense")[:top_k]

    async def search_sparse(self, _text: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self._candidates("sparse")[:top_k]

    def _candidates(self, channel: str) -> list[dict[str, Any]]:
        primary = _static_candidate(self._case, "primary", 0.92 if channel == "dense" else 0.35)
        support = _static_candidate(self._case, "support", 0.58 if channel == "dense" else 0.78)
        return [primary, support] if channel == "dense" else [support, primary]


def load_frozen_cases() -> list[EvaluationCase]:
    """Load R8 gold cases plus frozen P4.5 coverage-only questions."""

    r8 = json.loads(R8_CASES.read_text(encoding="utf-8"))
    p45 = json.loads(P45_CASES.read_text(encoding="utf-8"))["questions"]
    cases: list[EvaluationCase] = []
    for item in r8:
        case_id = str(item["id"])
        fixture = dict(item)
        cases.append(
            EvaluationCase(
                case_id=case_id,
                question=str(item["query"]),
                category=_r8_category(item),
                language="VI",
                critical=bool(item.get("safety_flags")),
                positive_evidence_ids=(f"{case_id}:primary",),
                dataset="retrieval_v5_release_cases",
                current_fixture=fixture,
            )
        )
    for item in p45:
        safety = str(item.get("safety_category") or "NONE")
        fixture = {
            "id": item["id"],
            "query": item["question"],
            "concept": str(item["category"]).lower(),
            "intent": "evaluation_coverage",
            "safety_flags": [] if safety == "NONE" else [safety.lower()],
        }
        cases.append(
            EvaluationCase(
                case_id=str(item["id"]),
                question=str(item["question"]),
                category=str(item["category"]),
                language=str(item["language"]),
                critical=bool(item["critical_expected"]),
                positive_evidence_ids=(),
                dataset="p45_production_shadow_questions",
                current_fixture=fixture,
            )
        )
    return cases


async def run_provider_free_current(case: EvaluationCase) -> SystemRunResult:
    started = time.perf_counter()
    record, timings = await asyncio.to_thread(run_current_locked_case, case.current_fixture)
    evidence_ids = tuple(record["v5_prompt_context"]["ids"])
    source_ids = tuple(record["v5_prompt_context"]["source_paths"])
    stage_timings = {key: round(float(value), 3) for key, value in timings.items()}
    stage_timings["total"] = round((time.perf_counter() - started) * 1000, 3)
    return SystemRunResult(
        system_id=CURRENT_SYSTEM_ID,
        case_id=case.case_id,
        evidence_ids=evidence_ids,
        source_ids=source_ids,
        answer="",
        citations=(),
        latency_ms=stage_timings,
        call_counts=_calls(reranker=1),
        status="retrieved",
    )


async def run_provider_free_minimal(case: EvaluationCase) -> SystemRunResult:
    service = MinimalRagService(
        vector_store=StaticVectorStore(case),
        embedder=_provider_free_embedder,
    )
    result = await service.run(case.question, generate_answer=False)
    return SystemRunResult(
        system_id=MINIMAL_RAG_SYSTEM_ID,
        case_id=case.case_id,
        evidence_ids=tuple(item.evidence_id for item in result.evidence),
        source_ids=tuple(item.source_id for item in result.evidence),
        answer=result.answer,
        citations=result.citations,
        latency_ms=result.latency_ms,
        call_counts=_calls(),
        status=result.status,
        error=result.error,
    )


async def run_live_current(case: EvaluationCase, provider: str | None, model: str | None) -> SystemRunResult:
    from src.agent.graph import run_clinical_agent

    started = time.perf_counter()
    result = await run_clinical_agent(
        case.question,
        llm_provider=provider,
        llm_model=model,
        allow_model_fallback=False,
        bypass_cache=True,
        evaluation_mode=True,
    )
    contexts = result.get("vector_contexts") or []
    evidence_ids = tuple(
        str(item.get("chunk_id") or item.get("id") or "") for item in contexts if isinstance(item, dict)
    )
    source_ids = tuple(
        str(item.get("source_identity") or item.get("source_path") or item.get("source_file") or "")
        for item in contexts
        if isinstance(item, dict)
    )
    return SystemRunResult(
        system_id=CURRENT_SYSTEM_ID,
        case_id=case.case_id,
        evidence_ids=tuple(value for value in evidence_ids if value),
        source_ids=tuple(value for value in source_ids if value),
        answer=str(result.get("answer") or ""),
        citations=tuple(str(value) for value in (result.get("sources") or [])),
        latency_ms={"total": round((time.perf_counter() - started) * 1000, 3)},
        call_counts={"not_instrumented": 1},
        status=str(result.get("retrieval_status") or "completed"),
    )


async def run_live_minimal(case: EvaluationCase, provider: str | None, model: str | None) -> SystemRunResult:
    result = await MinimalRagService(provider=provider, model=model).run(case.question)
    return SystemRunResult(
        system_id=MINIMAL_RAG_SYSTEM_ID,
        case_id=case.case_id,
        evidence_ids=tuple(item.evidence_id for item in result.evidence),
        source_ids=tuple(item.source_id for item in result.evidence),
        answer=result.answer,
        citations=result.citations,
        latency_ms=result.latency_ms,
        call_counts=result.call_counts,
        status=result.status,
        error=result.error,
    )


async def run_side_by_side(
    cases: list[EvaluationCase],
    current_runner: Callable[[EvaluationCase], Awaitable[SystemRunResult]],
    minimal_runner: Callable[[EvaluationCase], Awaitable[SystemRunResult]],
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[list[SystemRunResult], list[SystemRunResult]]:
    current: list[SystemRunResult] = []
    minimal: list[SystemRunResult] = []
    for case in cases:
        current.append(await _isolated_run(CURRENT_SYSTEM_ID, case, current_runner, timeout_seconds))
        minimal.append(await _isolated_run(MINIMAL_RAG_SYSTEM_ID, case, minimal_runner, timeout_seconds))
    return current, minimal


def retrieval_metrics(cases: list[EvaluationCase], results: list[SystemRunResult]) -> dict[str, Any]:
    by_id = {result.case_id: result for result in results}
    labeled = [case for case in cases if case.positive_evidence_ids]
    if not labeled:
        return {name: "N/A" for name in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr", "mean_positive_rank")}
    ranks: list[int | None] = []
    critical_hits = 0
    critical_total = 0
    misses: list[str] = []
    for case in labeled:
        result = by_id[case.case_id]
        rank = _positive_rank(result.evidence_ids, case.positive_evidence_ids)
        ranks.append(rank)
        if rank is None:
            misses.append(case.case_id)
        if case.critical:
            critical_total += 1
            critical_hits += int(rank is not None and rank <= 5)
    metrics: dict[str, Any] = {
        f"recall@{depth}": _ratio(sum(rank is not None and rank <= depth for rank in ranks), len(labeled))
        for depth in (1, 3, 5, 10)
    }
    found = [rank for rank in ranks if rank is not None]
    metrics.update(
        {
            "mrr": _ratio_value(sum(1.0 / rank for rank in found), len(labeled)),
            "mean_positive_rank": round(statistics.mean(found), 6) if found else "N/A",
            "mean_positive_rank_denominator": len(found),
            "labeled_case_denominator": len(labeled),
            "critical_source_coverage@5": _ratio(critical_hits, critical_total),
            "retrieval_miss_count": len(misses),
            "retrieval_miss_case_ids": misses,
        }
    )
    return metrics


def source_metrics(results: list[SystemRunResult]) -> dict[str, Any]:
    slots = sum(len(result.evidence_ids) for result in results)
    valid_slots = sum(
        min(len(result.evidence_ids), len(result.source_ids)) for result in results
    )
    unique_evidence = {value for result in results for value in result.evidence_ids}
    unique_sources = {value for result in results for value in result.source_ids if value}
    duplicate_slots = slots - sum(len(set(result.evidence_ids)) for result in results)
    return {
        "evidence_slots": slots,
        "provenance_valid": _ratio(valid_slots, slots),
        "unique_documents": len(unique_evidence),
        "unique_sources": len(unique_sources),
        "duplicate_chunk_pressure": _ratio(duplicate_slots, slots),
    }


def latency_metrics(results: list[SystemRunResult]) -> dict[str, Any]:
    totals = [result.latency_ms.get("total", 0.0) for result in results]
    return {
        "case_count": len(totals),
        "total_ms": round(sum(totals), 3),
        "mean_ms": round(statistics.mean(totals), 3) if totals else 0.0,
        "p95_ms": round(_percentile(totals, 0.95), 3) if totals else 0.0,
    }


def aggregate_call_counts(results: list[SystemRunResult]) -> dict[str, int]:
    keys = set().union(*(result.call_counts for result in results)) if results else set()
    return {key: sum(result.call_counts.get(key, 0) for result in results) for key in sorted(keys)}


def write_outputs(
    output_dir: Path,
    cases: list[EvaluationCase],
    current: list[SystemRunResult],
    minimal: list[SystemRunResult],
    *,
    mode: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "provider-free":
        current_metrics = retrieval_metrics(cases, current)
        minimal_metrics = retrieval_metrics(cases, minimal)
        gold_scope = "R8 locked expected primary source IDs"
    else:
        current_metrics = _live_retrieval_na()
        minimal_metrics = _live_retrieval_na()
        gold_scope = "N/A: locked fixture IDs do not identify live Qdrant points"
    summary = {
        "schema_version": "s1_side_by_side_v1",
        "mode": mode,
        "case_count": len(cases),
        "objective_gold_scope": gold_scope,
        "p45_answer_quality_metrics": "N/A: zero completed human labels",
        "current": {
            "retrieval": current_metrics,
            "sources": source_metrics(current),
            "latency": latency_metrics(current),
        },
        "minimal": {
            "retrieval": minimal_metrics,
            "sources": source_metrics(minimal),
            "latency": latency_metrics(minimal),
        },
    }
    files = {
        "s1_cases.json": [asdict(case) for case in cases],
        "s1_current_results.json": [asdict(result) for result in current],
        "s1_minimal_results.json": [asdict(result) for result in minimal],
        "s1_retrieval_metrics.json": summary,
        "s1_call_counts.json": {
            "mode": mode,
            "current": aggregate_call_counts(current),
            "minimal": aggregate_call_counts(minimal),
            "note": (
                "Provider-free mode counts real external calls as zero; reranker is a local logical stage."
                if mode == "provider-free"
                else "CURRENT_SYSTEM live calls are not instrumented; MINIMAL_RAG_V1 reports service-level calls."
            ),
        },
        "s1_complexity_metrics.json": _complexity_metrics(),
    }
    for name, payload in files.items():
        (output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


async def _isolated_run(
    system_id: str,
    case: EvaluationCase,
    runner: Callable[[EvaluationCase], Awaitable[SystemRunResult]],
    timeout_seconds: float,
) -> SystemRunResult:
    started = time.perf_counter()
    try:
        return await asyncio.wait_for(runner(case), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        error = "TimeoutError"
        status = "timeout"
    except Exception as exc:
        error = exc.__class__.__name__
        status = "failed"
    return SystemRunResult(
        system_id=system_id,
        case_id=case.case_id,
        evidence_ids=(),
        source_ids=(),
        answer="",
        citations=(),
        latency_ms={"total": round((time.perf_counter() - started) * 1000, 3)},
        call_counts=_calls(),
        status=status,
        error=error,
    )


def _static_candidate(case: EvaluationCase, kind: str, score: float) -> dict[str, Any]:
    candidate_id = f"{case.case_id}:{kind}"
    source_path = f"locked-fixture/{case.case_id}/{kind}.md"
    return {
        "id": candidate_id,
        "chunk_id": candidate_id,
        "document_id": f"locked-doc:{case.case_id}:{kind}",
        "source_path": source_path,
        "source_identity": source_path,
        "text": f"Locked source evidence ({kind}) for: {case.question}",
        "score": score,
    }


async def _provider_free_embedder(_query: str) -> list[float]:
    return [0.0]


def _r8_category(item: dict[str, Any]) -> str:
    if item.get("safety_flags"):
        return str(item["safety_flags"][0]).upper()
    mapping = {
        "drug_identity": "ENTITY_BRAND",
        "ingredient_question": "ENTITY_BRAND",
        "class_check": "TREATMENT",
        "acne_type": "SIMPLE_KNOWLEDGE",
        "treatment": "TREATMENT",
        "source_question": "SOURCE_PROVENANCE",
        "general_acne_question": "MECHANISM",
    }
    return mapping.get(str(item.get("intent")), "SIMPLE_KNOWLEDGE")


def _positive_rank(evidence_ids: tuple[str, ...], positives: tuple[str, ...]) -> int | None:
    positive_set = set(positives)
    return next((rank for rank, value in enumerate(evidence_ids, start=1) if value in positive_set), None)


def _ratio(numerator: int, denominator: int) -> dict[str, Any] | str:
    if not denominator:
        return "N/A"
    return {"numerator": numerator, "denominator": denominator, "ratio": round(numerator / denominator, 6)}


def _ratio_value(numerator: float, denominator: int) -> float | str:
    return round(numerator / denominator, 6) if denominator else "N/A"


def _live_retrieval_na() -> dict[str, str]:
    reason = "N/A: no live corpus point-ID gold contract"
    return {
        "recall@1": reason,
        "recall@3": reason,
        "recall@5": reason,
        "recall@10": reason,
        "mrr": reason,
        "mean_positive_rank": reason,
        "critical_source_coverage@5": reason,
        "retrieval_miss_count": reason,
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile)))
    return ordered[index]


def _calls(**updates: int) -> dict[str, int]:
    result = {
        "embedding": 0,
        "qdrant": 0,
        "neo4j": 0,
        "redis": 0,
        "reranker": 0,
        "generation": 0,
    }
    result.update(updates)
    return result


def _complexity_metrics() -> dict[str, Any]:
    minimal_path = PROJECT_ROOT / "src" / "evaluation" / "minimal_rag.py"
    harness_path = Path(__file__)
    return {
        "counting_method": "non-empty non-comment physical lines in directly owned files",
        "current_system": {
            "python_files": 105,
            "python_loc": 27303,
            "state_fields": 97,
            "meaningful_rule_families": 60,
            "datastores": 4,
            "source": "S0 measured baseline",
        },
        "minimal_rag_v1": {
            "implementation_files": 2,
            "implementation_loc": _logical_loc(minimal_path) + _logical_loc(PROJECT_ROOT / "src" / "retrieval" / "rrf.py"),
            "harness_files": 1,
            "harness_loc": _logical_loc(harness_path),
            "state_fields": 12,
            "major_stages": 5,
            "semantic_heuristics": 0,
            "ranking_heuristics_beyond_rrf_fixed_top_k": 0,
            "datastores": 1,
            "required_environment_references": 4,
        },
    }


def _logical_loc(path: Path) -> int:
    return sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("provider-free", "live"), default="provider-free")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    return parser.parse_args()


async def _main_async(args: argparse.Namespace) -> int:
    cases = load_frozen_cases()
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]
    if args.mode == "provider-free":
        current_runner = run_provider_free_current
        minimal_runner = run_provider_free_minimal
    else:
        current_runner = lambda case: run_live_current(case, args.provider, args.model)
        minimal_runner = lambda case: run_live_minimal(case, args.provider, args.model)
    current, minimal = await run_side_by_side(
        cases,
        current_runner,
        minimal_runner,
        timeout_seconds=args.timeout_seconds,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or PROJECT_ROOT / "reports" / "evaluation" / f"{timestamp}_s1_minimal_rag_baseline"
    summary = write_outputs(output_dir, cases, current, minimal, mode=args.mode)
    print(json.dumps({"output_dir": str(output_dir), **summary}, ensure_ascii=False, indent=2))
    return int(any(result.status in {"failed", "timeout"} for result in [*current, *minimal]))


def main() -> int:
    return asyncio.run(_main_async(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
