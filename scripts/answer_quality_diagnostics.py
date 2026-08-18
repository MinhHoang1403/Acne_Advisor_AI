"""Chạy chẩn đoán retrieval có evidence ID cho một tập case nhỏ.

Script này không phải benchmark lâm sàng và không đánh giá entailment của câu trả
lời. Nó chỉ ghi nhận một fact đã xác định có xuất hiện trong danh sách RRF và
packed context hay không. Khi dùng ``--live-retrieval``, script gọi embedding
query hiện hành và đọc Qdrant; nó không gọi generation, không ghi Redis và không
thay đổi dữ liệu Phase 1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

# ``python scripts/answer_quality_diagnostics.py`` sets ``scripts`` as the
# import root. Add the repository root before importing project modules.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.retrieval.context_packer import _render_block, pack_context
from src.retrieval.contracts import NormalizedQuery
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.service import (
    RRF_DENSE_WEIGHT,
    RRF_K,
    RRF_BM25_WEIGHT,
    EvidenceRetriever,
    _bounded_env,
    _to_candidate,
)


DEFAULT_CASES_PATH = REPOSITORY_ROOT / "tests" / "fixtures" / (
    "answer_quality_diagnostic_cases.json"
)
REQUIRED_CASE_FIELDS = {
    "case_id",
    "assessment_mode",
    "expected_entities",
    "evidence_groups",
    "required_facts",
    "acceptable_source_ids",
    "expected_behavior",
    "review_notes",
}


def load_diagnostic_cases(path: Path) -> list[dict[str, Any]]:
    """Load the checked-in, source-grounded diagnostic cases without mutation."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Diagnostic cases must be a JSON list.")

    case_ids: set[str] = set()
    for case in payload:
        if not isinstance(case, dict) or REQUIRED_CASE_FIELDS - case.keys():
            raise ValueError("A diagnostic case is missing required fields.")
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError("Diagnostic case IDs must be non-empty and unique.")
        case_ids.add(case_id)
        if case["assessment_mode"] == "retrieval_evidence":
            if not isinstance(case.get("question"), str) or not case["question"].strip():
                raise ValueError(f"Retrieval case {case_id} requires a question.")
            groups = case["evidence_groups"]
            if not isinstance(groups, list) or not groups:
                raise ValueError(f"Retrieval case {case_id} requires evidence groups.")
            if any(not isinstance(group.get("any_of"), list) or not group["any_of"] for group in groups):
                raise ValueError(f"Retrieval case {case_id} has an invalid evidence group.")
    return payload


def assess_retrieval_coverage(
    case: dict[str, Any],
    *,
    candidate_ids: list[str],
    packed_ids: list[str],
) -> dict[str, Any]:
    """Classify evidence presence only; it deliberately makes no truth verdict."""

    candidate_set = set(candidate_ids)
    packed_set = set(packed_ids)
    groups = list(case["evidence_groups"])
    group_results = [
        {
            "fact": group["fact"],
            "candidate_matches": sorted(candidate_set.intersection(group["any_of"])),
            "packed_matches": sorted(packed_set.intersection(group["any_of"])),
        }
        for group in groups
    ]
    candidate_complete = all(result["candidate_matches"] for result in group_results)
    packed_complete = all(result["packed_matches"] for result in group_results)
    if packed_complete:
        classification = "evidence_packed"
    elif candidate_complete:
        classification = "context_missing_required_fact"
    else:
        classification = "retrieval_miss"
    return {
        "case_id": case["case_id"],
        "classification": classification,
        "candidate_complete": candidate_complete,
        "packed_complete": packed_complete,
        "evidence_groups": group_results,
        "semantic_truth_checked": False,
    }


def _channel_trace(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render one read-only channel's rank and provider-native score."""

    return [
        {
            "candidate_id": str(result.get("id") or result.get("chunk_id") or ""),
            "rank": rank,
            "native_score": result.get("score"),
        }
        for rank, result in enumerate(results, start=1)
    ]


def _fused_candidate_trace(
    candidates: list[Any],
    packed: Any,
) -> list[dict[str, Any]]:
    """Expose packer observations without changing its runtime selection policy.

    This helper intentionally couples the diagnostic to the packer's existing
    private renderer so character costs match the prompt blocks exactly. It is
    diagnostic-only, not a retrieval API or a second selection implementation.
    """

    selected_ids = {item.item_id for item in packed.items}
    drop_reasons = {
        str(item.get("candidate_id") or ""): str(item.get("reason") or "")
        for item in packed.debug.get("dropped", [])
    }
    selected_count = 0
    rendered_chars = 0
    trace: list[dict[str, Any]] = []
    for candidate in candidates:
        selection_index = selected_count + 1
        separator_chars = 2 if selected_count else 0
        rendered_length = len(_render_block(candidate, selection_index))
        selected = candidate.candidate_id in selected_ids
        trace.append(
            {
                "candidate_id": candidate.candidate_id,
                "fused_rank": candidate.rank,
                "dense_rank": candidate.debug.get("dense_rank"),
                "bm25_rank": candidate.debug.get("bm25_rank"),
                "fused_score": candidate.fused_score,
                "source_id": candidate.payload.get("source_id"),
                "section": candidate.payload.get("header")
                or (candidate.payload.get("section_path") or [None])[-1],
                "text_chars": len(candidate.text),
                "rendered_chars": rendered_length,
                "cumulative_chars_before": rendered_chars,
                "remaining_chars_before": packed.debug["limits"]["max_chars"] - rendered_chars - separator_chars,
                "packed": selected,
                "drop_reason": drop_reasons.get(candidate.candidate_id),
            }
        )
        if selected:
            rendered_chars += separator_chars + rendered_length
            selected_count += 1
    return trace


async def collect_live_retrieval_observation(
    retriever: EvidenceRetriever,
    case: dict[str, Any],
) -> dict[str, Any]:
    """Observe the current retrieval primitives without altering their semantics.

    ``EvidenceRetriever`` intentionally exports packed context but not every
    intermediate RRF candidate. This diagnostic invokes its existing primitives
    with the same configured limits solely to record those IDs for localization.
    """

    query = " ".join(case["question"].split())
    candidate_limit = _bounded_env("RETRIEVAL_CANDIDATE_LIMIT", 16, 1, 50)
    context_items = min(8, _bounded_env("RETRIEVAL_CONTEXT_MAX_ITEMS", 8, 1, 20))
    context_chars = _bounded_env("RETRIEVAL_CONTEXT_MAX_CHARS", 6000, 512, 20000)
    dense_results, sparse_results = await asyncio.gather(
        retriever._dense_search(query, candidate_limit),
        retriever._vector_store.search_sparse(query, top_k=candidate_limit),
    )
    fused = reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        dense_weight=RRF_DENSE_WEIGHT,
        sparse_weight=RRF_BM25_WEIGHT,
        k=RRF_K,
    )
    candidates = [_to_candidate(item, rank) for rank, item in enumerate(fused, start=1)]
    packed = pack_context(
        NormalizedQuery(original_query=case["question"], normalized_text=query),
        candidates,
        max_items=context_items,
        max_chars=context_chars,
    )
    assessment = assess_retrieval_coverage(
        case,
        candidate_ids=[candidate.candidate_id for candidate in candidates],
        packed_ids=[item.item_id for item in packed.items],
    )
    return {
        **assessment,
        "channels": {"dense": len(dense_results), "bm25": len(sparse_results)},
        "channel_trace": {
            "dense": _channel_trace(dense_results),
            "bm25": _channel_trace(sparse_results),
        },
        "fused_candidate_count": len(candidates),
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "packed_ids": [item.item_id for item in packed.items],
        "fused_candidates": _fused_candidate_trace(candidates, packed),
        "packer": {
            "limits": packed.debug["limits"],
            "context_chars": len(packed.context_text),
            "selected_ids": packed.debug["selected_ids"],
        },
        "packer_drops": packed.debug.get("dropped", []),
    }


async def run_live_retrieval_diagnostic(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run only retrieval-evidence cases; conversation/safety remain explicit sentinels."""

    retriever = EvidenceRetriever()
    observations: list[dict[str, Any]] = []
    try:
        for case in cases:
            if case["assessment_mode"] == "source_scope_review":
                observations.append(
                    {
                        "case_id": case["case_id"],
                        "classification": "not_run",
                        "reason": "source_scope_review does not treat analogous evidence as direct retrieval support.",
                        "semantic_truth_checked": False,
                    }
                )
                continue
            if case["assessment_mode"] != "retrieval_evidence":
                observations.append(
                    {
                        "case_id": case["case_id"],
                        "classification": "not_run",
                        "reason": f"{case['assessment_mode']} requires its existing contract test or live provider review.",
                        "semantic_truth_checked": False,
                    }
                )
                continue
            try:
                observations.append(await collect_live_retrieval_observation(retriever, case))
            except Exception as exc:
                observations.append(
                    {
                        "case_id": case["case_id"],
                        "classification": "retrieval_runtime_error",
                        "error_type": type(exc).__name__,
                        "semantic_truth_checked": False,
                    }
                )
    finally:
        await retriever.close()
    return observations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only source-grounded retrieval diagnostic.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--live-retrieval",
        action="store_true",
        help="Call the configured query embedding provider and read Qdrant without generation or writes.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    cases = load_diagnostic_cases(args.cases)
    report: dict[str, Any] = {
        "diagnostic": "source_grounded_retrieval_presence",
        "semantic_truth_checked": False,
        "generation_provider_called": False,
        "datastore_writes": False,
        "case_count": len(cases),
        "cases": [],
    }
    if args.live_retrieval:
        report["query_embedding_provider_called"] = True
        report["cases"] = asyncio.run(run_live_retrieval_diagnostic(cases))
    else:
        report["query_embedding_provider_called"] = False
        report["cases"] = [
            {"case_id": case["case_id"], "classification": "not_run"} for case in cases
        ]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
