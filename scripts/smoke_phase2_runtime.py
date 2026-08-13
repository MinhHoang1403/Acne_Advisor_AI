#!/usr/bin/env python3
"""Phase 2 runtime smoke tests.

Default offline mode uses only deterministic local retrieval/quality helpers.
It does not call chat generation, LLMs, embedding APIs, or external services.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.entity_cards import build_entity_cards_from_taxonomy  # noqa: E402
from src.knowledge.entity_index import build_entity_point_payload  # noqa: E402
from src.quality.answer_verifier import verify_answer_quality  # noqa: E402
from src.retrieval.candidate_merge import merge_candidates  # noqa: E402
from src.retrieval.context_packer import pack_context  # noqa: E402
from src.retrieval.entity_retriever import retrieve_entity_candidates_from_payloads  # noqa: E402
from src.retrieval.metadata_boost import boost_chunk_results  # noqa: E402
from src.retrieval.query_expansion import expand_normalized_query  # noqa: E402
from src.retrieval.query_normalization import normalize_query  # noqa: E402
from src.retrieval.reranker import rerank_candidates  # noqa: E402

SMOKE_CASES = [
    {
        "id": "dalacin_t_identity",
        "query": "Dalacin T là gì?",
        "answer": "Dalacin T chứa clindamycin, thuộc nhóm kháng sinh bôi/topical antibiotic. Không nên dùng kháng sinh bôi đơn độc kéo dài.",
    },
    {
        "id": "epiduo_bpo",
        "query": "Epiduo có BPO không?",
        "answer": "Có. Epiduo chứa adapalene và benzoyl peroxide (BPO). Benzoyl peroxide không phải kháng sinh.",
    },
    {
        "id": "bp_not_antibiotic",
        "query": "Benzoyl peroxide có phải kháng sinh không?",
        "answer": "Không, benzoyl peroxide không phải kháng sinh. Đây là hoạt chất bôi có tác dụng kháng khuẩn.",
    },
    {
        "id": "differin_class",
        "query": "Differin thuộc nhóm gì?",
        "answer": "Differin chứa adapalene và thuộc nhóm topical retinoid/retinoid bôi.",
    },
    {
        "id": "clindamycin_not_retinoid",
        "query": "Clindamycin có phải retinoid không?",
        "answer": "Không, clindamycin không phải retinoid. Đây là kháng sinh bôi/topical antibiotic.",
    },
    {
        "id": "adapalene_not_antibiotic",
        "query": "Adapalene có phải kháng sinh không?",
        "answer": "Không, adapalene không phải kháng sinh. Adapalene là topical retinoid/retinoid bôi.",
    },
    {
        "id": "blackheads",
        "query": "Mụn đầu đen là gì?",
        "answer": "Mụn đầu đen là dạng comedone/nhân mụn mở, liên quan bít tắc nang lông và oxy hóa chất bã.",
    },
    {
        "id": "inflammatory_acne",
        "query": "Mụn viêm nên xử lý thế nào?",
        "answer": "Mụn viêm là tổn thương mụn có đỏ, đau hoặc mủ. Nên chăm sóc dịu nhẹ và gặp bác sĩ nếu viêm nặng, đau hoặc để sẹo.",
    },
]


def run_offline_smoke() -> dict[str, Any]:
    entity_payloads = [
        build_entity_point_payload(card, kb_version="acne_kb_v1")
        for card in build_entity_cards_from_taxonomy()
    ]
    cases: list[dict[str, Any]] = []
    errors: list[str] = []

    for case in SMOKE_CASES:
        try:
            normalized = normalize_query(case["query"])
            expansion = expand_normalized_query(normalized)
            entity_candidates = retrieve_entity_candidates_from_payloads(
                normalized_query=normalized,
                expansion=expansion,
                payloads=entity_payloads,
                collection_name="acne_entities",
                limit=8,
            )
            chunk_candidates = boost_chunk_results(
                [_synthetic_chunk(case["id"], normalized)],
                normalized,
                collection_name="acne_knowledge",
            )
            merged = merge_candidates(entity_candidates, chunk_candidates, normalized, limit=8)
            reranked, rerank_trace = rerank_candidates(normalized, merged, expansion, top_n=8)
            packed = pack_context(normalized, reranked, max_items=5)
            report = verify_answer_quality(
                query=case["query"],
                answer=case["answer"],
                normalized_query=normalized,
                packed_context=packed,
            )
            case_passed = bool(reranked and packed.items and report.passed)
            if not case_passed:
                errors.append(case["id"])
            cases.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "passed": case_passed,
                    "intent": normalized.intent,
                    "reranked_count": len(reranked),
                    "rerank_provider": rerank_trace.provider,
                    "packed_items": len(packed.items),
                    "answer_quality_passed": report.passed,
                    "issue_codes": [issue.code for issue in report.issues],
                }
            )
        except Exception as exc:
            errors.append(f"{case['id']}: {exc}")
            cases.append({"id": case["id"], "query": case["query"], "passed": False, "error": str(exc)})

    return {
        "passed": not errors,
        "mode": "offline",
        "cases": cases,
        "errors": errors,
    }


async def run_live_chat_smoke() -> dict[str, Any]:
    print(
        "WARNING: --live-chat may call LLM providers according to runtime configuration.",
        file=sys.stderr,
    )
    from src.agent.graph import run_clinical_agent

    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    for case in SMOKE_CASES[:3]:
        try:
            result = await run_clinical_agent(case["query"], bypass_cache=True)
            quality = result.get("answer_quality_report") or {}
            passed = bool(result.get("answer")) and quality.get("passed") is not False
            if not passed:
                errors.append(case["id"])
            cases.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "passed": passed,
                    "answer_quality_passed": quality.get("passed"),
                    "sources": result.get("sources", []),
                }
            )
        except Exception as exc:
            errors.append(f"{case['id']}: {exc}")
            cases.append({"id": case["id"], "query": case["query"], "passed": False, "error": str(exc)})
    return {"passed": not errors, "mode": "live-chat", "cases": cases, "errors": errors}


def _synthetic_chunk(case_id: str, normalized: Any) -> dict[str, Any]:
    return {
        "id": f"{case_id}:smoke_chunk",
        "chunk_id": f"{case_id}:smoke_chunk",
        "text": " ".join(
            [
                normalized.original_query,
                *normalized.drug_product,
                *normalized.active_ingredient,
                *normalized.drug_class,
                *normalized.condition,
                " ".join(normalized.metadata.get("concern", [])),
                " ".join(normalized.metadata.get("content_type", [])),
            ]
        ),
        "score": 0.1,
        "source_file": "smoke_phase2_runtime.py",
        "drug_product": normalized.drug_product,
        "active_ingredient": normalized.active_ingredient,
        "drug_class": normalized.drug_class,
        "condition": normalized.condition,
        "query_intent_hint": [normalized.intent],
        "concern": normalized.metadata.get("concern", []),
        "content_type": normalized.metadata.get("content_type", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 runtime smoke test.")
    parser.add_argument("--mode", choices=["offline", "live-chat"], default="offline")
    parser.add_argument("--live-chat", action="store_true", help="Run live chat smoke; may call configured LLM.")
    args = parser.parse_args()

    if args.live_chat:
        report = asyncio.run(run_live_chat_smoke())
    elif args.mode == "live-chat":
        report = asyncio.run(run_live_chat_smoke())
    else:
        report = run_offline_smoke()

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
