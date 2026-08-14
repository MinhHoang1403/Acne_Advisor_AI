#!/usr/bin/env python3
"""Provider-free structural smoke for the frozen runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.graph import clinical_graph  # noqa: E402
from src.agent.state import ClinicalState  # noqa: E402
from src.retrieval.context_packer import pack_context  # noqa: E402
from src.retrieval.contracts import NormalizedQuery, RetrievedCandidate  # noqa: E402
from src.retrieval.rrf import reciprocal_rank_fusion  # noqa: E402

SMOKE_QUERIES = (
    "Benzoyl peroxide có phải kháng sinh không?",
    "Adapalene thuộc nhóm gì?",
    "Mụn đầu đen là gì?",
    "Mụn viêm nên xử lý thế nào?",
    "Có nên dùng clindamycin đơn độc không?",
    "Retinoid có dùng khi mang thai không?",
    "Adapalene và benzoyl peroxide khác nhau thế nào?",
    "Khi nào cần gặp bác sĩ da liễu?",
)


def run_offline_smoke() -> dict:
    cases: list[dict] = []
    for index, query in enumerate(SMOKE_QUERIES, 1):
        point_id = f"fixture-{index}"
        dense = [{"id": point_id, "score": 0.8, "text": f"Evidence for {query}", "source_id": "fixture"}]
        bm25 = [{"id": point_id, "score": 5.0, "text": f"Evidence for {query}", "source_id": "fixture"}]
        fused = reciprocal_rank_fusion(dense, bm25)
        candidates = [
            RetrievedCandidate(
                candidate_id=point_id,
                source="chunk",
                collection="acne_knowledge",
                text=fused[0]["text"],
                score=fused[0]["rrf_score"],
                fused_score=fused[0]["rrf_score"],
                rank=1,
                payload={"chunk_id": point_id, "source_id": "fixture"},
            )
        ]
        normalized = NormalizedQuery(
            original_query=query,
            normalized_text=query,
            intent="medical_question",
        )
        packed = pack_context(normalized, candidates, max_items=4, max_chars=2000)
        passed = (
            fused[0]["dense_rank"] == 1
            and fused[0]["sparse_rank"] == 1
            and [item.item_id for item in packed.items] == [point_id]
            and packed.items[0].payload["source_id"] == "fixture"
        )
        cases.append({"id": point_id, "query": query, "passed": passed})

    graph_nodes = set(clinical_graph.get_graph().nodes) - {"__start__", "__end__"}
    errors = []
    if len(graph_nodes) != 8:
        errors.append(f"Expected 8 semantic graph nodes, found {len(graph_nodes)}")
    if len(ClinicalState.__annotations__) >= 97:
        errors.append("ClinicalState exceeds the frozen 68-field contract")
    if not all(case["passed"] for case in cases):
        errors.append("One or more structural retrieval cases failed")
    return {
        "mode": "offline",
        "passed": not errors,
        "architecture": "langgraph_dense_bm25_rrf",
        "graph_nodes": sorted(graph_nodes),
        "state_fields": len(ClinicalState.__annotations__),
        "cases": cases,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("offline",), default="offline")
    parser.parse_args()
    report = run_offline_smoke()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
