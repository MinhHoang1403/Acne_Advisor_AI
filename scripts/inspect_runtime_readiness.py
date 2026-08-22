#!/usr/bin/env python3
"""Đọc manifest/services để mô tả knowledge build và runtime capabilities.

Các check Qdrant/Neo4j chỉ đọc count/schema. Capability flags phản ánh cấu hình
thực tế: normal retrieval là Dense + BM25 + RRF và local reranker tùy chọn;
entity/graph không tham gia runtime answer grounding.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=False)

from src.agent.graph import clinical_graph  # noqa: E402
from src.agent.state import ClinicalState  # noqa: E402
from src.ingestion.manifest import validate_build_id  # noqa: E402
from src.observability.versioning import build_pipeline_version_manifest  # noqa: E402


def inspect_readiness() -> dict[str, Any]:
    knowledge = _knowledge_manifest_check()
    qdrant = _qdrant_check()
    neo4j = _neo4j_check()
    architecture = _architecture_check()
    checks = [knowledge, qdrant, neo4j, architecture]
    manifest = build_pipeline_version_manifest()
    return {
        "passed": all(check["passed"] for check in checks),
        "runtime_config": {
            "chunk_collection": os.getenv("QDRANT_COLLECTION_NAME", "acne_knowledge"),
            "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
            "embedding": {
                "embedding_model": os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2"),
                "embedding_dimensions": int(os.getenv("EMBEDDING_DIMENSIONS", "3072")),
            },
            "cache_answer_version": manifest["answer_cache_version"],
        },
        "checks": checks,
        "current_capabilities": {
            "langgraph_orchestrator": True,
            "bounded_agent_decision": True,
            "qdrant_dense_search": True,
            "qdrant_sparse_bm25_search": True,
            "rrf_fusion": True,
            "bounded_provenance_packing": True,
            "bounded_evidence_retry": True,
            "explicit_abstention": True,
            "entity_runtime_retrieval": False,
            "graph_runtime_retrieval": False,
            "reranker": bool((manifest.get("reranker") or {}).get("enabled")),
            "candidate_policy": False,
            "evidence_selector": False,
            "claim_shadow_verifier": False,
        },
        "pipeline_manifest": manifest,
    }


def _knowledge_manifest_check() -> dict[str, Any]:
    path = PROJECT_ROOT / "data" / "knowledge_build_manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        counts = data.get("counts") or {}
        configured_build = validate_build_id(os.getenv("KB_VERSION"))
        details = {
            "build_id": data.get("build_id"),
            "configured_kb_version": configured_build,
            "phase1_frozen": data.get("phase1_frozen"),
            "status": data.get("status"),
            "counts": counts,
        }
        passed = (
            data.get("build_id") == configured_build
            and data.get("phase1_frozen") is True
            and data.get("status") == "activated"
            and counts.get("sources") == 4
            and counts.get("knowledge_chunks") == 512
            and counts.get("entities") == 32
            and counts.get("graph_nodes") == 32
            and counts.get("graph_relationships") == 27
        )
        return {"name": "frozen_phase1_manifest", "passed": passed, "details": details}
    except ValueError as exc:
        return {
            "name": "frozen_phase1_manifest",
            "passed": False,
            "details": {"error": str(exc)},
        }
    except Exception as exc:
        return {"name": "frozen_phase1_manifest", "passed": False, "details": {"error": exc.__class__.__name__}}


def _qdrant_check() -> dict[str, Any]:
    collection = os.getenv("QDRANT_COLLECTION_NAME", "acne_knowledge")
    url = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
    request = urllib.request.Request(f"{url}/collections/{collection}", method="GET")
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if api_key:
        request.add_header("api-key", api_key)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))["result"]
        config = payload.get("config") or {}
        params = config.get("params") or {}
        vectors = params.get("vectors") or {}
        sparse = params.get("sparse_vectors") or {}
        dense = vectors.get("dense") or {}
        details = {
            "collection": collection,
            "points_count": payload.get("points_count"),
            "dense_size": dense.get("size"),
            "dense_distance": dense.get("distance"),
            "sparse_vectors": sorted(sparse),
        }
        passed = (
            payload.get("points_count") == 512
            and dense.get("size") == 3072
            and str(dense.get("distance") or "").casefold() == "cosine"
            and sorted(sparse) == ["bm25"]
        )
        return {"name": "qdrant_frozen_knowledge", "passed": passed, "details": details}
    except Exception as exc:
        return {"name": "qdrant_frozen_knowledge", "passed": False, "details": {"error": exc.__class__.__name__}}


def _neo4j_check() -> dict[str, Any]:
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
            auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "password")),
        )
        with driver:
            with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
                record = session.run(
                    "MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() RETURN nodes, count(r) AS relationships"
                ).single()
        details = {"nodes": record["nodes"], "relationships": record["relationships"]}
        return {
            "name": "neo4j_frozen_graph",
            "passed": details == {"nodes": 32, "relationships": 27},
            "details": details,
        }
    except Exception as exc:
        return {"name": "neo4j_frozen_graph", "passed": False, "details": {"error": exc.__class__.__name__}}


def _architecture_check() -> dict[str, Any]:
    nodes = set(clinical_graph.get_graph().nodes) - {"__start__", "__end__"}
    removed_paths = (
        "src/retrieval/candidate_policy.py",
        "src/retrieval/evidence_selector.py",
        "src/retrieval/v5_contracts.py",
        "src/quality/claim_grounding.py",
    )
    details = {
        "nodes": sorted(nodes),
        "node_count": len(nodes),
        "state_fields": len(ClinicalState.__annotations__),
        "removed_paths_present": [path for path in removed_paths if (PROJECT_ROOT / path).exists()],
    }
    return {
        "name": "minimal_agentic_rag_architecture",
        "passed": len(nodes) == 8 and len(ClinicalState.__annotations__) < 97 and not details["removed_paths_present"],
        "details": details,
    }


def main() -> int:
    report = inspect_readiness()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
