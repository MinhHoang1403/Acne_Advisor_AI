"""Tạo và lưu manifest mô tả một knowledge build đã được validation.

Manifest gom source hashes, contract IDs, collection names, counts và structural
hash để operator kiểm tra compatibility. File được ghi qua temporary path rồi
replace; module không tự build, activate hoặc mutate datastore.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.ingestion.bm25 import BM25_CONTRACT_ID, bm25_config
from src.ingestion.build import (
    BUILD_MANIFEST_SCHEMA,
    ENTITY_CARD_CONTRACT_ID,
    GRAPH_CONTRACT_ID,
    CompiledKnowledge,
)
from src.ingestion.chunking import CHUNK_CONTRACT_ID
from src.ingestion.embedding import (
    EMBEDDING_CONTRACT_ID,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_DISTANCE,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
)
from src.ingestion.filtering import FILTER_CONTRACT_ID
from src.ingestion.normalization import NORMALIZATION_CONTRACT_ID
from src.ingestion.parser import PARSER_CONFIGURATION, PARSER_CONTRACT_ID, ParsedArtifact
from src.ingestion.provenance import PROVENANCE_CONTRACT_ID
from src.ingestion.source_manifest import CanonicalSource


def build_manifest(
    compiled: CompiledKnowledge,
    sources: tuple[CanonicalSource, ...],
    artifacts: dict[str, ParsedArtifact],
    *,
    knowledge_collection: str,
    entity_collection: str,
    entity_count: int,
    graph_nodes: int,
    graph_relationships: int,
    git_commit: str,
    cache_stats: dict[str, Any],
    validation: dict[str, Any],
    phase1_frozen: bool = False,
) -> dict[str, Any]:
    """Tổng hợp identity, contracts, counts và validation thành manifest."""

    return {
        "schema": BUILD_MANIFEST_SCHEMA,
        "status": "completed" if validation.get("passed") else "failed",
        "phase1_frozen": phase1_frozen,
        "build_id": compiled.identity.build_id,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "source_manifest_hash": compiled.identity.source_manifest_hash,
        "taxonomy_hash": compiled.identity.taxonomy_hash,
        "contract_hash": compiled.identity.contract_hash,
        "sources": [
            {
                "source_id": source.source_id,
                "content_hash": source.sha256,
                "parsed_output_hash": artifacts[source.source_id].parsed_output_hash,
                "normalized_output_hash": artifacts[source.source_id].normalized_output_hash,
                "chunk_count": compiled.source_counts[source.source_id],
                "filtered_artifact_count": compiled.filtered_counts[source.source_id],
            }
            for source in sources
        ],
        "contracts": {
            "parser": {"id": PARSER_CONTRACT_ID, "configuration": PARSER_CONFIGURATION},
            "normalization": NORMALIZATION_CONTRACT_ID,
            "chunk": CHUNK_CONTRACT_ID,
            "filter": FILTER_CONTRACT_ID,
            "provenance": PROVENANCE_CONTRACT_ID,
            "embedding": {
                "id": EMBEDDING_CONTRACT_ID,
                "provider": EMBEDDING_PROVIDER,
                "model": EMBEDDING_MODEL,
                "dimensions": EMBEDDING_DIMENSIONS,
                "distance": EMBEDDING_DISTANCE,
                "task_type": None,
            },
            "bm25": {
                "id": BM25_CONTRACT_ID,
                "implementation": "qdrant_native",
                "field": "bm25",
                "modifier": "idf",
                "configuration": bm25_config().model_dump(mode="json"),
            },
            "entity_card": ENTITY_CARD_CONTRACT_ID,
            "graph": GRAPH_CONTRACT_ID,
            "semantic_enrichment": "removed",
        },
        "collections": {
            "knowledge_logical": "acne_knowledge",
            "knowledge_physical": knowledge_collection,
            "entity_logical": "acne_entities",
            "entity_physical": entity_collection,
        },
        "counts": {
            "sources": len(sources),
            "knowledge_chunks": len(compiled.records),
            "entities": entity_count,
            "graph_nodes": graph_nodes,
            "graph_relationships": graph_relationships,
        },
        "structural_hash": compiled.structural_hash,
        "cache": cache_stats,
        "validation": validation,
    }


def save_build_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Ghi manifest atomically qua file tạm trong cùng thư mục."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_build_manifest(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != BUILD_MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported knowledge-build manifest at {path}")
    return raw


__all__ = ["build_manifest", "load_build_manifest", "save_build_manifest"]
