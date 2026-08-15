"""Kiểm tra từng lớp của compiled knowledge và indexed datastores.

Các check xác nhận identity, schema, counts, provenance presence và một BM25
smoke query. Chúng là integrity checks của build, không đo retrieval quality và
không xác minh clinical truth. Chỉ ``validate_qdrant_collection`` gọi Qdrant;
các validator còn lại làm việc trên records trong bộ nhớ.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from src.ingestion.bm25 import BM25_VECTOR_NAME, bm25_document
from src.ingestion.build import CompiledKnowledge
from src.ingestion.chunking import CHUNK_MAX_CHARS
from src.ingestion.embedding import EMBEDDING_DIMENSIONS
from src.ingestion.provenance import validate_provenance
from src.ingestion.source_manifest import CanonicalSource
from src.knowledge.schemas import EntityCard


def validate_compiled_knowledge(
    compiled: CompiledKnowledge,
    sources: tuple[CanonicalSource, ...],
) -> dict[str, Any]:
    """Kiểm source coverage, identity, size cap và provenance của compiled records."""

    errors: list[str] = []
    warnings: list[str] = []
    source_ids = {source.source_id for source in sources}
    record_sources = Counter(record.get("source_id") for record in compiled.records)
    if set(record_sources) != source_ids:
        errors.append("compiled source coverage does not equal canonical manifest")
    identifiers = [record["chunk_id"] for record in compiled.records]
    if len(identifiers) != len(set(identifiers)):
        errors.append("duplicate chunk_id")
    for index, record in enumerate(compiled.records):
        issues = validate_provenance(record)
        if issues:
            errors.append(f"chunk {index} provenance: {','.join(issues)}")
        if len(record["text"]) > CHUNK_MAX_CHARS:
            errors.append(f"chunk {index} exceeds character cap")
        if record.get("build_id") != compiled.identity.build_id:
            errors.append(f"chunk {index} build ID mismatch")
        if ":\\" in str(record.get("source_path")):
            errors.append(f"chunk {index} has absolute Windows source path")
    return {
        "layer": "compiled_knowledge",
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts_by_source": dict(sorted(record_sources.items())),
        "provenance_completeness": (
            1.0 if compiled.records and not any("provenance" in error for error in errors) else 0.0
        ),
    }


def validate_entity_cards(cards: list[EntityCard], canonical_source_ids: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    identities: set[str] = set()
    for card in cards:
        identity = card.stable_id()
        if identity in identities:
            errors.append(f"duplicate EntityCard identity: {identity}")
        identities.add(identity)
        if not card.source_ids:
            errors.append(f"EntityCard missing source IDs: {card.canonical_name}")
        unknown = sorted(set(card.source_ids) - canonical_source_ids)
        if unknown:
            errors.append(f"EntityCard unknown source IDs {card.canonical_name}: {unknown}")
    return {"layer": "entity_cards", "passed": not errors, "errors": errors, "count": len(cards)}


def validate_graph_records(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    errors: list[str] = []
    node_keys = {(node["label"], node["canonical_name"]) for node in records.get("nodes", [])}
    if len(node_keys) != len(records.get("nodes", [])):
        errors.append("duplicate graph node identity")
    edge_keys: set[tuple[str, str, str, str, str]] = set()
    for edge in records.get("relationships", []):
        key = (
            edge["source_label"], edge["source_name"], edge["relationship"],
            edge["target_label"], edge["target_name"],
        )
        if key in edge_keys:
            errors.append(f"duplicate graph relationship: {key}")
        edge_keys.add(key)
        if (edge["source_label"], edge["source_name"]) not in node_keys:
            errors.append(f"dangling graph source: {key}")
        if (edge["target_label"], edge["target_name"]) not in node_keys:
            errors.append(f"dangling graph target: {key}")
        if not edge.get("properties", {}).get("source_ids"):
            errors.append(f"graph relationship missing provenance: {key}")
    return {
        "layer": "graph_records",
        "passed": not errors,
        "errors": errors,
        "nodes": len(node_keys),
        "relationships": len(edge_keys),
    }


async def validate_qdrant_collection(
    client: AsyncQdrantClient,
    *,
    collection_name: str,
    expected_points: int,
    smoke_query: str = "benzoyl peroxide",
) -> dict[str, Any]:
    """Đối chiếu Qdrant schema/count và chạy một BM25 integrity query."""

    errors: list[str] = []
    info = await client.get_collection(collection_name)
    params = info.config.params
    dense = params.vectors.get("dense") if isinstance(params.vectors, dict) else None
    sparse = params.sparse_vectors.get(BM25_VECTOR_NAME) if params.sparse_vectors else None
    if dense is None or dense.size != EMBEDDING_DIMENSIONS or dense.distance != models.Distance.COSINE:
        errors.append("dense schema mismatch")
    if sparse is None or sparse.modifier != models.Modifier.IDF:
        errors.append("BM25 sparse schema missing collection IDF")
    if int(info.points_count or 0) != expected_points:
        errors.append(f"point count mismatch: expected {expected_points}, got {info.points_count}")
    response = await client.query_points(
        collection_name=collection_name,
        query=bm25_document(smoke_query),
        using=BM25_VECTOR_NAME,
        limit=3,
        with_payload=True,
    )
    if not response.points:
        errors.append("BM25 smoke query returned no points")
    return {
        "layer": "qdrant",
        "passed": not errors,
        "errors": errors,
        "points": int(info.points_count or 0),
        "bm25_smoke_ids": [str(point.id) for point in response.points],
    }


def combine_validation_layers(*layers: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": all(layer.get("passed") is True for layer in layers),
        "layers": list(layers),
        "errors": [error for layer in layers for error in layer.get("errors", [])],
    }


__all__ = [
    "combine_validation_layers",
    "validate_compiled_knowledge",
    "validate_entity_cards",
    "validate_graph_records",
    "validate_qdrant_collection",
]
