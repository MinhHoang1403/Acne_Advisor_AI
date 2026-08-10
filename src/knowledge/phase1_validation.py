"""Strict, reusable validation primitives for a complete Phase 1 build."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any


CHUNK_REQUIRED_PAYLOAD_FIELDS = (
    "chunk_id",
    "document_id",
    "source_path",
    "content_hash",
    "chunk_index",
    "ingestion_run_id",
    "ingested_at",
    "text",
    "embedding_provider",
    "embedding_model",
    "embedding_dimensions",
    "kb_version",
)
ENTITY_REQUIRED_PAYLOAD_FIELDS = (
    "entity_id",
    "canonical_name",
    "entity_type",
    "taxonomy_version",
    "entity_schema_version",
    "point_id",
    "text",
    "embedding_provider",
    "embedding_model",
    "embedding_dimensions",
    "kb_version",
)


@dataclass
class CollectionReconciliation:
    role: str
    collection: str
    expected_count: int
    actual_count: int = 0
    expected_by_source: dict[str, int] = field(default_factory=dict)
    actual_by_source: dict[str, int] = field(default_factory=dict)
    schema: dict[str, Any] = field(default_factory=dict)
    payload_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["passed"] = self.passed
        return data


@dataclass
class Phase1ValidationReport:
    sources_expected: int
    sources_accounted_for: int
    knowledge: CollectionReconciliation | None = None
    entities: CollectionReconciliation | None = None
    graph: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        children_passed = all(
            child is None or child.passed
            for child in (self.knowledge, self.entities)
        )
        graph_passed = self.graph is not None and bool(self.graph.get("passed"))
        return (
            not self.errors
            and self.sources_expected == self.sources_accounted_for
            and children_passed
            and graph_passed
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sources_expected": self.sources_expected,
            "sources_accounted_for": self.sources_accounted_for,
            "knowledge": self.knowledge.as_dict() if self.knowledge else None,
            "entities": self.entities.as_dict() if self.entities else None,
            "graph": self.graph,
            "errors": self.errors,
            "warnings": self.warnings,
            "passed": self.passed,
        }


def inspect_named_vector_schema(params: Any) -> dict[str, Any]:
    if isinstance(params, dict):
        vectors_config = params.get("vectors")
        sparse_vectors_config = params.get("sparse_vectors")
    else:
        vectors_config = getattr(params, "vectors", None)
        sparse_vectors_config = getattr(params, "sparse_vectors", None)

    dense = _named_config(vectors_config, "dense")
    bm25 = _named_config(sparse_vectors_config, "bm25")
    dense_size = dense.get("size") if isinstance(dense, dict) else getattr(dense, "size", None)
    return {
        "has_dense": dense is not None,
        "dense_vector_name": "dense" if dense is not None else None,
        "dense_size": dense_size,
        "has_bm25": bm25 is not None,
        "sparse_vector_name": "bm25" if bm25 is not None else None,
    }


async def reconcile_qdrant_collection(
    *,
    client: Any,
    collection_name: str,
    role: str,
    expected_count: int,
    expected_dimensions: int,
    expected_by_source: dict[str, int] | None = None,
    expected_entity_ids: set[str] | None = None,
) -> CollectionReconciliation:
    """Verify count, named-vector schema, payloads, and stable identities."""

    result = CollectionReconciliation(
        role=role,
        collection=collection_name,
        expected_count=expected_count,
        expected_by_source=dict(expected_by_source or {}),
    )
    info = await client.get_collection(collection_name=collection_name)
    result.actual_count = int(getattr(info, "points_count", 0) or 0)
    result.schema = inspect_named_vector_schema(info.config.params)

    if result.actual_count != expected_count:
        result.errors.append(
            f"{role} collection count mismatch: expected {expected_count}, got {result.actual_count}"
        )
    if not result.schema["has_dense"] or result.schema["dense_size"] != expected_dimensions:
        result.errors.append(
            f"{role} dense schema mismatch: expected dense/{expected_dimensions}, "
            f"got {result.schema}"
        )
    if not result.schema["has_bm25"]:
        result.errors.append(f"{role} collection is missing sparse vector 'bm25'")

    required_fields = (
        CHUNK_REQUIRED_PAYLOAD_FIELDS if role == "knowledge" else ENTITY_REQUIRED_PAYLOAD_FIELDS
    )
    payloads = await _scroll_payloads(client, collection_name)
    result.payload_count = len(payloads)
    if len(payloads) != result.actual_count:
        result.errors.append(
            f"{role} payload scroll mismatch: points_count={result.actual_count}, "
            f"payloads={len(payloads)}"
        )

    actual_by_source: Counter[str] = Counter()
    actual_entity_ids: list[str] = []
    for index, payload in enumerate(payloads):
        missing = [field for field in required_fields if payload.get(field) in (None, "")]
        if missing:
            result.errors.append(
                f"{role} payload {index} missing required fields: {', '.join(missing)}"
            )
        if role == "knowledge":
            source_path = str(payload.get("source_path") or "")
            if source_path:
                actual_by_source[source_path] += 1
        else:
            entity_id = str(payload.get("entity_id") or "")
            if entity_id:
                actual_entity_ids.append(entity_id)

    result.actual_by_source = dict(sorted(actual_by_source.items()))
    if result.expected_by_source and result.actual_by_source != result.expected_by_source:
        result.errors.append(
            "knowledge per-source count mismatch: "
            f"expected {result.expected_by_source}, got {result.actual_by_source}"
        )

    if expected_entity_ids is not None:
        actual_set = set(actual_entity_ids)
        if len(actual_set) != len(actual_entity_ids):
            result.errors.append("entity collection contains duplicate entity_id payload values")
        if actual_set != expected_entity_ids:
            result.errors.append(
                "entity identity mismatch: "
                f"expected {sorted(expected_entity_ids)}, got {sorted(actual_set)}"
            )

    return result


def manifest_knowledge_expectations(
    manifest: dict[str, Any],
    *,
    required_source_identities: set[str],
) -> tuple[int, dict[str, int], list[str]]:
    """Build strict expected knowledge counts from per-source manifest entries."""

    errors: list[str] = []
    records = [
        record
        for record in (manifest.get("documents") or {}).values()
        if isinstance(record, dict)
    ]
    identity_counts = Counter(str(record.get("source_identity") or "") for record in records)
    for identity in sorted(required_source_identities):
        if identity_counts[identity] != 1:
            errors.append(
                f"manifest source identity {identity!r} expected once, found {identity_counts[identity]}"
            )
    unexpected_identities = sorted(
        identity
        for identity in identity_counts
        if identity and identity not in required_source_identities
    )
    if unexpected_identities:
        errors.append(
            "manifest contains sources outside the configured full-build source set: "
            + ", ".join(unexpected_identities)
        )

    expected_by_source: Counter[str] = Counter()
    for record in records:
        status = str(record.get("status") or "")
        if status not in {"knowledge_indexed_pending_phase1_validation", "completed"}:
            errors.append(
                f"manifest source {record.get('source_path')!r} is not knowledge-complete: {status}"
            )
            continue
        source_path = str(record.get("source_path") or "")
        point_count = int(record.get("qdrant_point_count", 0) or 0)
        if not source_path or point_count <= 0:
            errors.append(f"manifest source {source_path!r} has no eligible Qdrant points")
            continue
        expected_by_source[source_path] += point_count

    return sum(expected_by_source.values()), dict(sorted(expected_by_source.items())), errors


async def _scroll_payloads(client: Any, collection_name: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    offset: Any = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        payloads.extend(dict(point.payload or {}) for point in points)
        if offset is None:
            return payloads


def _named_config(config: Any, name: str) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    if hasattr(config, "get"):
        return config.get(name)
    return None


__all__ = [
    "CHUNK_REQUIRED_PAYLOAD_FIELDS",
    "ENTITY_REQUIRED_PAYLOAD_FIELDS",
    "CollectionReconciliation",
    "Phase1ValidationReport",
    "inspect_named_vector_schema",
    "manifest_knowledge_expectations",
    "reconcile_qdrant_collection",
]
