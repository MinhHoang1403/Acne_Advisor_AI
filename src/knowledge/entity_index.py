"""Qdrant entity-card index helpers for the canonical ``acne_entities`` alias."""

from __future__ import annotations

import inspect
import os
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from src.database.vector_store import (
    QDRANT_COLLECTION_NAME,
    qdrant_client_kwargs,
)
from src.ingestion.bm25 import BM25_VECTOR_NAME, bm25_document, bm25_sparse_vector_config
from src.knowledge.entity_cards import entity_card_to_text
from src.knowledge.schemas import EntityCard
from src.knowledge.taxonomy_models import normalize_taxonomy_alias
from src.knowledge.versioning import (
    get_embedding_metadata,
    get_knowledge_versions,
)


ENTITY_COLLECTION_DEFAULT = "acne_entities"
CHUNK_COLLECTION_DEFAULT = os.getenv("QDRANT_COLLECTION_NAME", "acne_knowledge")

EmbeddingProvider = Callable[[list[str]], list[list[float]] | Awaitable[list[list[float]]]]


def get_entity_collection_name() -> str:
    return os.getenv("ENTITY_QDRANT_COLLECTION_NAME", ENTITY_COLLECTION_DEFAULT)


def get_chunk_collection_name() -> str:
    qdrant_collection = os.getenv("QDRANT_COLLECTION_NAME", CHUNK_COLLECTION_DEFAULT).strip()
    qdrant_collection = qdrant_collection or CHUNK_COLLECTION_DEFAULT
    configured = os.getenv("CHUNK_QDRANT_COLLECTION_NAME", "").strip()
    return configured or qdrant_collection


def entity_identity_key(card_or_payload: EntityCard | dict[str, Any]) -> str:
    """Return the version-independent canonical identity for an entity card."""

    if isinstance(card_or_payload, EntityCard):
        entity_type = card_or_payload.entity_type
        metadata = card_or_payload.metadata
        canonical_name = card_or_payload.canonical_name
    else:
        entity_type = str(card_or_payload.get("entity_type") or "").strip()
        metadata = card_or_payload.get("metadata") if isinstance(card_or_payload.get("metadata"), dict) else {}
        canonical_name = str(card_or_payload.get("canonical_name") or "").strip()

    taxonomy_key = ""
    if isinstance(metadata, dict):
        taxonomy_key = str(metadata.get("taxonomy_key") or "").strip()
    identity_source = taxonomy_key or canonical_name
    normalized = normalize_taxonomy_alias(identity_source).replace(" ", "_")
    if not entity_type or not normalized:
        raise ValueError("Entity identity requires entity_type and taxonomy_key/canonical_name.")
    return f"{entity_type}:{normalized}"


def entity_point_id(card: EntityCard, kb_version: str = "frozen_phase1_build") -> str:
    """Return a deterministic Qdrant-compatible UUID for a new entity identity.

    ``kb_version`` is accepted for backward-compatible callers but does not
    participate in canonical identity.
    """

    _ = kb_version
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"acne_entity:{entity_identity_key(card)}"))


def build_entity_point_payload(
    card: EntityCard,
    kb_version: str = "frozen_phase1_build",
    *,
    point_id: str | None = None,
) -> dict[str, Any]:
    """Build the Qdrant payload for an entity card."""

    identity = entity_identity_key(card)
    payload = card.to_payload()
    payload.update(
        {
            "text": entity_card_to_text(card),
            "kb_version": kb_version,
            **get_embedding_metadata(),
            **get_knowledge_versions(),
            "entity_id": identity,
            "canonical_identity": identity,
            "point_id": point_id or entity_point_id(card, kb_version=kb_version),
        }
    )
    payload["kb_version"] = kb_version
    payload["taxonomy_version"] = card.taxonomy_version
    payload["entity_schema_version"] = card.entity_schema_version
    return payload


async def ensure_entity_collection(
    client: Any | None = None,
    collection_name: str | None = None,
    *,
    recreate: bool = False,
    embedding_dimensions: int | None = None,
) -> None:
    """Create or validate the entity Qdrant collection.

    Existing collections are never deleted unless ``recreate=True`` and the
    target is not the configured chunk collection.
    """

    collection_name = collection_name or get_entity_collection_name()
    embedding_dimensions = embedding_dimensions or int(
        get_embedding_metadata()["embedding_dimensions"]
    )
    owns_client = client is None

    protected_collections = {
        "acne_knowledge",
        CHUNK_COLLECTION_DEFAULT,
        QDRANT_COLLECTION_NAME,
        get_chunk_collection_name(),
    }
    if recreate and collection_name in protected_collections:
        raise RuntimeError(
            "Refusing to recreate the configured chunk collection. "
            f"Target collection was {collection_name!r}."
        )

    if client is None:
        from qdrant_client import AsyncQdrantClient  # type: ignore[import]

        client = AsyncQdrantClient(**qdrant_client_kwargs())

    try:
        from qdrant_client.models import Distance, VectorParams  # type: ignore[import]

        collections = await client.get_collections()
        existing_names = {collection.name for collection in collections.collections}

        if collection_name in existing_names and recreate:
            await client.delete_collection(collection_name=collection_name)
            existing_names.remove(collection_name)

        if collection_name not in existing_names:
            await client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=embedding_dimensions,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={BM25_VECTOR_NAME: bm25_sparse_vector_config()},
            )
            return

        info = await client.get_collection(collection_name=collection_name)
        _validate_entity_collection_schema(
            params=info.config.params,
            collection_name=collection_name,
            embedding_dimensions=embedding_dimensions,
        )
    finally:
        if owns_client:
            await client.close()


def build_entity_point(
    card: EntityCard,
    dense_vector: list[float],
    kb_version: str = "frozen_phase1_build",
    *,
    point_id: str | None = None,
) -> Any:
    """Build a point with dense and Qdrant-native BM25 inference inputs."""

    from qdrant_client.models import PointStruct  # type: ignore[import]

    payload = build_entity_point_payload(card, kb_version=kb_version, point_id=point_id)
    vector: dict[str, Any] = {
        "dense": dense_vector,
        BM25_VECTOR_NAME: bm25_document(payload["text"]),
    }

    return PointStruct(
        id=payload["point_id"],
        vector=vector,
        payload=payload,
    )


async def upsert_entity_cards(
    cards: list[EntityCard],
    embeddings: list[list[float]] | None = None,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    client: Any | None = None,
    collection_name: str | None = None,
    kb_version: str = "frozen_phase1_build",
    batch_size: int = 64,
) -> int:
    """Upsert entity cards into Qdrant using precomputed embeddings or a provider."""

    if embeddings is None:
        if embedding_provider is None:
            raise ValueError("Provide either embeddings or embedding_provider.")
        texts = [entity_card_to_text(card) for card in cards]
        maybe_embeddings = embedding_provider(texts)
        embeddings = await maybe_embeddings if inspect.isawaitable(maybe_embeddings) else maybe_embeddings

    if len(embeddings) != len(cards):
        raise ValueError(
            f"Embedding count mismatch: got {len(embeddings)}, expected {len(cards)}."
        )

    for index, vector in enumerate(embeddings):
        expected_dimensions = int(get_embedding_metadata()["embedding_dimensions"])
        if len(vector) != expected_dimensions:
            raise ValueError(
                f"Embedding dimension mismatch at card {index}: got {len(vector)}, "
                f"expected {expected_dimensions}."
            )

    collection_name = collection_name or get_entity_collection_name()
    owns_client = client is None
    if client is None:
        from qdrant_client import AsyncQdrantClient  # type: ignore[import]

        client = AsyncQdrantClient(**qdrant_client_kwargs())

    try:
        existing_point_ids = await _load_existing_entity_point_ids(
            client=client,
            collection_name=collection_name,
        )
        total = 0
        for start in range(0, len(cards), batch_size):
            batch_cards = cards[start:start + batch_size]
            batch_embeddings = embeddings[start:start + batch_size]
            points = [
                build_entity_point(
                    card,
                    dense_vector,
                    kb_version=kb_version,
                    point_id=existing_point_ids.get(entity_identity_key(card)),
                )
                for card, dense_vector in zip(batch_cards, batch_embeddings)
            ]
            await client.upsert(collection_name=collection_name, points=points)
            total += len(points)
        return total
    finally:
        if owns_client:
            await client.close()


async def _load_existing_entity_point_ids(client: Any, collection_name: str) -> dict[str, str]:
    """Read existing entity-card point IDs by canonical identity.

    This helper never mutates Qdrant. Duplicate identities are treated as an
    apply blocker because blindly upserting could leave stale duplicates.
    """

    point_ids: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    offset: Any = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            try:
                identity = entity_identity_key(payload)
            except ValueError:
                continue
            point_id = str(point.id)
            if identity in point_ids and point_ids[identity] != point_id:
                duplicates.setdefault(identity, [point_ids[identity]]).append(point_id)
                continue
            point_ids[identity] = point_id
        if offset is None:
            break
    if duplicates:
        raise RuntimeError(
            "Duplicate existing Qdrant entity identities block safe upsert: "
            + ", ".join(f"{identity}={ids}" for identity, ids in sorted(duplicates.items()))
        )
    return point_ids


def _validate_entity_collection_schema(
    params: Any,
    collection_name: str,
    embedding_dimensions: int,
) -> None:
    if isinstance(params, dict):
        vectors_config = params.get("vectors")
        sparse_vectors_config = params.get("sparse_vectors")
    else:
        vectors_config = getattr(params, "vectors", None)
        sparse_vectors_config = getattr(params, "sparse_vectors", None)

    dense_config = _get_named_config(vectors_config, "dense")
    bm25_config = _get_named_config(sparse_vectors_config, "bm25")
    schema_errors: list[str] = []

    if dense_config is None:
        schema_errors.append("missing named dense vector 'dense'")
    else:
        dense_size = (
            dense_config.get("size")
            if isinstance(dense_config, dict)
            else getattr(dense_config, "size", None)
        )
        if dense_size != embedding_dimensions:
            schema_errors.append(
                f"named vector 'dense' has size {dense_size}, expected {embedding_dimensions}"
            )

    if bm25_config is None:
        schema_errors.append("missing sparse vector 'bm25'")

    if schema_errors:
        raise RuntimeError(
            f"Qdrant entity collection schema mismatch for {collection_name!r}: "
            + "; ".join(schema_errors)
            + ". Do not delete chunk collections; create/migrate the entity collection manually "
            "or rerun the entity index script with a safe entity-only target."
        )


def _get_named_config(config: Any, name: str) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    if hasattr(config, "get"):
        return config.get(name)
    return None


__all__ = [
    "ENTITY_COLLECTION_DEFAULT",
    "build_entity_point",
    "build_entity_point_payload",
    "ensure_entity_collection",
    "entity_identity_key",
    "entity_point_id",
    "get_chunk_collection_name",
    "get_entity_collection_name",
    "upsert_entity_cards",
]
