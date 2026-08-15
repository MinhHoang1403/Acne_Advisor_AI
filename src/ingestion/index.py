"""Tạo candidate collections Qdrant cho knowledge records và EntityCards.

Module nhận Dense vectors đã được Gemini tạo, gửi chúng cùng BM25 ``Document``
cho Qdrant và kiểm số point sau upsert. Qdrant thực thi cosine/BM25 indexing;
Python không tự tính sparse representation hay search score tại đây.

Physical collection gắn ``build_id`` để có thể kiểm tra trước khi chuyển logical
alias. Entity collection là structural asset và các payload của nó được đánh dấu
không đủ điều kiện làm medical evidence trong normal answer path.

Điểm thường cần chỉnh sửa: batch/provider reuse trong ``resolve_embeddings``;
Qdrant schema/upsert trong hai hàm ``build_*_candidate``; alias activation trong
``switch_alias``. Đây là knowledge-preparation code, không chạy cho mỗi chat request.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from src.database.vector_store import qdrant_client_kwargs
from src.ingestion.bm25 import BM25_VECTOR_NAME, bm25_document, bm25_sparse_vector_config
from src.ingestion.build import CompiledKnowledge
from src.ingestion.embedding import EMBEDDING_DIMENSIONS, EmbeddingCache, embed_documents
from src.knowledge.entity_cards import entity_card_to_text
from src.knowledge.entity_identity import entity_point_id
from src.knowledge.schemas import EntityCard
from src.resilience.exceptions import ProviderUnavailableError


logger = logging.getLogger(__name__)
DENSE_VECTOR_NAME = "dense"
KNOWLEDGE_LOGICAL_COLLECTION = "acne_knowledge"
ENTITY_LOGICAL_COLLECTION = "acne_entities"


def knowledge_physical_collection(build_id: str) -> str:
    return f"acne_knowledge__{build_id}"


def entity_physical_collection(build_id: str) -> str:
    return f"acne_entities__{build_id}"


async def seed_embedding_cache_from_collection(
    client: AsyncQdrantClient,
    *,
    collection_name: str,
    point_ids: list[str],
    cache: EmbeddingCache,
) -> dict[str, int]:
    """Nạp lại exact Dense vectors và cô lập từng point không đọc được."""

    loaded = 0
    failed = 0
    for start in range(0, len(point_ids), 64):
        points, batch_failed = await _retrieve_dense_resilient(
            client, collection_name, point_ids[start:start + 64]
        )
        failed += batch_failed
        for point in points:
            payload = point.payload or {}
            text = str(payload.get("text") or payload.get("content") or "")
            vector = point.vector.get(DENSE_VECTOR_NAME) if isinstance(point.vector, dict) else None
            if text and isinstance(vector, list) and len(vector) == EMBEDDING_DIMENSIONS:
                cache.put(text, vector)
                loaded += 1
    return {"loaded": loaded, "unreadable": failed}


async def _retrieve_dense_resilient(
    client: AsyncQdrantClient,
    collection_name: str,
    point_ids: list[str],
) -> tuple[list[Any], int]:
    """Chia đôi batch lỗi để giữ lại các vector cũ vẫn đọc được."""

    if not point_ids:
        return [], 0
    try:
        points = await client.retrieve(
            collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=[DENSE_VECTOR_NAME],
        )
        return list(points), 0
    except Exception as exc:
        if len(point_ids) == 1:
            logger.warning("Unable to reuse legacy point %s: %s", point_ids[0], type(exc).__name__)
            return [], 1
        middle = len(point_ids) // 2
        left, left_failed = await _retrieve_dense_resilient(
            client, collection_name, point_ids[:middle]
        )
        right, right_failed = await _retrieve_dense_resilient(
            client, collection_name, point_ids[middle:]
        )
        return [*left, *right], left_failed + right_failed


async def resolve_embeddings(
    texts: list[str],
    *,
    cache: EmbeddingCache,
    api_key: str,
    batch_size: int = 16,
    batch_delay_seconds: float | None = None,
    max_retries: int = 4,
) -> tuple[list[list[float]], dict[str, int]]:
    """Ghép cache hit với vector mới và lưu ngay mỗi provider batch thành công.

    Gemini tạo vector cho cache miss. ``batch_size``, delay và retry là giới hạn
    vận hành/quota, không phải quality threshold. Hàm giữ đúng thứ tự input để
    vector tiếp tục khớp với compiled records khi upsert.
    """

    vectors: list[list[float] | None] = [cache.get(text) for text in texts]
    misses = [index for index, vector in enumerate(vectors) if vector is None]
    provider_calls = 0
    retry_count = 0
    delay = batch_delay_seconds
    if delay is None:
        configured_delay = float(os.getenv("EMBEDDING_BATCH_DELAY", "10") or "10")
        # Quota được tính theo từng input; delay hữu hạn giữ tốc độ batch trong
        # giới hạn cấu hình provider và không ảnh hưởng nội dung vector.
        delay = max(configured_delay, 10.0)
    for start in range(0, len(misses), batch_size):
        indexes = misses[start:start + batch_size]
        batch_texts = [texts[index] for index in indexes]
        batch_vectors = None
        for attempt in range(max_retries + 1):
            try:
                batch_vectors = await asyncio.to_thread(
                    embed_documents, batch_texts, api_key=api_key
                )
                provider_calls += 1
                break
            except ProviderUnavailableError:
                if attempt >= max_retries:
                    raise
                retry_count += 1
                retry_delay = 30.0 * (attempt + 1)
                logger.warning(
                    "Embedding batch rate-limited; retry %d/%d in %.0f seconds.",
                    attempt + 1,
                    max_retries,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)
        if batch_vectors is None:
            raise RuntimeError("Embedding provider returned no batch")
        for index, text, vector in zip(indexes, batch_texts, batch_vectors, strict=True):
            cache.put(text, vector)
            vectors[index] = vector
        if start + batch_size < len(misses) and delay > 0:
            await asyncio.sleep(delay)
    if any(vector is None for vector in vectors):
        raise RuntimeError("Embedding resolution left unresolved vectors")
    return [vector for vector in vectors if vector is not None], {
        "cache_hits": len(texts) - len(misses),
        "cache_misses": len(misses),
        "provider_calls": provider_calls,
        "retry_count": retry_count,
    }


async def build_knowledge_candidate(
    compiled: CompiledKnowledge,
    vectors: list[list[float]],
    *,
    client: AsyncQdrantClient | None = None,
    replace_candidate: bool = False,
) -> dict[str, Any]:
    """Tạo physical knowledge collection và upsert Dense + BM25 records."""

    if len(vectors) != len(compiled.records):
        raise ValueError("Dense vector count does not match compiled records")
    owns_client = client is None
    client = client or AsyncQdrantClient(**qdrant_client_kwargs())
    collection = knowledge_physical_collection(compiled.identity.build_id)
    try:
        names = {item.name for item in (await client.get_collections()).collections}
        if collection in names and replace_candidate:
            await client.delete_collection(collection)
            names.remove(collection)
        if collection not in names:
            await client.create_collection(
                collection_name=collection,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=EMBEDDING_DIMENSIONS,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={BM25_VECTOR_NAME: bm25_sparse_vector_config()},
            )
        for start in range(0, len(compiled.records), 32):
            points = [
                models.PointStruct(
                    id=record["chunk_id"],
                    vector={
                        DENSE_VECTOR_NAME: vector,
                        BM25_VECTOR_NAME: bm25_document(record["text"]),
                    },
                    payload=record,
                )
                for record, vector in zip(
                    compiled.records[start:start + 32],
                    vectors[start:start + 32],
                    strict=True,
                )
            ]
            await client.upsert(collection_name=collection, points=points, wait=True)
        info = await client.get_collection(collection)
        return {
            "collection": collection,
            "points": int(info.points_count or 0),
            "indexed_vectors": int(info.indexed_vectors_count or 0),
        }
    finally:
        if owns_client:
            await client.close()


async def build_entity_candidate(
    cards: list[EntityCard],
    vectors: list[list[float]],
    *,
    build_id: str,
    taxonomy_hash: str,
    client: AsyncQdrantClient | None = None,
    replace_candidate: bool = False,
) -> dict[str, Any]:
    """Tạo EntityCard index cho structural lookup, không cho runtime grounding."""

    if len(vectors) != len(cards):
        raise ValueError("Dense vector count does not match EntityCard count")
    owns_client = client is None
    client = client or AsyncQdrantClient(**qdrant_client_kwargs())
    collection = entity_physical_collection(build_id)
    try:
        names = {item.name for item in (await client.get_collections()).collections}
        if collection in names and replace_candidate:
            await client.delete_collection(collection)
            names.remove(collection)
        if collection not in names:
            await client.create_collection(
                collection_name=collection,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=EMBEDDING_DIMENSIONS,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={BM25_VECTOR_NAME: bm25_sparse_vector_config()},
            )
        points = []
        for card, vector in zip(cards, vectors, strict=True):
            text = entity_card_to_text(card)
            payload = card.to_payload()
            payload.update(
                {
                    "entity_id": card.stable_id(build_id),
                    "text": text,
                    "source_ids": card.source_ids,
                    "taxonomy_hash": taxonomy_hash,
                    "entity_schema": "source_backed_entity_card",
                    "build_id": build_id,
                    "medical_evidence_eligible": False,
                }
            )
            points.append(
                models.PointStruct(
                    id=entity_point_id(card),
                    vector={DENSE_VECTOR_NAME: vector, BM25_VECTOR_NAME: bm25_document(text)},
                    payload=payload,
                )
            )
        for start in range(0, len(points), 32):
            await client.upsert(collection_name=collection, points=points[start:start + 32], wait=True)
        info = await client.get_collection(collection)
        return {"collection": collection, "points": int(info.points_count or 0)}
    finally:
        if owns_client:
            await client.close()


async def switch_alias(
    client: AsyncQdrantClient,
    *,
    alias_name: str,
    target_collection: str,
) -> None:
    """Chuyển logical alias sang physical collection đã được validation."""

    aliases = {item.alias_name for item in (await client.get_aliases()).aliases}
    operations: list[Any] = []
    if alias_name in aliases:
        operations.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias_name)))
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name=target_collection,
                alias_name=alias_name,
            )
        )
    )
    await client.update_collection_aliases(operations)


__all__ = [
    "DENSE_VECTOR_NAME",
    "ENTITY_LOGICAL_COLLECTION",
    "KNOWLEDGE_LOGICAL_COLLECTION",
    "build_knowledge_candidate",
    "build_entity_candidate",
    "entity_physical_collection",
    "knowledge_physical_collection",
    "resolve_embeddings",
    "seed_embedding_cache_from_collection",
    "switch_alias",
]
