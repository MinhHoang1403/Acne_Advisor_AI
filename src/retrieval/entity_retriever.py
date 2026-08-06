"""Exact/payload entity retrieval for the ``acne_entities_v1`` Qdrant collection."""

from __future__ import annotations

import logging
from typing import Any

from src.database.vector_store import qdrant_client_kwargs
from src.knowledge.entity_cards import entity_card_to_text
from src.knowledge.entity_index import get_entity_collection_name
from src.knowledge.normalizer import normalize_text_key
from src.knowledge.schemas import EntityCard
from src.retrieval.contracts import NormalizedQuery, QueryExpansion, RetrievedCandidate

logger = logging.getLogger(__name__)

ENTITY_PAYLOAD_FIELDS = (
    "entity_type",
    "canonical_name",
    "aliases",
    "active_ingredients",
    "drug_class",
    "safety_contexts",
    "contraindications",
    "side_effects",
    "used_for",
    "kb_version",
    "taxonomy_version",
    "entity_schema_version",
)


class EntityRetriever:
    """Read-only exact payload matcher for entity cards."""

    _shared_client: Any | None = None
    _payload_cache: dict[str, list[dict[str, Any]]] = {}

    def __init__(self, collection_name: str | None = None) -> None:
        self.collection_name = collection_name or get_entity_collection_name()
        self._client: Any | None = None

    async def retrieve(
        self,
        normalized_query: NormalizedQuery,
        expansion: QueryExpansion,
        limit: int = 8,
    ) -> list[RetrievedCandidate]:
        """Retrieve entity cards by scrolling small entity collection and matching payloads."""

        payloads = await self._scroll_entity_payloads()
        return retrieve_entity_candidates_from_payloads(
            normalized_query=normalized_query,
            expansion=expansion,
            payloads=payloads,
            collection_name=self.collection_name,
            limit=limit,
        )

    async def _scroll_entity_payloads(self) -> list[dict[str, Any]]:
        from qdrant_client import AsyncQdrantClient  # type: ignore[import]

        cached = self.__class__._payload_cache.get(self.collection_name)
        if cached is not None:
            return [dict(payload) for payload in cached]
        if self.__class__._shared_client is None:
            self.__class__._shared_client = AsyncQdrantClient(**qdrant_client_kwargs())
        self._client = self.__class__._shared_client

        points, _ = await self._client.scroll(
            collection_name=self.collection_name,
            limit=128,
            with_payload=True,
            with_vectors=False,
        )

        payloads: list[dict[str, Any]] = []
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            payload["_qdrant_point_id"] = str(getattr(point, "id", ""))
            payloads.append(payload)
        self.__class__._payload_cache[self.collection_name] = [dict(payload) for payload in payloads]
        return payloads

    async def close(self) -> None:
        """Keep the immutable entity-card cache and shared client for reuse."""

        return None

    @classmethod
    async def close_shared_client(cls) -> None:
        if cls._shared_client is not None:
            await cls._shared_client.close()
            cls._shared_client = None
        cls._payload_cache.clear()


def retrieve_entity_candidates_from_payloads(
    normalized_query: NormalizedQuery,
    expansion: QueryExpansion,
    payloads: list[dict[str, Any]],
    collection_name: str | None = None,
    limit: int = 8,
) -> list[RetrievedCandidate]:
    """Return exact entity matches from already-loaded payloads.

    This helper is intentionally pure so tests do not need Qdrant.
    """

    collection = collection_name or get_entity_collection_name()
    match_terms = _query_match_terms(normalized_query, expansion)
    candidates: list[RetrievedCandidate] = []

    for payload in payloads:
        matched_metadata = _match_payload(payload, match_terms, normalized_query)
        if not matched_metadata:
            continue
        score = _score_match(matched_metadata, normalized_query.intent)
        candidates.append(
            RetrievedCandidate(
                candidate_id=_candidate_id(payload),
                source="entity",
                collection=collection,
                text=_entity_text(payload),
                score=score,
                fused_score=None,
                payload=_stable_entity_payload(payload),
                matched_metadata=matched_metadata,
                rank=None,
                debug={"match_terms": sorted(match_terms)},
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.score or 0.0,
            _entity_type_priority(candidate.payload.get("entity_type"), normalized_query.intent),
        ),
        reverse=True,
    )
    return candidates[:limit]


def _query_match_terms(normalized_query: NormalizedQuery, expansion: QueryExpansion) -> set[str]:
    values = [
        *normalized_query.drug_product,
        *normalized_query.active_ingredient,
        *normalized_query.drug_class,
        *normalized_query.condition,
        *normalized_query.safety_context,
        *normalized_query.aliases,
        *expansion.canonical_terms,
        *expansion.alias_terms,
    ]
    return {normalize_text_key(value) for value in values if value}


def _match_payload(
    payload: dict[str, Any],
    match_terms: set[str],
    normalized_query: NormalizedQuery,
) -> dict[str, Any]:
    fields = {
        "canonical_name": [payload.get("canonical_name")],
        "entity_id": [payload.get("entity_id")],
        "aliases": _as_list(payload.get("aliases")),
        "active_ingredients": _as_list(payload.get("active_ingredients")),
        "drug_class": _as_list(payload.get("drug_class")),
        "safety_contexts": _as_list(payload.get("safety_contexts")),
        "used_for": _as_list(payload.get("used_for")),
    }
    matched: dict[str, Any] = {}
    for field, raw_values in fields.items():
        values = [str(value) for value in raw_values if value]
        hits = [
            value for value in values
            if normalize_text_key(value) in match_terms
        ]
        if hits:
            matched[field] = hits

    if normalized_query.intent == "acne_type" and payload.get("entity_type") == "condition":
        if "acne_vulgaris" in normalized_query.condition:
            matched.setdefault("condition", ["acne_vulgaris"])

    return matched


def _score_match(matched_metadata: dict[str, Any], intent: str) -> float:
    score = 0.35
    weights = {
        "canonical_name": 0.35,
        "aliases": 0.20,
        "active_ingredients": 0.18,
        "drug_class": 0.14,
        "entity_id": 0.12,
        "condition": 0.12,
        "safety_contexts": 0.10,
        "used_for": 0.08,
    }
    for field, weight in weights.items():
        if matched_metadata.get(field):
            score += weight
    if intent in {"drug_identity", "ingredient_question", "class_check"}:
        if matched_metadata.get("canonical_name") or matched_metadata.get("aliases"):
            score += 0.15
    return round(min(score, 1.0), 6)


def _entity_type_priority(entity_type: Any, intent: str) -> int:
    if intent in {"drug_identity", "ingredient_question"}:
        order = {"drug_product": 5, "active_ingredient": 4, "drug_class": 3}
    elif intent == "class_check":
        order = {"active_ingredient": 5, "drug_product": 4, "drug_class": 3}
    elif intent == "acne_type":
        order = {"condition": 5, "safety_context": 3}
    else:
        order = {"active_ingredient": 4, "drug_product": 4, "drug_class": 3, "condition": 2}
    return order.get(str(entity_type), 0)


def _candidate_id(payload: dict[str, Any]) -> str:
    return str(
        payload.get("entity_id")
        or payload.get("point_id")
        or payload.get("_qdrant_point_id")
        or f"{payload.get('entity_type', 'entity')}:{payload.get('canonical_name', '')}"
    )


def _entity_text(payload: dict[str, Any]) -> str:
    text = str(payload.get("text") or "").strip()
    if text:
        return text
    try:
        card = EntityCard(**{
            field: payload.get(field)
            for field in (
                "entity_type",
                "canonical_name",
                "aliases",
                "active_ingredients",
                "drug_class",
                "used_for",
                "side_effects",
                "contraindications",
                "safety_contexts",
                "source_ids",
                "taxonomy_version",
                "entity_schema_version",
                "metadata",
            )
            if field in payload
        })
        return entity_card_to_text(card)
    except Exception:
        return str(payload.get("canonical_name") or "")


def _stable_entity_payload(payload: dict[str, Any]) -> dict[str, Any]:
    stable = {field: payload.get(field, [] if field.endswith("s") else None) for field in ENTITY_PAYLOAD_FIELDS}
    for extra in ("entity_id", "point_id", "metadata", "text"):
        if extra in payload:
            stable[extra] = payload[extra]
    return stable


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


__all__ = [
    "ENTITY_PAYLOAD_FIELDS",
    "EntityRetriever",
    "retrieve_entity_candidates_from_payloads",
]
