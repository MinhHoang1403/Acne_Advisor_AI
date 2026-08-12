"""
src/database/graph_store.py – Neo4j Knowledge Graph Store
=========================================================
Read-only async client for querying the Neo4j knowledge graph
populated during Phase 1 ingestion.

Supported schemas
-----------------
Current deterministic graph:
    Node labels : DrugProduct, ActiveIngredient, DrugClass, Condition,
                  SafetyContext
    Node props  : canonical_name, entity_id, aliases, metadata_json, ...
    Relationships : HAS_ACTIVE_INGREDIENT, BELONGS_TO_CLASS
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.database.neo4j_queries import ENTITY_CONTEXT_CYPHER, KEYWORD_SEARCH_CYPHER
from src.quality.safe_fallback import sanitize_fallback_reason

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (from .env)
# ---------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE") or None


def _normalize_entity_names(entity_names: list[str]) -> tuple[list[str], list[str]]:
    """Return exact names plus casefolded canonical/alias lookup names."""
    cleaned = [name.strip() for name in entity_names if name and name.strip()]
    canonical_names = sorted(set(cleaned), key=str.casefold)
    lookup_names = sorted({name.casefold() for name in cleaned})
    return canonical_names, lookup_names


def _normalize_keywords(keywords: list[str]) -> list[str]:
    """Normalize keyword fallback terms without allowing very broad matches."""
    return sorted(
        {
            keyword.strip().casefold()
            for keyword in keywords
            if len(keyword.strip()) >= 3
        }
    )


class Neo4jGraphStore:
    """Async read-only client for the acne knowledge graph in Neo4j.

    Usage
    -----
    store = Neo4jGraphStore()
    facts = await store.get_entity_context(["isotretinoin", "acne vulgaris"])
    await store.close()
    """

    _shared_driver: Any | None = None

    def __init__(self) -> None:
        from neo4j import AsyncGraphDatabase  # type: ignore[import]

        if self.__class__._shared_driver is None:
            self.__class__._shared_driver = AsyncGraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
            )
            logger.debug("Created process-level Neo4j driver for %s", NEO4J_URI)
        self._driver = self.__class__._shared_driver

    async def get_entity_context(
        self,
        entity_names: list[str],
        limit: int = 50,
        *,
        raise_on_error: bool = False,
    ) -> list[dict[str, Any]]:
        """Get 1-hop relationships for given entity names.

        Parameters
        ----------
        entity_names : list[str]
            Entity names to look up. Current deterministic graph matches
            ``canonical_name``.
        limit : int
            Maximum number of fact records to return.

        Returns
        -------
        list[dict]
            Each dict contains: entity, entity_type, description,
            relationship, related_entity, related_type,
            related_description, evidence.
        """
        if not entity_names:
            return []

        canonical_names, lookup_names = _normalize_entity_names(entity_names)

        if not canonical_names and not lookup_names:
            return []

        facts: list[dict[str, Any]] = []

        try:
            async with self._driver.session(database=NEO4J_DATABASE) as session:
                result = await session.run(
                    ENTITY_CONTEXT_CYPHER,
                    canonical_names=canonical_names,
                    lookup_names=lookup_names,
                    limit=limit,
                )
                records = [record async for record in result]

                seen_entities: set[str] = set()
                for record in records:
                    fact = dict(record)
                    rel = fact.get("relationship")
                    entity = fact.get("entity", "")

                    if rel is not None:
                        # Has a relationship – include it
                        facts.append(fact)
                        seen_entities.add(entity)
                    elif entity not in seen_entities:
                        # Entity without relationships (isolated node)
                        facts.append(fact)
                        seen_entities.add(entity)

        except Exception as exc:
            logger.error("Neo4j get_entity_context failed: %s", sanitize_fallback_reason(exc))
            if raise_on_error:
                raise

        logger.debug(
            "Neo4j: queried %d entity names → %d facts",
            len(set(canonical_names + lookup_names)),
            len(facts),
        )

        return facts

    async def search_by_keywords(
        self,
        keywords: list[str],
        limit: int = 30,
        *,
        raise_on_error: bool = False,
    ) -> list[dict[str, Any]]:
        """Fallback: search entities whose name contains any keyword.

        Used when ``graph_nodes`` from Qdrant payload are empty.

        Parameters
        ----------
        keywords : list[str]
            Keywords to search (min 3 chars each, lowercased).
        limit : int
            Maximum number of fact records to return.
        """
        if not keywords:
            return []

        keywords = _normalize_keywords(keywords)

        if not keywords:
            return []

        facts: list[dict[str, Any]] = []

        try:
            async with self._driver.session(database=NEO4J_DATABASE) as session:
                result = await session.run(
                    KEYWORD_SEARCH_CYPHER,
                    keywords=keywords,
                    limit=limit,
                )
                records = [record async for record in result]

                for record in records:
                    facts.append(dict(record))

        except Exception as exc:
            logger.error("Neo4j keyword search failed: %s", sanitize_fallback_reason(exc))
            if raise_on_error:
                raise

        logger.debug(
            "Neo4j: keyword search with %d sanitized keywords → %d facts",
            len(keywords),
            len(facts),
        )

        return facts

    async def close(self) -> None:
        """Keep the shared read driver alive for the process lifetime."""

        return None

    @classmethod
    async def close_shared_driver(cls) -> None:
        """Close the shared driver explicitly during controlled process shutdown."""

        if cls._shared_driver is not None:
            await cls._shared_driver.close()
            cls._shared_driver = None
