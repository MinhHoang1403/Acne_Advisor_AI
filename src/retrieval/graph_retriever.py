"""Truy xuất đồ thị hai tầng (Lexical + Clinical GraphRAG) trên Neo4j.

Module cung cấp kênh truy xuất thứ 3 song song với Qdrant Dense và Qdrant BM25:
1. Trích xuất các điểm neo đặc hiệu (hoạt chất, biệt dược, phân nhóm thuốc,
   ngữ cảnh an toàn, phân thể mụn và chủ đề chăm sóc lâm sàng) từ câu hỏi.
2. Duyệt đồ thị đa bước có định hướng y khoa trên Neo4j:
   - Khớp trực tiếp ``MENTIONS_ENTITY`` và ``HAS_CLINICAL_FACET``.
   - Mở rộng cấu trúc 1 bước có kiểm soát:
     * ``DrugProduct -[:HAS_ACTIVE_INGREDIENT]-> ActiveIngredient``
     * ``DrugClass <-[:BELONGS_TO_CLASS]- ActiveIngredient``
     * ``SafetyContext <-[:CONTRAINDICATED_IN]- DrugProduct/ActiveIngredient/DrugClass``
     * ``ActiveIngredient/DrugClass -[:CLINICAL_INTERACTION]- ActiveIngredient/DrugClass``
   - Cộng điểm giao thoa (co-mention synergy) cho các ``KnowledgeChunk`` nằm tại
     giao điểm của nhiều ràng buộc lâm sàng trong cùng câu hỏi.
3. Mở rộng láng giềng ``NEXT_CHUNK`` có kiểm soát cho các đoạn bảng biểu bị tách
   để đưa qua BGE Cross-Encoder Reranker chấm điểm trước khi đóng gói ngữ cảnh.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import json
import logging
import os
import time
from typing import Any

from src.knowledge.lexical_graph import (
    QueryGraphAnchors,
    extract_query_graph_anchors,
)

logger = logging.getLogger(__name__)


@dataclass
class GraphSearchResult:
    """Kết quả truy vấn kênh đồ thị Neo4j kèm đường dẫn giải thích (provenance paths)."""

    candidates: list[dict[str, Any]] = field(default_factory=list)
    anchors: dict[str, Any] = field(default_factory=dict)
    paths: list[str] = field(default_factory=list)
    status: str = "disabled"
    elapsed_ms: float = 0.0


def _neo4j_runtime_config() -> tuple[str, str, str]:
    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687").strip()
    user = os.getenv("NEO4J_USERNAME", "").strip()
    password = os.getenv("NEO4J_PASSWORD", "").strip()
    auth_env = os.getenv("NEO4J_AUTH", "").strip()
    if (not user or not password) and "/" in auth_env:
        parsed_user, parsed_pass = auth_env.split("/", 1)
        user = user or parsed_user.strip()
        password = password or parsed_pass.strip()
    return uri, (user or "neo4j"), (password or "password")


# Truy vấn Cypher hợp nhất đa hướng trên đồ thị hai tầng:
# - Tính điểm khớp trực tiếp thực thể, khớp trực tiếp khía cạnh lâm sàng, mở rộng
#   taxonomy/interaction 1 bước có định hướng, và thưởng điểm giao thoa đa ràng buộc.
_HYBRID_GRAPH_SEARCH_CYPHER = """
CALL () {
  MATCH (de)
  WHERE (de:ActiveIngredient OR de:DrugProduct OR de:DrugClass OR de:SafetyContext)
    AND de.canonical_name IN $entity_names
  WITH de
  MATCH (de)<-[me:MENTIONS_ENTITY]-(c:KnowledgeChunk)
  WHERE ($kb_version = '' OR c.kb_version = $kb_version)
  RETURN c, 'entity' AS kind, head(labels(de)) + ':' + de.canonical_name AS token, coalesce(me.weight, 1.0) AS w
  UNION ALL
  MATCH (df:ClinicalFacet)
  WHERE df.facet_key IN $facet_keys
  WITH df
  MATCH (df)<-[hf:HAS_CLINICAL_FACET]-(c:KnowledgeChunk)
  WHERE ($kb_version = '' OR c.kb_version = $kb_version)
  RETURN c, 'facet' AS kind, 'ClinicalFacet:' + df.facet_key AS token, coalesce(hf.weight, 1.0) AS w
  UNION ALL
  MATCH (seed_prod:DrugProduct)-[:HAS_ACTIVE_INGREDIENT]->(prod_ing:ActiveIngredient)
  WHERE seed_prod.canonical_name IN $entity_names
  WITH seed_prod, prod_ing
  MATCH (prod_ing)<-[:MENTIONS_ENTITY]-(c:KnowledgeChunk)
  WHERE ($kb_version = '' OR c.kb_version = $kb_version)
  RETURN c, 'expansion' AS kind, seed_prod.canonical_name + '-[HAS_ACTIVE_INGREDIENT]->' + prod_ing.canonical_name AS token, 1.10 AS w
  UNION ALL
  MATCH (seed_cls:DrugClass)<-[:BELONGS_TO_CLASS]-(cls_ing:ActiveIngredient)
  WHERE seed_cls.canonical_name IN $entity_names
    AND NOT cls_ing.canonical_name IN $entity_names
  WITH seed_cls, cls_ing
  MATCH (cls_ing)<-[:MENTIONS_ENTITY]-(c:KnowledgeChunk)
  WHERE ($kb_version = '' OR c.kb_version = $kb_version)
  RETURN c, 'expansion' AS kind, seed_cls.canonical_name + '-[CLASS_MEMBER]->' + cls_ing.canonical_name AS token, 0.72 AS w
  UNION ALL
  MATCH (seed_safe:SafetyContext)<-[:CONTRAINDICATED_IN]-(contra)
  WHERE seed_safe.canonical_name IN $entity_names
  WITH seed_safe, contra
  MATCH (contra)<-[:MENTIONS_ENTITY]-(c:KnowledgeChunk)
  WHERE ($kb_version = '' OR c.kb_version = $kb_version)
  RETURN c, 'expansion' AS kind, seed_safe.canonical_name + '<-[CONTRAINDICATED_IN]-' + contra.canonical_name AS token, 0.88 AS w
  UNION ALL
  MATCH (seed_int)-[:CLINICAL_INTERACTION]-(inter_target)
  WHERE (seed_int:ActiveIngredient OR seed_int:DrugClass)
    AND seed_int.canonical_name IN $entity_names
  WITH seed_int, inter_target
  MATCH (inter_target)<-[:MENTIONS_ENTITY]-(c:KnowledgeChunk)
  WHERE ($kb_version = '' OR c.kb_version = $kb_version)
  RETURN c, 'expansion' AS kind, seed_int.canonical_name + '-[CLINICAL_INTERACTION]->' + inter_target.canonical_name AS token, 0.65 AS w
}
WITH c, kind, token, max(w) AS weight
WITH c, collect({kind: kind, token: token, weight: weight}) AS hits
WITH c,
     [h IN hits WHERE h.kind = 'entity' | h.token] AS matched_entities,
     [h IN hits WHERE h.kind = 'facet' | h.token] AS matched_facets,
     [h IN hits WHERE h.kind = 'expansion' | h.token] AS expanded_paths,
     reduce(s = 0.0, h IN hits | s + h.weight) AS base_score,
     size([h IN hits WHERE h.kind IN ['entity', 'facet']]) AS direct_anchor_hits,
     size([h IN hits WHERE h.kind = 'expansion']) AS expansion_hits
WITH c, matched_entities, matched_facets, expanded_paths,
     (
       base_score
       + CASE
           WHEN direct_anchor_hits >= 2 THEN 0.85 * (direct_anchor_hits - 1)
           WHEN direct_anchor_hits = 1 AND expansion_hits >= 1 THEN 0.35
           WHEN direct_anchor_hits = 0 AND expansion_hits >= 2 THEN 0.55
           ELSE 0.0
         END
     ) AS graph_score
ORDER BY graph_score DESC,
         CASE WHEN c.content_kind = 'text' THEN 1 ELSE 0 END DESC,
         c.point_id ASC
LIMIT $top_k
RETURN c.point_id AS point_id,
       c.text AS text,
       c.payload_json AS payload_json,
       round(graph_score * 10000.0) / 10000.0 AS score,
       matched_entities,
       matched_facets,
       expanded_paths
"""

_ADJACENT_CHUNKS_CYPHER = """
MATCH (seed:KnowledgeChunk)-[r:NEXT_CHUNK]-(neighbor:KnowledgeChunk)
WHERE seed.point_id IN $seed_ids
  AND NOT neighbor.point_id IN $exclude_ids
  AND ($kb_version = '' OR neighbor.kb_version = $kb_version)
  AND (
    coalesce(r.same_table, false) = true
    OR (seed.content_kind = 'table' AND coalesce(r.same_section, false) = true)
  )
WITH neighbor,
     max(CASE WHEN coalesce(r.same_table, false) = true THEN 2 ELSE 1 END) AS priority
ORDER BY priority DESC, neighbor.chunk_index ASC, neighbor.point_id ASC
LIMIT $limit
RETURN neighbor.point_id AS point_id,
       neighbor.text AS text,
       neighbor.payload_json AS payload_json
"""


def _hydrate_chunk_candidate(
    *,
    point_id: str,
    text: str,
    payload_json: str | None,
    score: float,
    matched_entities: list[str] | None = None,
    matched_facets: list[str] | None = None,
    expanded_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Khôi phục bản ghi candidate đồng nhất 100% với cấu trúc trả về từ Qdrant."""

    payload: dict[str, Any] = {}
    if payload_json:
        try:
            parsed = json.loads(payload_json)
            if isinstance(parsed, dict):
                payload = parsed
        except ValueError:
            payload = {}

    payload.setdefault("chunk_id", point_id)
    payload.setdefault("text", text)
    if matched_entities or matched_facets or expanded_paths:
        payload["graph_evidence"] = {
            "matched_entities": list(matched_entities or []),
            "matched_facets": list(matched_facets or []),
            "expanded_paths": list(expanded_paths or []),
        }
    return {
        "id": point_id,
        "score": float(score),
        **payload,
    }


class Neo4jGraphStore:
    """Adapter bất đồng bộ dùng chung driver cho truy vấn GraphRAG tại thời điểm runtime."""

    _shared_driver: Any | None = None
    _shared_config: tuple[str, str, str] | None = None

    def __init__(self, driver: Any | None = None) -> None:
        self._injected_driver = driver

    def _get_driver(self) -> Any:
        if self._injected_driver is not None:
            return self._injected_driver

        from neo4j import AsyncGraphDatabase  # type: ignore[import]

        config = _neo4j_runtime_config()
        cls = self.__class__
        if cls._shared_driver is None or cls._shared_config != config:
            old_driver = cls._shared_driver
            uri, user, password = config
            cls._shared_driver = AsyncGraphDatabase.driver(
                uri,
                auth=(user, password),
                max_connection_pool_size=16,
                connection_acquisition_timeout=4.0,
            )
            cls._shared_config = config
            if old_driver is not None:
                try:
                    import asyncio

                    loop = asyncio.get_running_loop()
                    loop.create_task(old_driver.close())
                except RuntimeError:
                    pass
            logger.debug("Created process-level Neo4j AsyncDriver for GraphRAG read operations.")
        return cls._shared_driver

    async def search_graph(
        self,
        query: str,
        *,
        top_k: int = 16,
        kb_version: str | None = None,
    ) -> GraphSearchResult:
        """Thực thi truy xuất đồ thị hai tầng cho câu hỏi người dùng."""

        started = time.perf_counter()
        anchors: QueryGraphAnchors = extract_query_graph_anchors(query)
        anchor_trace = anchors.to_trace_dict()

        if not anchors.has_specific_anchors:
            return GraphSearchResult(
                candidates=[],
                anchors=anchor_trace,
                paths=[],
                status="no_anchors",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            )

        driver = self._get_driver()
        clean_kb = str(kb_version or "").strip()
        candidates: list[dict[str, Any]] = []
        collected_paths: list[str] = list(anchors.brand_bridge_paths)
        seen_paths: set[str] = set(collected_paths)

        async with driver.session() as session:
            result = await session.run(
                _HYBRID_GRAPH_SEARCH_CYPHER,
                entity_names=anchors.entity_names,
                facet_keys=anchors.facet_keys,
                kb_version=clean_kb,
                top_k=max(1, int(top_k)),
            )
            rows = await result.data()

        for row in rows:
            point_id = str(row.get("point_id") or "").strip()
            if not point_id:
                continue
            matched_entities = [str(x) for x in (row.get("matched_entities") or []) if x]
            matched_facets = [str(x) for x in (row.get("matched_facets") or []) if x]
            expanded_paths = [str(x) for x in (row.get("expanded_paths") or []) if x]

            candidates.append(
                _hydrate_chunk_candidate(
                    point_id=point_id,
                    text=str(row.get("text") or ""),
                    payload_json=row.get("payload_json"),
                    score=float(row.get("score") or 0.0),
                    matched_entities=matched_entities,
                    matched_facets=matched_facets,
                    expanded_paths=expanded_paths,
                )
            )
            for path_item in [*matched_entities, *matched_facets, *expanded_paths]:
                if path_item not in seen_paths and len(collected_paths) < 12:
                    seen_paths.add(path_item)
                    collected_paths.append(path_item)

        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        return GraphSearchResult(
            candidates=candidates,
            anchors=anchor_trace,
            paths=collected_paths,
            status="ok" if candidates else "empty",
            elapsed_ms=elapsed_ms,
        )

    async def fetch_adjacent_table_chunks(
        self,
        seed_chunk_ids: Sequence[str],
        *,
        exclude_ids: set[str] | None = None,
        kb_version: str | None = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Lấy các đoạn bảng biểu liền kề qua cạnh NEXT_CHUNK để tránh đứt gãy ngữ cảnh bảng."""

        clean_seeds = [str(cid).strip() for cid in seed_chunk_ids if str(cid).strip()]
        if not clean_seeds or limit <= 0:
            return []

        excluded = set(exclude_ids or set()) | set(clean_seeds)
        driver = self._get_driver()
        async with driver.session() as session:
            result = await session.run(
                _ADJACENT_CHUNKS_CYPHER,
                seed_ids=clean_seeds,
                exclude_ids=sorted(excluded),
                kb_version=str(kb_version or "").strip(),
                limit=int(limit),
            )
            rows = await result.data()

        neighbors: list[dict[str, Any]] = []
        for row in rows:
            point_id = str(row.get("point_id") or "").strip()
            if not point_id:
                continue
            neighbors.append(
                _hydrate_chunk_candidate(
                    point_id=point_id,
                    text=str(row.get("text") or ""),
                    payload_json=row.get("payload_json"),
                    score=0.0,
                )
            )
        return neighbors

    async def close(self) -> None:
        """Giữ shared driver sống hết vòng đời process; chỉ đóng qua close_shared_driver."""

        return None

    @classmethod
    async def close_shared_driver(cls) -> None:
        """Đóng Neo4j shared driver an toàn khi FastAPI process shutdown."""

        driver = cls._shared_driver
        cls._shared_driver = None
        cls._shared_config = None
        if driver is not None:
            await driver.close()


__all__ = [
    "GraphSearchResult",
    "Neo4jGraphStore",
]
