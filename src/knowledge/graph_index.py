"""Ghi và đối chiếu taxonomy/entity graph deterministic trong Neo4j.

Graph mô tả quan hệ cấu trúc giữa product, active ingredient và drug class. Normal
chat retrieval không query graph này để grounding answer; evidence runtime đến từ
Qdrant knowledge chunks. Module chỉ phục vụ build/activation/validation và có
side effect lên Neo4j khi operator gọi rõ ràng.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from typing import Any

from src.knowledge.graph_schema import (
    ENTITY_GRAPH_LABELS,
    ENTITY_GRAPH_RELATIONSHIPS,
    get_entity_graph_constraints,
    get_entity_graph_indexes,
)


logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

PrimitiveNeo4jValue = str | int | float | bool


def get_neo4j_driver() -> Any:
    """Create an async Neo4j driver using the project's env config."""

    try:
        from neo4j import AsyncGraphDatabase  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("Missing dependency. Run: pip install neo4j") from exc

    return AsyncGraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )


async def apply_entity_graph_schema(driver: Any) -> None:
    """Apply deterministic entity graph constraints and indexes."""

    async with driver.session() as session:
        for statement in get_entity_graph_constraints():
            await session.run(statement)
        for statement in get_entity_graph_indexes():
            await session.run(statement)


async def upsert_entity_graph(driver: Any, records: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    """MERGE node/relationship deterministic sau khi flatten Neo4j properties."""

    node_count = 0
    relationship_count = 0

    async with driver.session() as session:
        for node in records.get("nodes", []):
            label = _safe_label(node["label"])
            properties = sanitize_neo4j_properties(
                {key: value for key, value in node.items() if key != "label"}
            )
            result = await session.run(
                (
                    f"MERGE (n:{label} {{canonical_name: $canonical_name}}) "
                    "ON CREATE SET n.created_at = datetime() "
                    "SET n += $properties, "
                    "n.updated_at = datetime() "
                    "RETURN count(n) AS materialized_count"
                ),
                canonical_name=node["canonical_name"],
                properties=properties,
            )
            materialized = _record_count(await _result_single(result), "materialized_count")
            if materialized != 1:
                raise RuntimeError(
                    "Entity graph node did not materialize: "
                    f"{label}:{node['canonical_name']} (count={materialized})"
                )
            node_count += materialized

        for relationship in records.get("relationships", []):
            source_label = _safe_label(relationship["source_label"])
            target_label = _safe_label(relationship["target_label"])
            rel_type = _safe_relationship(relationship["relationship"])
            properties = sanitize_neo4j_properties(
                dict(relationship.get("properties") or {})
            )
            result = await session.run(
                (
                    f"MATCH (src:{source_label} {{canonical_name: $source_name}}) "
                    f"MATCH (tgt:{target_label} {{canonical_name: $target_name}}) "
                    f"MERGE (src)-[r:{rel_type}]->(tgt) "
                    "ON CREATE SET r.created_at = datetime() "
                    "SET r += $properties, "
                    "r.updated_at = datetime() "
                    "RETURN count(src) AS source_count, count(tgt) AS target_count, "
                    "count(r) AS materialized_count"
                ),
                source_name=relationship["source_name"],
                target_name=relationship["target_name"],
                properties=properties,
            )
            record = await _result_single(result)
            source_count = _record_count(record, "source_count")
            target_count = _record_count(record, "target_count")
            materialized = _record_count(record, "materialized_count")
            if source_count != 1 or target_count != 1 or materialized != 1:
                raise RuntimeError(
                    "Entity graph relationship did not materialize: "
                    f"{source_label}:{relationship['source_name']}-[{rel_type}]->"
                    f"{target_label}:{relationship['target_name']} "
                    f"(source={source_count}, target={target_count}, relationship={materialized})"
                )
            relationship_count += materialized

    logger.info(
        "Entity graph upsert complete: nodes=%d relationships=%d",
        node_count,
        relationship_count,
    )
    return {"nodes": node_count, "relationships": relationship_count}


async def replace_entity_graph(
    driver: Any,
    records: dict[str, list[dict[str, Any]]],
    *,
    build_id: str,
) -> dict[str, Any]:
    """Materialize một build rồi xóa canonical records thuộc build ID khác."""

    await apply_entity_graph_schema(driver)
    materialized = await upsert_entity_graph(driver, records)
    async with driver.session() as session:
        relationship_count_result = await session.run(
            "MATCH ()-[r]->() "
            "WHERE type(r) IN $relationship_types "
            "AND coalesce(r.kb_version, '') <> $build_id "
            "RETURN count(r) AS removed",
            relationship_types=list(ENTITY_GRAPH_RELATIONSHIPS),
            build_id=build_id,
        )
        stale_relationships_removed = _record_count(
            await _result_single(relationship_count_result), "removed"
        )
        relationship_delete_result = await session.run(
            "MATCH ()-[r]->() "
            "WHERE type(r) IN $relationship_types "
            "AND coalesce(r.kb_version, '') <> $build_id "
            "DELETE r",
            relationship_types=list(ENTITY_GRAPH_RELATIONSHIPS),
            build_id=build_id,
        )
        await relationship_delete_result.consume()
        count_result = await session.run(
            "MATCH (n) "
            "WHERE any(label IN labels(n) WHERE label IN $labels) "
            "AND coalesce(n.kb_version, '') <> $build_id "
            "RETURN count(n) AS removed",
            labels=list(ENTITY_GRAPH_LABELS),
            build_id=build_id,
        )
        removed = _record_count(await _result_single(count_result), "removed")
        delete_result = await session.run(
            "MATCH (n) "
            "WHERE any(label IN labels(n) WHERE label IN $labels) "
            "AND coalesce(n.kb_version, '') <> $build_id "
            "DETACH DELETE n",
            labels=list(ENTITY_GRAPH_LABELS),
            build_id=build_id,
        )
        await delete_result.consume()
    validation = await validate_entity_graph_records(driver, records)
    if not validation["passed"]:
        raise RuntimeError(f"Canonical graph reconciliation failed: {validation['errors']}")
    return {
        **materialized,
        "stale_relationships_removed": stale_relationships_removed,
        "stale_nodes_removed": removed,
        "validation": validation,
    }


async def validate_entity_graph(driver: Any) -> dict[str, Any]:
    """Validate minimal deterministic relationships in Neo4j."""

    required_checks = {
        "clindamycin_topical_antibiotic": (
            "MATCH (:ActiveIngredient {canonical_name:'clindamycin'})"
            "-[:BELONGS_TO_CLASS]->"
            "(:DrugClass {canonical_name:'topical_antibiotic'}) RETURN count(*) AS count"
        ),
        "epiduo_has_adapalene": (
            "MATCH (:DrugProduct {canonical_name:'Epiduo'})"
            "-[:HAS_ACTIVE_INGREDIENT]->"
            "(:ActiveIngredient {canonical_name:'adapalene'}) RETURN count(*) AS count"
        ),
        "epiduo_has_bpo": (
            "MATCH (:DrugProduct {canonical_name:'Epiduo'})"
            "-[:HAS_ACTIVE_INGREDIENT]->"
            "(:ActiveIngredient {canonical_name:'benzoyl_peroxide'}) RETURN count(*) AS count"
        ),
        "differin_has_adapalene": (
            "MATCH (:DrugProduct {canonical_name:'Differin'})"
            "-[:HAS_ACTIVE_INGREDIENT]->"
            "(:ActiveIngredient {canonical_name:'adapalene'}) RETURN count(*) AS count"
        ),
        "tazorac_has_tazarotene": (
            "MATCH (:DrugProduct {canonical_name:'Tazorac'})"
            "-[:HAS_ACTIVE_INGREDIENT]->"
            "(:ActiveIngredient {canonical_name:'tazarotene'}) RETURN count(*) AS count"
        ),
        "tazarotene_topical_retinoid": (
            "MATCH (:ActiveIngredient {canonical_name:'tazarotene'})"
            "-[:BELONGS_TO_CLASS]->"
            "(:DrugClass {canonical_name:'topical_retinoid'}) RETURN count(*) AS count"
        ),
        "bpo_not_topical_or_oral_antibiotic": (
            "MATCH (:ActiveIngredient {canonical_name:'benzoyl_peroxide'})"
            "-[:BELONGS_TO_CLASS]->"
            "(c:DrugClass) "
            "WHERE c.canonical_name IN ['topical_antibiotic', 'oral_antibiotic'] "
            "RETURN count(*) AS count"
        ),
    }

    results: dict[str, Any] = {"checks": {}, "passed": True}
    async with driver.session() as session:
        for name, cypher in required_checks.items():
            result = await session.run(cypher)
            record = await result.single()
            count = int(record["count"]) if record else 0
            if name == "bpo_not_topical_or_oral_antibiotic":
                passed = count == 0
            else:
                passed = count > 0
            results["checks"][name] = {"count": count, "passed": passed}
            results["passed"] = results["passed"] and passed

        label_counts: dict[str, int] = {}
        for label in ENTITY_GRAPH_LABELS:
            result = await session.run(f"MATCH (n:{label}) RETURN count(n) AS count")
            record = await result.single()
            label_counts[label] = int(record["count"]) if record else 0
        results["nodes_by_label"] = label_counts

        relationship_counts: dict[str, int] = {}
        for rel_type in ENTITY_GRAPH_RELATIONSHIPS:
            result = await session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count")
            record = await result.single()
            relationship_counts[rel_type] = int(record["count"]) if record else 0
        results["relationships_by_type"] = relationship_counts

    return results


async def validate_entity_graph_records(
    driver: Any,
    records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Đối chiếu từng record và count với graph deterministic mong đợi."""

    expected_nodes = records.get("nodes", [])
    expected_relationships = records.get("relationships", [])
    node_counts = Counter(str(node["label"]) for node in expected_nodes)
    relationship_counts = Counter(
        str(relationship["relationship"]) for relationship in expected_relationships
    )
    expected_nodes_by_label = {
        label: node_counts.get(label, 0)
        for label in ENTITY_GRAPH_LABELS
    }
    expected_relationships_by_type = {
        rel_type: relationship_counts.get(rel_type, 0)
        for rel_type in ENTITY_GRAPH_RELATIONSHIPS
    }
    results: dict[str, Any] = {
        "passed": True,
        "expected_nodes": len(expected_nodes),
        "expected_relationships": len(expected_relationships),
        "nodes_by_label": expected_nodes_by_label,
        "relationships_by_type": expected_relationships_by_type,
        "actual_nodes_by_label": {},
        "actual_relationships_by_type": {},
        "node_checks": [],
        "relationship_checks": [],
        "errors": [],
    }

    async with driver.session() as session:
        for node in expected_nodes:
            label = _safe_label(node["label"])
            result = await session.run(
                f"MATCH (n:{label} {{canonical_name: $canonical_name}}) "
                "RETURN count(n) AS count",
                canonical_name=node["canonical_name"],
            )
            count = _record_count(await _result_single(result), "count")
            passed = count == 1
            check = {
                "label": label,
                "canonical_name": node["canonical_name"],
                "count": count,
                "passed": passed,
            }
            results["node_checks"].append(check)
            if not passed:
                results["errors"].append(
                    f"expected exactly one node {label}:{node['canonical_name']}, got {count}"
                )

        for relationship in expected_relationships:
            source_label = _safe_label(relationship["source_label"])
            target_label = _safe_label(relationship["target_label"])
            rel_type = _safe_relationship(relationship["relationship"])
            result = await session.run(
                (
                    f"MATCH (src:{source_label} {{canonical_name: $source_name}}) "
                    f"MATCH (tgt:{target_label} {{canonical_name: $target_name}}) "
                    f"MATCH (src)-[r:{rel_type}]->(tgt) "
                    "RETURN count(r) AS count"
                ),
                source_name=relationship["source_name"],
                target_name=relationship["target_name"],
            )
            count = _record_count(await _result_single(result), "count")
            passed = count == 1
            check = {
                "source": f"{source_label}:{relationship['source_name']}",
                "relationship": rel_type,
                "target": f"{target_label}:{relationship['target_name']}",
                "count": count,
                "passed": passed,
            }
            results["relationship_checks"].append(check)
            if not passed:
                results["errors"].append(
                    "expected exactly one relationship "
                    f"{check['source']}-[{rel_type}]->{check['target']}, got {count}"
                )

        for label, expected_count in sorted(expected_nodes_by_label.items()):
            result = await session.run(f"MATCH (n:{_safe_label(label)}) RETURN count(n) AS count")
            actual_count = _record_count(await _result_single(result), "count")
            results["actual_nodes_by_label"][label] = actual_count
            if actual_count != expected_count:
                results["errors"].append(
                    f"node count mismatch for {label}: expected {expected_count}, got {actual_count}"
                )

        for rel_type, expected_count in sorted(expected_relationships_by_type.items()):
            result = await session.run(
                f"MATCH ()-[r:{_safe_relationship(rel_type)}]->() RETURN count(r) AS count"
            )
            actual_count = _record_count(await _result_single(result), "count")
            results["actual_relationships_by_type"][rel_type] = actual_count
            if actual_count != expected_count:
                results["errors"].append(
                    "relationship count mismatch for "
                    f"{rel_type}: expected {expected_count}, got {actual_count}"
                )

    results["passed"] = not results["errors"]
    return results


async def _result_single(result: Any) -> Any:
    single = getattr(result, "single", None)
    if single is None:
        return None
    record = single()
    if hasattr(record, "__await__"):
        record = await record
    return record


def _record_count(record: Any, key: str) -> int:
    if record is None:
        return 0
    try:
        value = record.get(key, 0) if hasattr(record, "get") else record[key]
        return int(value or 0)
    except (KeyError, TypeError, ValueError):
        return 0


def sanitize_neo4j_properties(properties: dict[str, Any]) -> dict[str, Any]:
    """Chuyển properties thành dạng phẳng mà Neo4j chấp nhận.

    Primitive và list primitive giữ nguyên; ``None`` bị bỏ. Map, nested list hoặc
    object phức tạp được serialize thành deterministic JSON ở key ``*_json``
    (ví dụ ``metadata`` thành ``metadata_json``), tránh gửi nested map vào Cypher.
    """

    sanitized: dict[str, Any] = {}
    for key, value in properties.items():
        if value is None:
            continue

        if _is_primitive_neo4j_value(value):
            sanitized[key] = value
            continue

        if isinstance(value, list) and all(
            _is_primitive_neo4j_value(item) for item in value
        ):
            sanitized[key] = value
            continue

        sanitized[f"{key}_json"] = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    return sanitized


def _is_primitive_neo4j_value(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _safe_label(label: str) -> str:
    if label not in ENTITY_GRAPH_LABELS:
        raise ValueError(f"Unsupported entity graph label: {label}")
    return label


def _safe_relationship(relationship: str) -> str:
    if relationship not in ENTITY_GRAPH_RELATIONSHIPS:
        raise ValueError(f"Unsupported entity graph relationship: {relationship}")
    return relationship


__all__ = [
    "apply_entity_graph_schema",
    "get_neo4j_driver",
    "sanitize_neo4j_properties",
    "replace_entity_graph",
    "upsert_entity_graph",
    "validate_entity_graph",
    "validate_entity_graph_records",
]
