#!/usr/bin/env python3
"""Read-only Neo4j schema validator for the deterministic entity graph."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

from src.database.neo4j_queries import (  # noqa: E402
    ENTITY_CONTEXT_CYPHER,
    KEYWORD_SEARCH_CYPHER,
    extract_neo4j_notifications,
    is_critical_neo4j_notification,
)
from src.knowledge.graph_schema import (  # noqa: E402
    CANONICAL_ENTITY_GRAPH_LABELS,
    CANONICAL_ENTITY_GRAPH_RELATIONSHIPS,
    CANONICAL_NODE_SCHEMAS,
    CANONICAL_RELATIONSHIP_SCHEMAS,
    LEGACY_GRAPH_PROPERTIES,
)
from src.knowledge.taxonomy_models import ALLOWED_CLASSLESS_ACTIVE_INGREDIENTS  # noqa: E402


def _check(name: str, passed: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details or {}}


def validate_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    failures: list[str] = []

    labels = dict(snapshot.get("labels") or {})
    rel_types = dict(snapshot.get("relationship_types") or {})
    property_keys = dict(snapshot.get("property_keys") or {})
    rel_property_keys = dict(snapshot.get("relationship_property_keys") or {})
    properties_by_label = snapshot.get("properties_by_label") or {}
    duplicate_canonical_names = snapshot.get("duplicate_canonical_names") or []
    relationship_directions = snapshot.get("relationship_directions") or []
    constraints = snapshot.get("constraints") or []
    indexes = snapshot.get("indexes") or []
    query_notifications = snapshot.get("query_notifications") or {}

    def add(name: str, passed: bool, details: dict[str, Any] | None = None) -> None:
        checks.append(_check(name, passed, details))
        if not passed:
            failures.append(name)

    add("has_nodes", int(snapshot.get("node_count", 0) or 0) > 0, {"node_count": snapshot.get("node_count", 0)})
    add(
        "has_relationships",
        int(snapshot.get("relationship_count", 0) or 0) > 0,
        {"relationship_count": snapshot.get("relationship_count", 0)},
    )
    add(
        "canonical_labels_present",
        all(label in labels for label in CANONICAL_ENTITY_GRAPH_LABELS),
        {"labels": labels},
    )
    add(
        "canonical_relationship_types_present",
        all(rel_type in rel_types for rel_type in CANONICAL_ENTITY_GRAPH_RELATIONSHIPS),
        {"relationship_types": rel_types},
    )

    missing_properties: dict[str, list[str]] = {}
    for label, schema in CANONICAL_NODE_SCHEMAS.items():
        present = set(properties_by_label.get(label, []))
        missing = sorted(schema.required_properties - present)
        if missing:
            missing_properties[label] = missing
    add("required_node_properties_present", not missing_properties, {"missing": missing_properties})

    missing_rel_properties = sorted(
        {
            prop
            for schema in CANONICAL_RELATIONSHIP_SCHEMAS.values()
            for prop in schema.required_properties
            if prop not in rel_property_keys
        }
    )
    add(
        "required_relationship_properties_present",
        not missing_rel_properties,
        {"missing": missing_rel_properties},
    )

    legacy_found = sorted(prop for prop in LEGACY_GRAPH_PROPERTIES if prop in property_keys or prop in rel_property_keys)
    add("no_legacy_properties", not legacy_found, {"legacy_found": legacy_found})

    add("no_duplicate_canonical_names", not duplicate_canonical_names, {"duplicates": duplicate_canonical_names})
    add(
        "no_orphan_drug_products",
        int(snapshot.get("orphan_drug_products", 0) or 0) == 0,
        {"orphan_drug_products": snapshot.get("orphan_drug_products", 0)},
    )
    classless_actives = list(snapshot.get("active_ingredients_without_class") or [])
    unexpected_classless = sorted(set(classless_actives) - ALLOWED_CLASSLESS_ACTIVE_INGREDIENTS)
    add(
        "active_ingredients_have_supported_classification",
        not unexpected_classless,
        {
            "allowed_classless": sorted(ALLOWED_CLASSLESS_ACTIVE_INGREDIENTS),
            "active_ingredients_without_class": sorted(classless_actives),
            "unexpected": unexpected_classless,
        },
    )

    invalid_directions = []
    for item in relationship_directions:
        rel_type = item.get("relationship_type")
        schema = CANONICAL_RELATIONSHIP_SCHEMAS.get(str(rel_type))
        if schema is None:
            invalid_directions.append(item)
            continue
        source_labels = set(item.get("source_labels") or [])
        target_labels = set(item.get("target_labels") or [])
        if not (source_labels & schema.source_labels) or not (target_labels & schema.target_labels):
            invalid_directions.append(item)
    add("relationship_directions_valid", not invalid_directions, {"invalid": invalid_directions})

    constraint_names = {item.get("name") for item in constraints if isinstance(item, dict)}
    required_constraints = {
        f"{label.lower()}_canonical_name_unique"
        for label in CANONICAL_ENTITY_GRAPH_LABELS
    }
    missing_constraints = sorted(required_constraints - constraint_names)
    add("required_constraints_present", not missing_constraints, {"missing": missing_constraints})

    index_names = {item.get("name") for item in indexes if isinstance(item, dict)}
    required_indexes = {
        f"{label.lower()}_{suffix}"
        for label in CANONICAL_ENTITY_GRAPH_LABELS
        for suffix in ("entity_id", "kb_version")
    }
    missing_indexes = sorted(required_indexes - index_names)
    add("required_indexes_present", not missing_indexes, {"missing": missing_indexes})

    critical_notifications: list[dict[str, Any]] = []
    for query_name, notifications in query_notifications.items():
        for notification in notifications or []:
            if is_critical_neo4j_notification(notification):
                critical_notifications.append({"query": query_name, **notification})
    add("runtime_queries_without_critical_notifications", not critical_notifications, {"critical": critical_notifications})

    report = {
        "passed": not failures,
        "node_count": int(snapshot.get("node_count", 0) or 0),
        "relationship_count": int(snapshot.get("relationship_count", 0) or 0),
        "labels": labels,
        "relationship_types": rel_types,
        "property_keys": property_keys,
        "relationship_property_keys": rel_property_keys,
        "constraints": constraints,
        "indexes": indexes,
        "warnings": warnings,
        "checks": checks,
        "failures": failures,
    }
    return report


async def collect_neo4j_snapshot() -> dict[str, Any]:
    from neo4j import AsyncGraphDatabase  # type: ignore[import]

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    database = os.getenv("NEO4J_DATABASE") or None
    driver = AsyncGraphDatabase.driver(uri, auth=(username, password))
    try:
        async with driver.session(database=database) as session:
            snapshot: dict[str, Any] = {
                "node_count": await _single_count(session, "MATCH (n) RETURN count(n) AS count"),
                "relationship_count": await _single_count(session, "MATCH ()-[r]->() RETURN count(r) AS count"),
                "labels": await _key_count(session, "MATCH (n) UNWIND labels(n) AS key RETURN key, count(*) AS count ORDER BY key"),
                "relationship_types": await _key_count(session, "MATCH ()-[r]->() RETURN type(r) AS key, count(*) AS count ORDER BY key"),
                "property_keys": await _key_count(session, "MATCH (n) UNWIND keys(n) AS key RETURN key, count(*) AS count ORDER BY key"),
                "relationship_property_keys": await _key_count(session, "MATCH ()-[r]->() UNWIND keys(r) AS key RETURN key, count(*) AS count ORDER BY key"),
                "properties_by_label": await _properties_by_label(session),
                "duplicate_canonical_names": await _records(
                    session,
                    """
                    MATCH (n)
                    WHERE any(label IN labels(n) WHERE label IN $labels)
                    WITH labels(n)[0] AS label, n.canonical_name AS canonical_name, count(*) AS count
                    WHERE count > 1
                    RETURN label, canonical_name, count
                    ORDER BY label, canonical_name
                    """,
                    labels=list(CANONICAL_ENTITY_GRAPH_LABELS),
                ),
                "orphan_drug_products": await _single_count(
                    session,
                    "MATCH (n:DrugProduct) WHERE NOT (n)-[:HAS_ACTIVE_INGREDIENT]->(:ActiveIngredient) RETURN count(n) AS count",
                ),
                "active_ingredients_without_class": await _single_names(
                    session,
                    "MATCH (n:ActiveIngredient) WHERE NOT (n)-[:BELONGS_TO_CLASS]->(:DrugClass) RETURN n.canonical_name AS canonical_name ORDER BY canonical_name",
                ),
                "relationship_directions": await _records(
                    session,
                    """
                    MATCH (src)-[r]->(tgt)
                    RETURN type(r) AS relationship_type,
                           labels(src) AS source_labels,
                           labels(tgt) AS target_labels,
                           count(*) AS count
                    ORDER BY relationship_type, source_labels, target_labels
                    """,
                ),
                "constraints": await _records(
                    session,
                    "SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties RETURN name, type, entityType, labelsOrTypes, properties ORDER BY name",
                ),
                "indexes": await _records(
                    session,
                    "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, state RETURN name, type, entityType, labelsOrTypes, properties, state ORDER BY name",
                ),
                "query_notifications": await _runtime_query_notifications(session),
            }
            return snapshot
    finally:
        await driver.close()


async def _single_count(session: Any, cypher: str, **params: Any) -> int:
    result = await session.run(cypher, **params)
    record = await result.single()
    await result.consume()
    return int(record["count"]) if record else 0


async def _single_names(session: Any, cypher: str, **params: Any) -> list[str]:
    result = await session.run(cypher, **params)
    names = [str(record["canonical_name"]) async for record in result]
    await result.consume()
    return names


async def _key_count(session: Any, cypher: str, **params: Any) -> dict[str, int]:
    result = await session.run(cypher, **params)
    output = {str(record["key"]): int(record["count"]) async for record in result}
    await result.consume()
    return output


async def _records(session: Any, cypher: str, **params: Any) -> list[dict[str, Any]]:
    result = await session.run(cypher, **params)
    records = [dict(record) async for record in result]
    await result.consume()
    return records


async def _properties_by_label(session: Any) -> dict[str, list[str]]:
    records = await _records(
        session,
        """
        MATCH (n)
        UNWIND labels(n) AS label
        UNWIND keys(n) AS property_key
        RETURN label, property_key, count(*) AS count
        ORDER BY label, property_key
        """,
    )
    output: dict[str, set[str]] = {}
    for record in records:
        output.setdefault(str(record["label"]), set()).add(str(record["property_key"]))
    return {label: sorted(properties) for label, properties in sorted(output.items())}


async def _runtime_query_notifications(session: Any) -> dict[str, list[dict[str, Any]]]:
    checks = {
        "entity_context": (
            ENTITY_CONTEXT_CYPHER,
            {
                "canonical_names": ["benzoyl_peroxide"],
                "lookup_names": ["benzoyl_peroxide", "benzoyl peroxide"],
                "limit": 5,
            },
        ),
        "keyword_search": (
            KEYWORD_SEARCH_CYPHER,
            {"keywords": ["benzoyl"], "limit": 5},
        ),
    }
    output: dict[str, list[dict[str, Any]]] = {}
    for name, (cypher, params) in checks.items():
        result = await session.run(cypher, **params)
        _ = [dict(record) async for record in result]
        summary = await result.consume()
        output[name] = extract_neo4j_notifications(summary)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate deterministic Neo4j schema without writing data.")
    parser.add_argument("--offline", action="store_true", help="Validate an in-memory canonical snapshot only.")
    return parser.parse_args(argv)


def offline_snapshot() -> dict[str, Any]:
    labels = {label: 1 for label in CANONICAL_ENTITY_GRAPH_LABELS}
    return {
        "node_count": len(CANONICAL_ENTITY_GRAPH_LABELS),
        "relationship_count": len(CANONICAL_ENTITY_GRAPH_RELATIONSHIPS),
        "labels": labels,
        "relationship_types": {rel: 1 for rel in CANONICAL_ENTITY_GRAPH_RELATIONSHIPS},
        "property_keys": {
            "aliases": 1,
            "canonical_name": 1,
            "entity_schema_version": 1,
            "entity_type": 1,
            "kb_version": 1,
            "metadata_json": 1,
            "source_ids": 1,
            "taxonomy_version": 1,
        },
        "relationship_property_keys": {
            "confidence": 1,
            "created_by": 1,
            "kb_version": 1,
            "source": 1,
            "source_ids": 1,
            "taxonomy_version": 1,
        },
        "properties_by_label": {
            label: sorted(CANONICAL_NODE_SCHEMAS[label].required_properties)
            for label in CANONICAL_ENTITY_GRAPH_LABELS
        },
        "duplicate_canonical_names": [],
        "orphan_drug_products": 0,
        "active_ingredients_without_class": sorted(ALLOWED_CLASSLESS_ACTIVE_INGREDIENTS),
        "relationship_directions": [
            {
                "relationship_type": "HAS_ACTIVE_INGREDIENT",
                "source_labels": ["DrugProduct"],
                "target_labels": ["ActiveIngredient"],
                "count": 1,
            },
            {
                "relationship_type": "BELONGS_TO_CLASS",
                "source_labels": ["ActiveIngredient"],
                "target_labels": ["DrugClass"],
                "count": 1,
            },
            {
                "relationship_type": "CONTRAINDICATED_IN",
                "source_labels": ["ActiveIngredient"],
                "target_labels": ["SafetyContext"],
                "count": 1,
            },
        ],
        "constraints": [
            {"name": f"{label.lower()}_canonical_name_unique"}
            for label in CANONICAL_ENTITY_GRAPH_LABELS
        ],
        "indexes": [
            {"name": f"{label.lower()}_{suffix}"}
            for label in CANONICAL_ENTITY_GRAPH_LABELS
            for suffix in ("entity_id", "kb_version")
        ],
        "query_notifications": {"entity_context": [], "keyword_search": []},
    }


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        snapshot = offline_snapshot() if args.offline else await collect_neo4j_snapshot()
        report = validate_snapshot(snapshot)
    except Exception as exc:
        report = {
            "passed": False,
            "node_count": 0,
            "relationship_count": 0,
            "labels": {},
            "relationship_types": {},
            "property_keys": {},
            "constraints": [],
            "indexes": [],
            "warnings": [],
            "checks": [_check("neo4j_reachable", False, {"error": str(exc)})],
            "failures": ["neo4j_reachable"],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
