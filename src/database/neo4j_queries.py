"""Canonical read-only Neo4j runtime queries and notification helpers."""

from __future__ import annotations

from typing import Any


ENTITY_CONTEXT_CYPHER = """
MATCH (n)
WHERE n.canonical_name IN $canonical_names
   OR toLower(n.canonical_name) IN $lookup_names
   OR ANY(alias IN n.aliases WHERE toLower(alias) IN $lookup_names)
OPTIONAL MATCH (n)-[r]-(m)
RETURN n.canonical_name AS entity,
       n.canonical_name AS subject,
       n.entity_id AS subject_graph_node_id,
       n.entity_type AS entity_type,
       n.metadata_json AS description,
       type(r) AS relationship,
       type(r) AS predicate,
       m.canonical_name AS related_entity,
       m.canonical_name AS object,
       CASE WHEN m IS NULL THEN null ELSE m.entity_id END AS object_graph_node_id,
       CASE WHEN m IS NULL THEN null ELSE m.entity_type END AS related_type,
       m.metadata_json AS related_description,
       CASE WHEN r IS NULL THEN null ELSE r.source END AS evidence,
       CASE WHEN r IS NULL THEN null ELSE r.source END AS evidence_source,
       CASE WHEN r IS NULL THEN null ELSE coalesce(r.confidence, 1.0) END AS confidence
LIMIT $limit
"""


KEYWORD_SEARCH_CYPHER = """
MATCH (n)
WHERE ANY(
    kw IN $keywords
    WHERE toLower(n.canonical_name) CONTAINS kw
       OR ANY(alias IN n.aliases WHERE toLower(alias) CONTAINS kw)
)
OPTIONAL MATCH (n)-[r]-(m)
RETURN n.canonical_name AS entity,
       n.canonical_name AS subject,
       n.entity_id AS subject_graph_node_id,
       n.entity_type AS entity_type,
       n.metadata_json AS description,
       type(r) AS relationship,
       type(r) AS predicate,
       m.canonical_name AS related_entity,
       m.canonical_name AS object,
       CASE WHEN m IS NULL THEN null ELSE m.entity_id END AS object_graph_node_id,
       CASE WHEN m IS NULL THEN null ELSE m.entity_type END AS related_type,
       m.metadata_json AS related_description,
       CASE WHEN r IS NULL THEN null ELSE r.source END AS evidence,
       CASE WHEN r IS NULL THEN null ELSE r.source END AS evidence_source,
       CASE WHEN r IS NULL THEN null ELSE coalesce(r.confidence, 1.0) END AS confidence
LIMIT $limit
"""


CRITICAL_NOTIFICATION_MARKERS = (
    "property key does not exist",
    "label does not exist",
    "relationship type does not exist",
    "cartesian product",
    "unbounded",
)


def extract_neo4j_notifications(summary: Any) -> list[dict[str, Any]]:
    """Return sanitized Neo4j query notifications across driver APIs."""

    raw_items = getattr(summary, "gql_status_objects", None)
    if raw_items is None:
        raw_items = getattr(summary, "notifications", None)
    notifications: list[dict[str, Any]] = []
    for item in raw_items or []:
        record = {
            "code": _string_attr(item, "gql_status") or _string_attr(item, "code"),
            "title": _string_attr(item, "title"),
            "description": _string_attr(item, "status_description")
            or _string_attr(item, "description"),
            "severity": _string_attr(item, "raw_severity")
            or _enum_or_string_attr(item, "severity"),
            "classification": _string_attr(item, "raw_classification")
            or _enum_or_string_attr(item, "classification"),
        }
        if is_critical_neo4j_notification(record):
            record["critical"] = True
        notifications.append(record)
    return notifications


def is_critical_neo4j_notification(notification: dict[str, Any]) -> bool:
    text = " ".join(
        str(notification.get(key) or "").lower()
        for key in ("code", "title", "description", "classification")
    )
    return any(marker in text for marker in CRITICAL_NOTIFICATION_MARKERS)


def _string_attr(item: Any, name: str) -> str | None:
    value = getattr(item, name, None)
    if value is None:
        return None
    return str(value)


def _enum_or_string_attr(item: Any, name: str) -> str | None:
    value = getattr(item, name, None)
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)


__all__ = [
    "ENTITY_CONTEXT_CYPHER",
    "KEYWORD_SEARCH_CYPHER",
    "extract_neo4j_notifications",
    "is_critical_neo4j_notification",
]
