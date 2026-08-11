"""Deterministic Neo4j schema and record builder for taxonomy entity graphs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from src.knowledge.schemas import EntityCard
from src.knowledge.versioning import get_knowledge_versions


ENTITY_TYPE_TO_LABEL = {
    "drug_product": "DrugProduct",
    "active_ingredient": "ActiveIngredient",
    "drug_class": "DrugClass",
    "condition": "Condition",
    "safety_context": "SafetyContext",
}

ENTITY_GRAPH_LABELS = (
    "DrugProduct",
    "ActiveIngredient",
    "DrugClass",
    "Condition",
    "SafetyContext",
    "SideEffect",
)

ENTITY_GRAPH_RELATIONSHIPS = (
    "HAS_ACTIVE_INGREDIENT",
    "BELONGS_TO_CLASS",
    "USED_FOR",
    "HAS_SIDE_EFFECT",
    "CONTRAINDICATED_IN",
)

RELATIONSHIP_PROPERTIES = {
    "source": "taxonomy",
    "confidence": 1.0,
    "created_by": "entity_graph_builder",
}

NEO4J_SCHEMA_VERSION = "neo4j_schema_v1"

CANONICAL_ENTITY_GRAPH_LABELS = (
    "ActiveIngredient",
    "Condition",
    "DrugClass",
    "DrugProduct",
    "SafetyContext",
)

CANONICAL_ENTITY_GRAPH_RELATIONSHIPS = (
    "BELONGS_TO_CLASS",
    "CONTRAINDICATED_IN",
    "HAS_ACTIVE_INGREDIENT",
)

LEGACY_GRAPH_PROPERTIES = ("name", "description", "evidence")


@dataclass(frozen=True)
class NodeSchema:
    label: str
    required_properties: frozenset[str]
    optional_properties: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RelationshipSchema:
    relationship_type: str
    source_labels: frozenset[str]
    target_labels: frozenset[str]
    required_properties: frozenset[str]
    optional_properties: frozenset[str] = frozenset()


_CANONICAL_NODE_REQUIRED_PROPERTIES = frozenset(
    {
        "aliases",
        "canonical_name",
        "entity_schema_version",
        "entity_type",
        "kb_version",
        "metadata_json",
        "source_ids",
        "taxonomy_version",
    }
)
_CANONICAL_NODE_OPTIONAL_PROPERTIES = frozenset({"created_at", "entity_id", "updated_at"})
_CANONICAL_REL_REQUIRED_PROPERTIES = frozenset(
    {
        "confidence",
        "created_by",
        "kb_version",
        "source",
        "source_ids",
        "taxonomy_version",
    }
)
_CANONICAL_REL_OPTIONAL_PROPERTIES = frozenset({"created_at", "updated_at"})

CANONICAL_NODE_SCHEMAS: dict[str, NodeSchema] = {
    label: NodeSchema(
        label=label,
        required_properties=_CANONICAL_NODE_REQUIRED_PROPERTIES,
        optional_properties=_CANONICAL_NODE_OPTIONAL_PROPERTIES,
    )
    for label in CANONICAL_ENTITY_GRAPH_LABELS
}

CANONICAL_RELATIONSHIP_SCHEMAS: dict[str, RelationshipSchema] = {
    "HAS_ACTIVE_INGREDIENT": RelationshipSchema(
        relationship_type="HAS_ACTIVE_INGREDIENT",
        source_labels=frozenset({"DrugProduct"}),
        target_labels=frozenset({"ActiveIngredient"}),
        required_properties=_CANONICAL_REL_REQUIRED_PROPERTIES,
        optional_properties=_CANONICAL_REL_OPTIONAL_PROPERTIES,
    ),
    "BELONGS_TO_CLASS": RelationshipSchema(
        relationship_type="BELONGS_TO_CLASS",
        source_labels=frozenset({"DrugProduct", "ActiveIngredient", "DrugClass"}),
        target_labels=frozenset({"DrugClass"}),
        required_properties=_CANONICAL_REL_REQUIRED_PROPERTIES,
        optional_properties=_CANONICAL_REL_OPTIONAL_PROPERTIES,
    ),
    "CONTRAINDICATED_IN": RelationshipSchema(
        relationship_type="CONTRAINDICATED_IN",
        source_labels=frozenset({"ActiveIngredient"}),
        target_labels=frozenset({"SafetyContext"}),
        required_properties=_CANONICAL_REL_REQUIRED_PROPERTIES,
        optional_properties=_CANONICAL_REL_OPTIONAL_PROPERTIES,
    ),
}


def get_entity_graph_constraints() -> list[str]:
    """Return idempotent Neo4j uniqueness constraints for entity graph labels."""

    return [
        (
            f"CREATE CONSTRAINT {label.lower()}_canonical_name_unique IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.canonical_name IS UNIQUE"
        )
        for label in ENTITY_GRAPH_LABELS
    ]


def get_entity_graph_indexes() -> list[str]:
    """Return minimal lookup indexes for deterministic entity graph nodes."""

    indexes: list[str] = []
    for label in ENTITY_GRAPH_LABELS:
        indexes.append(
            f"CREATE INDEX {label.lower()}_entity_id IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.entity_id)"
        )
        indexes.append(
            f"CREATE INDEX {label.lower()}_kb_version IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.kb_version)"
        )
    return indexes


def build_entity_graph_records(
    cards: list[EntityCard],
    kb_version: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build deterministic Neo4j node/relationship records from entity cards."""

    versions = get_knowledge_versions()
    resolved_kb_version = kb_version or versions["kb_version"]
    cards_by_key = {
        _entity_key(ENTITY_TYPE_TO_LABEL[card.entity_type], card.canonical_name): card
        for card in cards
        if card.entity_type in ENTITY_TYPE_TO_LABEL
    }

    nodes: dict[tuple[str, str], dict[str, Any]] = {}
    relationships: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    def add_node(
        label: str,
        canonical_name: str,
        *,
        entity_type: str | None = None,
        card: EntityCard | None = None,
        aliases: list[str] | None = None,
        source_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        canonical_name = str(canonical_name).strip()
        if not canonical_name:
            return
        key = (label, canonical_name)
        if key in nodes:
            return
        card_entity_id = card.stable_id(resolved_kb_version) if card else None
        nodes[key] = {
            "label": label,
            "canonical_name": canonical_name,
            "entity_type": entity_type or _label_to_entity_type(label),
            "aliases": _dedupe(card.aliases if card else (aliases or [])),
            "entity_id": card_entity_id,
            "taxonomy_version": card.taxonomy_version if card else versions["taxonomy_version"],
            "entity_schema_version": (
                card.entity_schema_version if card else versions["entity_schema_version"]
            ),
            "kb_version": resolved_kb_version,
            "source_ids": _dedupe(card.source_ids if card else (source_ids or [])),
            "metadata": dict(card.metadata if card else (metadata or {})),
        }

    def add_relationship(
        source_label: str,
        source_name: str,
        relationship: str,
        target_label: str,
        target_name: str,
    ) -> None:
        source_name = str(source_name).strip()
        target_name = str(target_name).strip()
        if not source_name or not target_name:
            return
        if relationship not in ENTITY_GRAPH_RELATIONSHIPS:
            return
        key = (source_label, source_name, relationship, target_label, target_name)
        if key in relationships:
            return
        relationships[key] = {
            "source_label": source_label,
            "source_name": source_name,
            "relationship": relationship,
            "target_label": target_label,
            "target_name": target_name,
            "properties": {
                **RELATIONSHIP_PROPERTIES,
                "kb_version": resolved_kb_version,
                "taxonomy_version": versions["taxonomy_version"],
                "source_ids": _relationship_source_ids(
                    cards_by_key.get(_entity_key(source_label, source_name)),
                    cards_by_key.get(_entity_key(target_label, target_name)),
                ),
            },
        }

    for card in cards:
        label = ENTITY_TYPE_TO_LABEL.get(card.entity_type)
        if not label:
            continue
        add_node(label, card.canonical_name, entity_type=card.entity_type, card=card)

    for card in cards:
        label = ENTITY_TYPE_TO_LABEL.get(card.entity_type)
        if not label:
            continue

        if card.entity_type == "drug_product":
            for ingredient in card.active_ingredients:
                target_card = cards_by_key.get(_entity_key("ActiveIngredient", ingredient))
                add_node("ActiveIngredient", ingredient, card=target_card)
                add_relationship(
                    "DrugProduct",
                    card.canonical_name,
                    "HAS_ACTIVE_INGREDIENT",
                    "ActiveIngredient",
                    ingredient,
                )
            for class_name in card.drug_class:
                target_card = cards_by_key.get(_entity_key("DrugClass", class_name))
                add_node("DrugClass", class_name, card=target_card)
                add_relationship(
                    "DrugProduct",
                    card.canonical_name,
                    "BELONGS_TO_CLASS",
                    "DrugClass",
                    class_name,
                )

        if card.entity_type in {"active_ingredient", "drug_class"}:
            for class_name in card.drug_class:
                target_card = cards_by_key.get(_entity_key("DrugClass", class_name))
                add_node("DrugClass", class_name, card=target_card)
                add_relationship(
                    label,
                    card.canonical_name,
                    "BELONGS_TO_CLASS",
                    "DrugClass",
                    class_name,
                )

        for condition in card.used_for:
            target_card = cards_by_key.get(_entity_key("Condition", condition))
            add_node("Condition", condition, card=target_card)
            add_relationship(label, card.canonical_name, "USED_FOR", "Condition", condition)

        for side_effect in card.side_effects:
            add_node("SideEffect", side_effect)
            add_relationship(label, card.canonical_name, "HAS_SIDE_EFFECT", "SideEffect", side_effect)

        for contraindication in card.contraindications:
            target_card = cards_by_key.get(_entity_key("SafetyContext", contraindication))
            add_node("SafetyContext", contraindication, card=target_card)
            add_relationship(
                label,
                card.canonical_name,
                "CONTRAINDICATED_IN",
                "SafetyContext",
                contraindication,
            )

    return {
        "nodes": list(nodes.values()),
        "relationships": list(relationships.values()),
    }


def summarize_graph_records(records: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    """Return count summaries for dry-run output."""

    return {
        "nodes_by_label": dict(
            sorted(Counter(node["label"] for node in records["nodes"]).items())
        ),
        "relationships_by_type": dict(
            sorted(Counter(rel["relationship"] for rel in records["relationships"]).items())
        ),
    }


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _relationship_source_ids(
    source_card: EntityCard | None,
    target_card: EntityCard | None,
) -> list[str]:
    """Keep deterministic edge provenance traceable to curated endpoints."""

    source_ids = [
        *(source_card.source_ids if source_card else []),
        *(target_card.source_ids if target_card else []),
    ]
    return _dedupe(source_ids)


def _entity_key(label: str, canonical_name: str) -> str:
    return f"{label}:{canonical_name}".lower()


def _label_to_entity_type(label: str) -> str:
    for entity_type, mapped_label in ENTITY_TYPE_TO_LABEL.items():
        if mapped_label == label:
            return entity_type
    return label.lower()


__all__ = [
    "ENTITY_GRAPH_LABELS",
    "ENTITY_GRAPH_RELATIONSHIPS",
    "ENTITY_TYPE_TO_LABEL",
    "CANONICAL_ENTITY_GRAPH_LABELS",
    "CANONICAL_ENTITY_GRAPH_RELATIONSHIPS",
    "CANONICAL_NODE_SCHEMAS",
    "CANONICAL_RELATIONSHIP_SCHEMAS",
    "LEGACY_GRAPH_PROPERTIES",
    "NEO4J_SCHEMA_VERSION",
    "NodeSchema",
    "RelationshipSchema",
    "build_entity_graph_records",
    "get_entity_graph_constraints",
    "get_entity_graph_indexes",
    "summarize_graph_records",
]
