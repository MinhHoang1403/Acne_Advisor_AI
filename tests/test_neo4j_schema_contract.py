from __future__ import annotations

from src.knowledge.graph_schema import (
    CANONICAL_ENTITY_GRAPH_LABELS,
    CANONICAL_ENTITY_GRAPH_RELATIONSHIPS,
    CANONICAL_NODE_SCHEMAS,
    CANONICAL_RELATIONSHIP_SCHEMAS,
)


def test_canonical_labels_and_relationships_are_unique() -> None:
    assert len(CANONICAL_ENTITY_GRAPH_LABELS) == len(set(CANONICAL_ENTITY_GRAPH_LABELS))
    assert len(CANONICAL_ENTITY_GRAPH_RELATIONSHIPS) == len(set(CANONICAL_ENTITY_GRAPH_RELATIONSHIPS))


def test_node_schema_required_optional_do_not_overlap() -> None:
    for label, schema in CANONICAL_NODE_SCHEMAS.items():
        assert schema.label == label
        assert schema.required_properties
        assert not (schema.required_properties & schema.optional_properties)
        assert "canonical_name" in schema.required_properties


def test_relationship_schema_endpoints_are_canonical() -> None:
    canonical_labels = set(CANONICAL_ENTITY_GRAPH_LABELS)
    for rel_type, schema in CANONICAL_RELATIONSHIP_SCHEMAS.items():
        assert rel_type in CANONICAL_ENTITY_GRAPH_RELATIONSHIPS
        assert schema.source_labels <= canonical_labels
        assert schema.target_labels <= canonical_labels
        assert schema.required_properties
        assert not (schema.required_properties & schema.optional_properties)

