from __future__ import annotations

from scripts.build_entity_graph import build_dry_run_summary
from src.knowledge.entity_cards import build_entity_cards_from_taxonomy
from src.knowledge.graph_index import sanitize_neo4j_properties
from src.knowledge.graph_schema import (
    build_entity_graph_records,
    get_entity_graph_constraints,
)


def _records() -> dict[str, list[dict]]:
    return build_entity_graph_records(build_entity_cards_from_taxonomy())


def _node_set(records: dict[str, list[dict]]) -> set[tuple[str, str]]:
    return {
        (node["label"], node["canonical_name"])
        for node in records["nodes"]
    }


def _relationship_set(records: dict[str, list[dict]]) -> set[tuple[str, str, str, str, str]]:
    return {
        (
            rel["source_label"],
            rel["source_name"],
            rel["relationship"],
            rel["target_label"],
            rel["target_name"],
        )
        for rel in records["relationships"]
    }


def test_build_graph_records_required_nodes() -> None:
    nodes = _node_set(_records())

    assert ("DrugProduct", "Dalacin T") in nodes
    assert ("DrugProduct", "Epiduo") in nodes
    assert ("DrugProduct", "Differin") in nodes
    assert ("DrugProduct", "Tazorac") in nodes
    assert ("ActiveIngredient", "clindamycin") in nodes
    assert ("ActiveIngredient", "adapalene") in nodes
    assert ("ActiveIngredient", "benzoyl_peroxide") in nodes
    assert ("ActiveIngredient", "tazarotene") in nodes
    assert ("DrugClass", "topical_antibiotic") in nodes
    assert ("DrugClass", "topical_retinoid") in nodes


def test_dalacin_relationships() -> None:
    relationships = _relationship_set(_records())

    assert (
        "DrugProduct",
        "Dalacin T",
        "HAS_ACTIVE_INGREDIENT",
        "ActiveIngredient",
        "clindamycin",
    ) in relationships
    assert (
        "ActiveIngredient",
        "clindamycin",
        "BELONGS_TO_CLASS",
        "DrugClass",
        "topical_antibiotic",
    ) in relationships


def test_epiduo_relationships() -> None:
    relationships = _relationship_set(_records())

    assert (
        "DrugProduct",
        "Epiduo",
        "HAS_ACTIVE_INGREDIENT",
        "ActiveIngredient",
        "adapalene",
    ) in relationships
    assert (
        "DrugProduct",
        "Epiduo",
        "HAS_ACTIVE_INGREDIENT",
        "ActiveIngredient",
        "benzoyl_peroxide",
    ) in relationships
    assert (
        "ActiveIngredient",
        "adapalene",
        "BELONGS_TO_CLASS",
        "DrugClass",
        "topical_retinoid",
    ) in relationships
    assert (
        "ActiveIngredient",
        "benzoyl_peroxide",
        "BELONGS_TO_CLASS",
        "DrugClass",
        "topical_antibiotic",
    ) not in relationships
    assert (
        "ActiveIngredient",
        "benzoyl_peroxide",
        "BELONGS_TO_CLASS",
        "DrugClass",
        "oral_antibiotic",
    ) not in relationships


def test_differin_relationships() -> None:
    relationships = _relationship_set(_records())

    assert (
        "DrugProduct",
        "Differin",
        "HAS_ACTIVE_INGREDIENT",
        "ActiveIngredient",
        "adapalene",
    ) in relationships
    assert (
        "ActiveIngredient",
        "adapalene",
        "BELONGS_TO_CLASS",
        "DrugClass",
        "topical_retinoid",
    ) in relationships


def test_tazorac_relationships() -> None:
    relationships = _relationship_set(_records())

    assert (
        "DrugProduct",
        "Tazorac",
        "HAS_ACTIVE_INGREDIENT",
        "ActiveIngredient",
        "tazarotene",
    ) in relationships
    assert (
        "ActiveIngredient",
        "tazarotene",
        "BELONGS_TO_CLASS",
        "DrugClass",
        "topical_retinoid",
    ) in relationships


def test_constraints_have_unique_canonical_name() -> None:
    constraints = get_entity_graph_constraints()
    labels = [
        "DrugProduct",
        "ActiveIngredient",
        "DrugClass",
        "Condition",
        "SafetyContext",
        "SideEffect",
    ]

    for label in labels:
        matching = [constraint for constraint in constraints if f"(n:{label})" in constraint]
        assert matching
        assert "REQUIRE n.canonical_name IS UNIQUE" in matching[0]


def test_dry_run_build_entity_graph_no_neo4j_required() -> None:
    summary = build_dry_run_summary()

    assert summary["card_count"] == 32
    assert summary["node_count"] >= 20
    assert summary["relationship_count"] >= 1
    assert summary["nodes_by_label"]["DrugProduct"] == 4
    assert "HAS_ACTIVE_INGREDIENT" in summary["relationships_by_type"]
    preview_names = {node["canonical_name"] for node in summary["preview"]["nodes"]}
    assert {"Dalacin T", "Epiduo", "Differin", "benzoyl_peroxide"}.issubset(preview_names)


def test_sanitize_neo4j_properties_flattens_nested_maps() -> None:
    node = next(
        node
        for node in _records()["nodes"]
        if node["label"] == "DrugProduct" and node["canonical_name"] == "Differin"
    )

    sanitized = sanitize_neo4j_properties(
        {key: value for key, value in node.items() if key != "label"}
    )

    for required_field in (
        "entity_id",
        "canonical_name",
        "entity_type",
        "aliases",
        "taxonomy_version",
        "entity_schema_version",
        "kb_version",
        "source_ids",
    ):
        assert required_field in sanitized

    assert "metadata" not in sanitized
    assert "metadata_json" in sanitized
    assert '"taxonomy_key": "differin"' in sanitized["metadata_json"]

    for value in sanitized.values():
        assert not isinstance(value, dict)
        if isinstance(value, list):
            assert all(isinstance(item, (str, int, float, bool)) for item in value)


def test_sanitize_neo4j_properties_serializes_complex_relationship_property() -> None:
    sanitized = sanitize_neo4j_properties(
        {
            "source": "taxonomy",
            "confidence": 1.0,
            "aliases": ["Differin", "Epiduo"],
            "nested": {"a": ["b"]},
            "nested_list": [{"a": "b"}],
            "none_value": None,
        }
    )

    assert sanitized["source"] == "taxonomy"
    assert sanitized["confidence"] == 1.0
    assert sanitized["aliases"] == ["Differin", "Epiduo"]
    assert sanitized["nested_json"] == '{"a": ["b"]}'
    assert sanitized["nested_list_json"] == '[{"a": "b"}]'
    assert "none_value" not in sanitized
    assert "nested" not in sanitized
    assert "nested_list" not in sanitized
