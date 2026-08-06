from __future__ import annotations

import re

from src.database.neo4j_queries import ENTITY_CONTEXT_CYPHER, KEYWORD_SEARCH_CYPHER


def _static_property_accesses(cypher: str, property_name: str) -> list[str]:
    return re.findall(rf"\b[a-zA-Z]\.{property_name}\b", cypher)


def test_runtime_queries_do_not_use_legacy_static_properties() -> None:
    combined = f"{ENTITY_CONTEXT_CYPHER}\n{KEYWORD_SEARCH_CYPHER}"

    assert _static_property_accesses(combined, "name") == []
    assert _static_property_accesses(combined, "description") == []
    assert _static_property_accesses(combined, "evidence") == []


def test_runtime_queries_are_parameterized_and_limited() -> None:
    for cypher in (ENTITY_CONTEXT_CYPHER, KEYWORD_SEARCH_CYPHER):
        assert "$limit" in cypher
        assert "LIMIT $limit" in cypher
        assert "$canonical_names" in cypher or "$keywords" in cypher


def test_runtime_queries_project_stable_fact_contract() -> None:
    for field in (
        "entity",
        "entity_type",
        "description",
        "relationship",
        "related_entity",
        "related_type",
        "related_description",
        "evidence",
        "subject",
        "predicate",
        "object",
        "confidence",
        "evidence_source",
        "subject_graph_node_id",
        "object_graph_node_id",
    ):
        assert f" AS {field}" in ENTITY_CONTEXT_CYPHER
        assert f" AS {field}" in KEYWORD_SEARCH_CYPHER

