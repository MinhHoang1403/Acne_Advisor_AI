from __future__ import annotations

from pathlib import Path

import pytest

from scripts import rebuild_phase1_entity_layer as rebuild
from src.knowledge.entity_cards import build_entity_cards_from_taxonomy
from src.knowledge.entity_index import build_entity_point_payload
from src.knowledge.graph_schema import build_entity_graph_records


def _desired_identity_set() -> set[str]:
    return rebuild.identity_set_from_payloads(
        [build_entity_point_payload(card) for card in build_entity_cards_from_taxonomy()]
    )


def test_desired_entity_layer_contains_tazorac_and_regression_entities() -> None:
    identities = _desired_identity_set()
    graph = build_entity_graph_records(build_entity_cards_from_taxonomy())
    relationships = rebuild.relationship_set(graph)

    assert "drug_product:tazorac" in identities
    assert "active_ingredient:tazarotene" in identities
    assert "drug_product:differin" in identities
    assert "drug_product:epiduo" in identities
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


def test_rebuild_plan_adds_tazorac_without_knowledge_or_manifest_mutation() -> None:
    desired_graph = build_entity_graph_records(build_entity_cards_from_taxonomy())
    current_entities = _desired_identity_set() - {
        "drug_product:tazorac",
        "active_ingredient:tazarotene",
    }
    current_nodes = rebuild.graph_node_set(desired_graph) - {
        ("DrugProduct", "Tazorac"),
        ("ActiveIngredient", "tazarotene"),
    }
    current_relationships = {
        rel
        for rel in rebuild.relationship_set(desired_graph)
        if "Tazorac" not in rel and "tazarotene" not in rel
    }

    plan = rebuild.summarize_rebuild_plan(
        current_entity_identities=current_entities,
        current_graph_nodes=current_nodes,
        current_graph_relationships=current_relationships,
        knowledge_count=638,
        manifest_hash="ABC",
    )

    assert plan["qdrant_entity_count_proposed"] == 32
    assert "drug_product:tazorac" in plan["entities_added"]
    assert "active_ingredient:tazarotene" in plan["entities_added"]
    assert plan["acne_knowledge_mutation_count"] == 0
    assert plan["manifest_mutation_count"] == 0
    assert rebuild.validate_plan(plan) == []


def test_rebuild_plan_blocks_unexpected_removals() -> None:
    plan = rebuild.summarize_rebuild_plan(
        current_entity_identities=_desired_identity_set() | {"drug_product:unexpected"},
        current_graph_nodes=rebuild.graph_node_set(
            build_entity_graph_records(build_entity_cards_from_taxonomy())
        ),
        current_graph_relationships=rebuild.relationship_set(
            build_entity_graph_records(build_entity_cards_from_taxonomy())
        ),
        knowledge_count=638,
        manifest_hash="ABC",
    )

    failures = rebuild.validate_plan(plan)

    assert any("unexpected entity removals" in failure for failure in failures)


def test_backup_dir_must_be_outside_repo() -> None:
    with pytest.raises(RuntimeError, match="outside the repository"):
        rebuild.ensure_backup_dir(Path("artifacts") / "bad-backup")


def test_duplicate_detection_helpers() -> None:
    duplicates = rebuild.duplicate_entities(
        [
            {
                "id": "1",
                "payload": {
                    "entity_type": "drug_product",
                    "canonical_name": "Tazorac",
                    "metadata": {"taxonomy_key": "tazorac"},
                },
            },
            {
                "id": "2",
                "payload": {
                    "entity_type": "drug_product",
                    "canonical_name": "Tazorac",
                    "metadata": {"taxonomy_key": "tazorac"},
                },
            },
        ]
    )

    assert duplicates == {"drug_product:tazorac": ["1", "2"]}


def test_embed_cards_uses_single_item_embedding_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    cards = build_entity_cards_from_taxonomy()[:2]
    calls: list[list[str]] = []

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(rebuild, "build_google_genai_client", lambda api_key: object())

    def fake_embed_texts_sync(texts: list[str], **_: object) -> list[list[float]]:
        calls.append(texts)
        return [[0.0] * 3072]

    monkeypatch.setattr(rebuild, "embed_texts_sync", fake_embed_texts_sync)

    vectors = rebuild.embed_cards(cards)

    assert len(vectors) == len(cards)
    assert calls
    assert all(len(batch) == 1 for batch in calls)
