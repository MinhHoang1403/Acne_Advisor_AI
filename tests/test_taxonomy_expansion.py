from __future__ import annotations

from src.knowledge.entity_cards import build_entity_cards_from_taxonomy
from src.knowledge.entity_identity import entity_point_id
from src.knowledge.graph_schema import (
    CANONICAL_RELATIONSHIP_SCHEMAS,
    ENTITY_TYPE_TO_LABEL,
    build_entity_graph_records,
)
from src.knowledge.normalizer import (
    CANONICAL_CORPUS_SOURCE_IDS,
    CURATED_TAXONOMY_SOURCE_IDS,
    DEFAULT_TAXONOMY_PATH,
    DrugEntityNormalizer,
)


def _cards():
    return build_entity_cards_from_taxonomy()


def _node_identities(records: dict[str, list[dict]]) -> set[tuple[str, str]]:
    return {(node["label"], node["canonical_name"]) for node in records["nodes"]}


def _relationship_identities(records: dict[str, list[dict]]) -> set[tuple[str, str, str, str, str]]:
    return {
        (
            relation["source_label"],
            relation["source_name"],
            relation["relationship"],
            relation["target_label"],
            relation["target_name"],
        )
        for relation in records["relationships"]
    }


def test_active_taxonomy_inventory_is_unique_and_matches_yaml_sections() -> None:
    normalizer = DrugEntityNormalizer()
    cards = _cards()
    raw = DrugEntityNormalizer._load_taxonomy(DEFAULT_TAXONOMY_PATH)
    expected_count = sum(
        len(raw.get(section, {}) or {})
        for section in (
            "drug_products",
            "active_ingredients",
            "drug_classes",
            "conditions",
            "safety_contexts",
        )
    )

    assert len(cards) == expected_count
    assert len({card.stable_id() for card in cards}) == len(cards)
    assert len({(card.entity_type, card.canonical_name) for card in cards}) == len(cards)
    assert all(card.canonical_name and card.entity_type for card in cards)
    assert normalizer.incompatible_alias_collisions() == {}


def test_aliases_do_not_ambiguously_canonicalize_bp_or_generic_retinoid() -> None:
    normalizer = DrugEntityNormalizer()

    for alias in ("benzoyl peroxide", "BP", "BPO"):
        matches = normalizer.normalize_mention(alias)
        assert [(card.entity_type, card.canonical_name) for card in matches] == [
            ("active_ingredient", "benzoyl_peroxide")
        ]

    assert normalizer.normalize_mention("retinoid") == []


def test_source_supported_p1_entities_and_brand_aliases_resolve() -> None:
    normalizer = DrugEntityNormalizer()
    active_entities = {
        "salicylic acid": "salicylic_acid",
        "clascoterone": "clascoterone",
        "minocycline": "minocycline",
        "sarecycline": "sarecycline",
        "erythromycin": "erythromycin",
        "spironolactone": "spironolactone",
        "trifarotene": "trifarotene",
    }
    brand_aliases = {
        "Arazlo": "tazarotene",
        "Avage": "tazarotene",
        "Fabior": "tazarotene",
        "Retin-A": "tretinoin",
        "Aklief": "trifarotene",
    }

    for mention, canonical_name in active_entities.items():
        matches = normalizer.normalize_mention(mention)
        assert ("active_ingredient", canonical_name) in {
            (card.entity_type, card.canonical_name) for card in matches
        }

    for mention, canonical_name in brand_aliases.items():
        matches = normalizer.normalize_mention(mention)
        assert ("active_ingredient", canonical_name) in {
            (card.entity_type, card.canonical_name) for card in matches
        }

    coc_matches = normalizer.normalize_mention("combined oral contraceptive")
    assert [(card.entity_type, card.canonical_name) for card in coc_matches] == [
        ("drug_class", "combined_oral_contraceptive")
    ]


def test_entity_card_provenance_uses_only_canonical_or_curated_ids() -> None:
    valid_source_ids = CANONICAL_CORPUS_SOURCE_IDS | CURATED_TAXONOMY_SOURCE_IDS

    for card in _cards():
        assert card.source_ids
        assert set(card.source_ids) <= valid_source_ids


def test_graph_and_entity_qdrant_identity_parity_is_exact() -> None:
    cards = _cards()
    records = build_entity_graph_records(cards)
    expected_nodes = {
        (ENTITY_TYPE_TO_LABEL[card.entity_type], card.canonical_name)
        for card in cards
    }
    assert _node_identities(records) == expected_nodes
    assert len({card.stable_id() for card in cards}) == len(cards)
    assert len({entity_point_id(card) for card in cards}) == len(cards)


def test_graph_edges_have_valid_endpoints_directions_and_provenance() -> None:
    records = build_entity_graph_records(_cards())
    nodes = _node_identities(records)
    valid_source_ids = CANONICAL_CORPUS_SOURCE_IDS | CURATED_TAXONOMY_SOURCE_IDS

    for relation in records["relationships"]:
        assert (relation["source_label"], relation["source_name"]) in nodes
        assert (relation["target_label"], relation["target_name"]) in nodes
        schema = CANONICAL_RELATIONSHIP_SCHEMAS[relation["relationship"]]
        assert relation["source_label"] in schema.source_labels
        assert relation["target_label"] in schema.target_labels
        assert relation["properties"]["source"] == "taxonomy"
        assert relation["properties"]["source_ids"]
        assert set(relation["properties"]["source_ids"]) <= valid_source_ids


def test_classless_actives_and_safety_edges_are_explicitly_curated() -> None:
    cards = _cards()
    classless = {
        card.canonical_name
        for card in cards
        if card.entity_type == "active_ingredient" and not card.drug_class
    }
    relationships = _relationship_identities(build_entity_graph_records(cards))

    assert classless == {"azelaic_acid", "benzoyl_peroxide"}
    assert (
        "ActiveIngredient",
        "tazarotene",
        "CONTRAINDICATED_IN",
        "SafetyContext",
        "pregnancy",
    ) in relationships
    assert (
        "ActiveIngredient",
        "isotretinoin",
        "CONTRAINDICATED_IN",
        "SafetyContext",
        "pregnancy",
    ) in relationships


def test_expanded_taxonomy_graph_is_reproducible_and_preserves_tazorac_chain() -> None:
    first = build_entity_graph_records(_cards())
    second = build_entity_graph_records(_cards())
    relationships = _relationship_identities(first)

    assert first == second
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
