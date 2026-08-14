from __future__ import annotations

from src.knowledge.entity_cards import build_entity_cards_from_taxonomy, entity_card_to_text
from src.knowledge.entity_identity import entity_identity_key, entity_point_id


def _find_card(entity_type: str, canonical_name: str):
    for card in build_entity_cards_from_taxonomy():
        if card.entity_type == entity_type and card.canonical_name == canonical_name:
            return card
    raise AssertionError(f"Missing card {entity_type}:{canonical_name}")


def test_build_entity_cards_contains_required_products() -> None:
    product_names = {
        card.canonical_name
        for card in build_entity_cards_from_taxonomy()
        if card.entity_type == "drug_product"
    }

    assert {"Differin", "Epiduo", "Dalacin T", "Tazorac"}.issubset(product_names)


def test_dalacin_card_payload() -> None:
    card = _find_card("drug_product", "Dalacin T")

    assert card.canonical_name == "Dalacin T"
    assert "clindamycin" in card.active_ingredients
    assert "topical_antibiotic" in card.drug_class


def test_epiduo_card_payload() -> None:
    card = _find_card("drug_product", "Epiduo")

    assert "adapalene" in card.active_ingredients
    assert "benzoyl_peroxide" in card.active_ingredients


def test_differin_card_payload() -> None:
    card = _find_card("drug_product", "Differin")

    assert "adapalene" in card.active_ingredients
    assert "topical_retinoid" in card.drug_class


def test_tazorac_card_payload() -> None:
    product = _find_card("drug_product", "Tazorac")
    ingredient = _find_card("active_ingredient", "tazarotene")

    assert "tazarotene" in product.active_ingredients
    assert "topical_retinoid" in product.drug_class
    assert "tazaroten" in ingredient.aliases
    assert "topical_retinoid" in ingredient.drug_class


def test_benzoyl_peroxide_entity_not_antibiotic() -> None:
    card = _find_card("active_ingredient", "benzoyl_peroxide")
    text = entity_card_to_text(card).lower()

    assert "topical_antibiotic" not in card.drug_class
    assert "oral_antibiotic" not in card.drug_class
    assert card.metadata["not_antibiotic"] is True
    assert "not an antibiotic" in text


def test_entity_stable_id_deterministic() -> None:
    card = _find_card("drug_product", "Epiduo")

    assert card.stable_id("acne_kb_v1") == card.stable_id("acne_kb_v1")
    assert card.stable_id("acne_kb_v1") != card.stable_id("acne_kb_v2")


def test_entity_point_identity_matches_frozen_contract() -> None:
    cards = build_entity_cards_from_taxonomy()

    assert len(cards) == 32
    assert len({entity_identity_key(card) for card in cards}) == 32
    assert len({entity_point_id(card) for card in cards}) == 32
    assert entity_point_id(_find_card("drug_product", "Epiduo")) == "c7fa15d3-0a5a-5c77-9176-ffe251114253"
    assert entity_point_id(_find_card("active_ingredient", "benzoyl_peroxide")) == "b7747ed6-eaf9-58d2-a50c-89fa46fcb5ad"
