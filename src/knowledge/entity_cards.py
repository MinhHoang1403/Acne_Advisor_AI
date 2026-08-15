"""Tạo knowledge card tập trung theo entity từ acne drug taxonomy."""

from __future__ import annotations

from collections.abc import Iterable

from src.knowledge.normalizer import DrugEntityNormalizer
from src.knowledge.schemas import EntityCard


BP_NOT_ANTIBIOTIC_NOTE = "benzoyl peroxide is not an antibiotic"


def _dedupe_cards(cards: Iterable[EntityCard]) -> list[EntityCard]:
    seen: set[str] = set()
    output: list[EntityCard] = []
    for card in cards:
        key = card.stable_id()
        if key in seen:
            continue
        seen.add(key)
        output.append(card)
    return output


def build_entity_cards_from_taxonomy(
    normalizer: DrugEntityNormalizer | None = None,
) -> list[EntityCard]:
    """Tạo một ``EntityCard`` ổn định cho mỗi taxonomy entry."""

    normalizer = normalizer or DrugEntityNormalizer()
    cards: list[EntityCard] = []

    for entity_type in (
        "drug_product",
        "active_ingredient",
        "drug_class",
        "condition",
        "safety_context",
    ):
        cards.extend(normalizer.cards_by_type[entity_type].values())  # type: ignore[index]

    enriched_cards: list[EntityCard] = []
    for card in _dedupe_cards(cards):
        metadata = dict(card.metadata)
        if _is_benzoyl_peroxide_card(card):
            metadata["not_antibiotic"] = True
            metadata["clinical_note"] = BP_NOT_ANTIBIOTIC_NOTE
            card = card.model_copy(update={"metadata": metadata})
        enriched_cards.append(card)

    return enriched_cards


def entity_card_to_text(card: EntityCard) -> str:
    """Tạo biểu diễn text ngắn gọn để embedding/search một entity card."""

    lines = [
        f"Entity type: {card.entity_type}",
        f"Name: {card.canonical_name}",
    ]

    optional_lines = [
        ("Aliases", card.aliases),
        ("Active ingredients", card.active_ingredients),
        ("Drug class", card.drug_class),
        ("Used for", card.used_for),
        ("Side effects", card.side_effects),
        ("Contraindications", card.contraindications),
        ("Safety contexts", card.safety_contexts),
        ("Source IDs", card.source_ids),
    ]
    for label, values in optional_lines:
        if values:
            lines.append(f"{label}: {', '.join(values)}")

    if _is_benzoyl_peroxide_card(card):
        lines.append("Clinical note: benzoyl peroxide is not an antibiotic.")

    return "\n".join(lines)


def _is_benzoyl_peroxide_card(card: EntityCard) -> bool:
    taxonomy_key = str(card.metadata.get("taxonomy_key") or "").lower()
    canonical = card.canonical_name.lower()
    return taxonomy_key == "benzoyl_peroxide" or canonical == "benzoyl_peroxide"


__all__ = [
    "BP_NOT_ANTIBIOTIC_NOTE",
    "build_entity_cards_from_taxonomy",
    "entity_card_to_text",
]
