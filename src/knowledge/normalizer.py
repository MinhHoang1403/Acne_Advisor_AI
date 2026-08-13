"""Rule-based drug and acne entity normalizer."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml

from src.knowledge.schemas import EntityCard, EntityType, canonical_text_key
from src.knowledge.taxonomy_models import (
    TaxonomyCatalog,
    load_taxonomy_catalog,
    normalize_taxonomy_alias,
)


# This YAML is the only active source-backed taxonomy for runtime normalization,
# EntityCards, and the deterministic Phase 1 graph.
DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "data" / "taxonomy" / "drug_aliases.yaml"
ACTIVE_TAXONOMY_CONTRACT = "data/taxonomy/drug_aliases.yaml"

CANONICAL_CORPUS_SOURCE_IDS = frozenset(
    {
        "nice_ng198_2026_08",
        "aad_acne_guideline_2024",
        "vn_moh_dermatology_4416_2023_acne",
        "aad_public_acne_2026_07",
    }
)
CURATED_TAXONOMY_SOURCE_IDS = frozenset({"pfizer_dalacin_t_label_current"})

SECTION_TO_ENTITY_TYPE: dict[str, EntityType] = {
    "drug_products": "drug_product",
    "active_ingredients": "active_ingredient",
    "drug_classes": "drug_class",
    "conditions": "condition",
    "safety_contexts": "safety_context",
}


def normalize_text_key(text: str) -> str:
    """Normalize free text while preserving Vietnamese characters."""

    return normalize_taxonomy_alias(text)


def _ascii_text_key(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    without_marks = "".join(
        char for char in normalized
        if unicodedata.category(char) != "Mn"
    )
    without_marks = without_marks.replace("đ", "d").replace("Đ", "D")
    return normalize_text_key(without_marks)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not value:
            continue
        key = normalize_text_key(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


class DrugEntityNormalizer:
    """Load drug taxonomy and map mentions/queries to canonical entity cards."""

    def __init__(self, taxonomy_path: str | Path | None = None) -> None:
        self.taxonomy_path = Path(taxonomy_path) if taxonomy_path else DEFAULT_TAXONOMY_PATH
        self.taxonomy = self._load_taxonomy(self.taxonomy_path)
        self.taxonomy_version = str(self.taxonomy.get("version") or "drug_taxonomy_v1")
        self.cards_by_type: dict[EntityType, dict[str, EntityCard]] = {
            "drug_product": {},
            "active_ingredient": {},
            "drug_class": {},
            "condition": {},
            "safety_context": {},
        }
        self.alias_index: dict[str, list[EntityCard]] = {}
        self._build_indexes()

    @staticmethod
    def _load_taxonomy(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Drug taxonomy file not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Drug taxonomy must be a mapping: {path}")
        return data

    def _build_indexes(self) -> None:
        if "entities" in self.taxonomy:
            self._build_indexes_from_v2_catalog(load_taxonomy_catalog(self.taxonomy_path))
            return

        for section, entity_type in SECTION_TO_ENTITY_TYPE.items():
            entries = self.taxonomy.get(section, {}) or {}
            if not isinstance(entries, dict):
                raise ValueError(f"Taxonomy section must be a mapping: {section}")
            for canonical_key, raw_entry in entries.items():
                if not isinstance(raw_entry, dict):
                    raise ValueError(f"Taxonomy entry must be a mapping: {section}.{canonical_key}")
                card = self._entry_to_card(entity_type, str(canonical_key), raw_entry)
                lookup_keys = {
                    normalize_text_key(str(canonical_key)),
                    _ascii_text_key(str(canonical_key)),
                    normalize_text_key(card.canonical_name),
                    _ascii_text_key(card.canonical_name),
                }
                lookup_keys.update(normalize_text_key(alias) for alias in card.aliases)
                lookup_keys.update(_ascii_text_key(alias) for alias in card.aliases)
                for key in lookup_keys:
                    self.alias_index.setdefault(key, [])
                    if card not in self.alias_index[key]:
                        self.alias_index[key].append(card)
                self.cards_by_type[entity_type][normalize_text_key(str(canonical_key))] = card
                self.cards_by_type[entity_type][normalize_text_key(card.canonical_name)] = card

    def _build_indexes_from_v2_catalog(self, catalog: TaxonomyCatalog) -> None:
        self.taxonomy_version = catalog.taxonomy_version
        for card in catalog.to_entity_cards(verified_only=True):
            lookup_keys = {
                normalize_text_key(card.canonical_name),
                _ascii_text_key(card.canonical_name),
                *[normalize_text_key(alias) for alias in card.aliases],
                *[_ascii_text_key(alias) for alias in card.aliases],
            }
            taxonomy_key = card.metadata.get("taxonomy_key")
            if isinstance(taxonomy_key, str):
                lookup_keys.add(normalize_text_key(taxonomy_key))
                lookup_keys.add(_ascii_text_key(taxonomy_key))
            for key in lookup_keys:
                self.alias_index.setdefault(key, [])
                if card not in self.alias_index[key]:
                    self.alias_index[key].append(card)
            self.cards_by_type[card.entity_type][normalize_text_key(card.canonical_name)] = card
            if isinstance(taxonomy_key, str):
                self.cards_by_type[card.entity_type][normalize_text_key(taxonomy_key)] = card

    def _entry_to_card(
        self,
        entity_type: EntityType,
        canonical_key: str,
        entry: dict[str, Any],
    ) -> EntityCard:
        payload: dict[str, Any] = {
            "entity_type": entity_type,
            "canonical_name": entry.get("canonical_name") or canonical_key,
            "aliases": entry.get("aliases") or [],
            "taxonomy_version": self.taxonomy_version,
            "metadata": {"taxonomy_key": canonical_key},
        }
        for field in [
            "active_ingredients",
            "drug_class",
            "used_for",
            "side_effects",
            "contraindications",
            "safety_contexts",
            "source_ids",
        ]:
            if field in entry:
                payload[field] = entry.get(field) or []
        return EntityCard(**payload)

    def match_alias(self, text: str) -> list[EntityCard]:
        """Find entity aliases inside text using normalized token boundaries."""

        normalized_text = normalize_text_key(text)
        ascii_text = _ascii_text_key(text)
        matches: list[EntityCard] = []
        for alias_key in sorted(self.alias_index, key=len, reverse=True):
            if not alias_key:
                continue
            pattern = rf"(?<!\w){re.escape(alias_key)}(?!\w)"
            if re.search(pattern, normalized_text, flags=re.UNICODE) or re.search(pattern, ascii_text, flags=re.UNICODE):
                matches.extend(self.alias_index[alias_key])
        return self._dedupe_cards(matches)

    def normalize_mention(self, mention: str) -> list[EntityCard]:
        """Normalize a short mention such as 'Dalacin T' or 'Epiduo'."""

        key = normalize_text_key(mention)
        return self._dedupe_cards(self.alias_index.get(key, []))

    def incompatible_alias_collisions(self) -> dict[str, list[str]]:
        """Return aliases resolving to more than one runtime entity identity."""

        collisions: dict[str, list[str]] = {}
        for alias, cards in sorted(self.alias_index.items()):
            identities = sorted(
                {
                    f"{card.entity_type}:{self._canonical_key_for_card(card)}"
                    for card in cards
                }
            )
            if len(identities) > 1:
                collisions[alias] = identities
        return collisions

    def expand_query(self, query: str) -> dict[str, Any]:
        direct_cards = self.match_alias(query)
        direct_cards = self._filter_contrast_class_mentions(query, direct_cards)
        expanded_cards = self._expand_related_cards(direct_cards)
        all_cards = self._dedupe_cards([*direct_cards, *expanded_cards])

        active_ingredients: list[str] = []
        drug_class: list[str] = []
        conditions: list[str] = []
        safety_contexts: list[str] = []
        expanded_terms: list[str] = []

        for card in all_cards:
            expanded_terms.append(card.canonical_name)
            expanded_terms.extend(card.aliases)
            active_ingredients.extend(card.active_ingredients)
            drug_class.extend(card.drug_class)
            if card.entity_type == "active_ingredient":
                active_ingredients.append(self._canonical_key_for_card(card))
            elif card.entity_type == "drug_class":
                drug_class.append(self._canonical_key_for_card(card))
            elif card.entity_type == "condition":
                conditions.append(self._canonical_key_for_card(card))
            elif card.entity_type == "safety_context":
                safety_contexts.append(self._canonical_key_for_card(card))

        return {
            "original_query": query,
            "normalized_entities": [card.to_payload() for card in all_cards],
            "expanded_terms": _dedupe(expanded_terms),
            "active_ingredients": _dedupe(active_ingredients),
            "drug_class": _dedupe(drug_class),
            "condition": _dedupe(conditions),
            "safety_context": _dedupe(safety_contexts),
        }

    def get_entity_card(self, entity_type: str, canonical_key_or_name: str) -> EntityCard | None:
        if entity_type not in self.cards_by_type:
            return None
        return self.cards_by_type[entity_type].get(normalize_text_key(canonical_key_or_name))  # type: ignore[index]

    def _expand_related_cards(self, cards: list[EntityCard]) -> list[EntityCard]:
        related: list[EntityCard] = []
        for card in cards:
            for ingredient in card.active_ingredients:
                ingredient_card = self.get_entity_card("active_ingredient", ingredient)
                if ingredient_card:
                    related.append(ingredient_card)
            for class_name in card.drug_class:
                class_card = self.get_entity_card("drug_class", class_name)
                if class_card:
                    related.append(class_card)
            for safety_context in card.safety_contexts:
                safety_card = self.get_entity_card("safety_context", safety_context)
                if safety_card:
                    related.append(safety_card)
        return related

    def _filter_contrast_class_mentions(
        self,
        query: str,
        cards: list[EntityCard],
    ) -> list[EntityCard]:
        """Avoid assigning the compared class as truth in yes/no contrast questions."""

        if not self._is_yes_no_class_membership_question(query):
            return cards

        subject_cards = [card for card in cards if card.entity_type != "drug_class"]
        if not subject_cards:
            return cards

        allowed_classes = self._classes_for_subject_cards(subject_cards)
        filtered: list[EntityCard] = []
        for card in cards:
            if card.entity_type != "drug_class":
                filtered.append(card)
                continue
            class_key = self._canonical_key_for_card(card)
            if class_key in allowed_classes:
                filtered.append(card)
        return filtered

    @staticmethod
    def _is_yes_no_class_membership_question(query: str) -> bool:
        ascii_query = _ascii_text_key(query)
        return (
            "co phai" in ascii_query
            and "khong" in ascii_query
        ) or (
            " is " in f" {ascii_query} "
            and (" antibiotic" in ascii_query or " retinoid" in ascii_query)
        )

    def _classes_for_subject_cards(self, cards: list[EntityCard]) -> set[str]:
        classes: set[str] = set()
        for card in cards:
            classes.update(card.drug_class)
            for ingredient in card.active_ingredients:
                ingredient_card = self.get_entity_card("active_ingredient", ingredient)
                if ingredient_card:
                    classes.update(ingredient_card.drug_class)
        return {normalize_text_key(class_name).replace(" ", "_") for class_name in classes}

    @staticmethod
    def _dedupe_cards(cards: list[EntityCard]) -> list[EntityCard]:
        seen: set[str] = set()
        output: list[EntityCard] = []
        for card in cards:
            key = card.stable_id()
            if key in seen:
                continue
            seen.add(key)
            output.append(card)
        return output

    @staticmethod
    def _canonical_key_for_card(card: EntityCard) -> str:
        taxonomy_key = card.metadata.get("taxonomy_key")
        if isinstance(taxonomy_key, str) and taxonomy_key:
            return taxonomy_key
        return canonical_text_key(card.canonical_name).replace(" ", "_")


__all__ = [
    "ACTIVE_TAXONOMY_CONTRACT",
    "CANONICAL_CORPUS_SOURCE_IDS",
    "CURATED_TAXONOMY_SOURCE_IDS",
    "DEFAULT_TAXONOMY_PATH",
    "DrugEntityNormalizer",
    "normalize_text_key",
]
