"""Pydantic schemas for structured acne knowledge entity cards."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


EntityType = Literal[
    "drug_product",
    "active_ingredient",
    "drug_class",
    "condition",
    "safety_context",
]


def canonical_text_key(text: str) -> str:
    """Trả key chữ thường deterministic cho schema ID và lookup."""

    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("-", " ").replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized.strip().lower())
    return normalized


def _clean_string(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).strip())


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        item = _clean_string(value)
        if not item:
            continue
        key = canonical_text_key(item)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned


class KnowledgeEntityBase(BaseModel):
    """Shared normalization behavior for knowledge entity schemas."""

    model_config = ConfigDict(extra="forbid")

    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)

    @field_validator("canonical_name", mode="before")
    @classmethod
    def _normalize_canonical_name(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError("canonical_name must be a string")
        cleaned = _clean_string(value)
        if not cleaned:
            raise ValueError("canonical_name cannot be empty")
        return cleaned

    @field_validator("*", mode="before")
    @classmethod
    def _normalize_string_lists(cls, value: Any) -> Any:
        if isinstance(value, list):
            return _dedupe_strings(value)
        return value


class DrugProduct(KnowledgeEntityBase):
    aliases: list[str] = Field(default_factory=list)
    active_ingredients: list[str] = Field(default_factory=list)
    drug_class: list[str] = Field(default_factory=list)
    product_type: str | None = None
    source_ids: list[str] = Field(default_factory=list)

    @field_validator("product_type", mode="before")
    @classmethod
    def _normalize_optional_string(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("product_type must be a string")
        cleaned = _clean_string(value)
        return cleaned or None


class ActiveIngredient(KnowledgeEntityBase):
    aliases: list[str] = Field(default_factory=list)
    drug_class: list[str] = Field(default_factory=list)
    used_for: list[str] = Field(default_factory=list)
    side_effects: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    safety_contexts: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class DrugClass(KnowledgeEntityBase):
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    examples: list[str] = Field(default_factory=list)
    used_for: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)

    @field_validator("description", mode="before")
    @classmethod
    def _normalize_optional_string(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("description must be a string")
        cleaned = _clean_string(value)
        return cleaned or None


class Condition(KnowledgeEntityBase):
    aliases: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    severity_markers: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class SafetyContext(KnowledgeEntityBase):
    aliases: list[str] = Field(default_factory=list)
    context_type: str | None = None
    cautions: list[str] = Field(default_factory=list)
    contraindicated_items: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)

    @field_validator("context_type", mode="before")
    @classmethod
    def _normalize_optional_string(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("context_type must be a string")
        cleaned = canonical_text_key(value).replace(" ", "_")
        return cleaned or None


class EntityCard(BaseModel):
    """Flattened entity card used as payload for retrieval and future KG enrichment."""

    model_config = ConfigDict(extra="forbid")

    entity_type: EntityType
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    active_ingredients: list[str] = Field(default_factory=list)
    drug_class: list[str] = Field(default_factory=list)
    used_for: list[str] = Field(default_factory=list)
    side_effects: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    safety_contexts: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    taxonomy_version: str = "acne_taxonomy_2026_08"
    entity_schema_version: str = "source_backed_entity_card"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("canonical_name", mode="before")
    @classmethod
    def _normalize_canonical_name(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError("canonical_name must be a string")
        cleaned = _clean_string(value)
        if not cleaned:
            raise ValueError("canonical_name cannot be empty")
        return cleaned

    @field_validator("*", mode="before")
    @classmethod
    def _normalize_lists(cls, value: Any) -> Any:
        if isinstance(value, list):
            return _dedupe_strings(value)
        return value

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def stable_id(self, kb_version: str = "frozen_phase1_build") -> str:
        # Stable ID hash gồm build/taxonomy/schema/type/canonical name rồi lấy 24
        # hex characters. Đây là deterministic identity, không phải security token.
        raw = "|".join(
            [
                kb_version,
                self.taxonomy_version,
                self.entity_schema_version,
                self.entity_type,
                canonical_text_key(self.canonical_name),
            ]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"{self.entity_type}:{digest[:24]}"


__all__ = [
    "ActiveIngredient",
    "Condition",
    "DrugClass",
    "DrugProduct",
    "EntityCard",
    "EntityType",
    "SafetyContext",
    "canonical_text_key",
]
