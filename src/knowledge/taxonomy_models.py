"""Versioned taxonomy contracts and deterministic validation helpers."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from src.knowledge.schemas import EntityCard, EntityType, canonical_text_key


TAXONOMY_SCHEMA_VERSION = "taxonomy_schema_v2"
DEFAULT_TAXONOMY_V2_PATH = Path(__file__).resolve().parents[2] / "data" / "taxonomy" / "drug_taxonomy_v2.yaml"

ReviewStatus = Literal["draft", "verified", "rejected"]
SourceType = Literal["pdf", "json", "existing_taxonomy", "manual_review"]

# The active taxonomy deliberately keeps these as ingredients without inventing
# a self-named DrugClass just to satisfy a graph completeness heuristic.
ALLOWED_CLASSLESS_ACTIVE_INGREDIENTS = frozenset({"azelaic_acid", "benzoyl_peroxide"})


def normalize_taxonomy_name(text: str) -> str:
    """Normalize canonical/display names for deterministic taxonomy IDs."""

    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    normalized = re.sub(r"[()/]+", " ", normalized)
    normalized = normalized.replace("-", " ").replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized.strip().lower())
    return normalized


def normalize_taxonomy_alias(text: str) -> str:
    """Normalize aliases in the same view used by query/entity matching."""

    return normalize_taxonomy_name(text)


def accentless_taxonomy_alias(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    without_marks = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    without_marks = without_marks.replace("đ", "d").replace("Đ", "D")
    return normalize_taxonomy_alias(without_marks)


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_type: SourceType
    locator: str | None = None
    note: str | None = None

    @field_validator("source_id")
    @classmethod
    def _source_id_not_empty(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("source_id cannot be empty")
        return cleaned


class TaxonomyEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_name: str
    entity_type: EntityType
    display_name: str
    aliases: list[str] = Field(default_factory=list)
    normalized_aliases: list[str] = Field(default_factory=list)
    source_references: list[SourceReference]
    review_status: ReviewStatus
    reviewed_at: str | None = None
    taxonomy_version: str
    entity_schema_version: str

    @field_validator("canonical_name", "display_name")
    @classmethod
    def _clean_required_string(cls, value: str) -> str:
        cleaned = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or "")).strip())
        if not cleaned:
            raise ValueError("taxonomy string cannot be empty")
        return cleaned

    @field_validator("aliases", "normalized_aliases", mode="before")
    @classmethod
    def _clean_aliases(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("aliases must be a list")
        output: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise TypeError("aliases must be strings")
            cleaned = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", item).strip())
            if not cleaned:
                raise ValueError("alias cannot be empty")
            key = normalize_taxonomy_alias(cleaned)
            if key in seen:
                continue
            seen.add(key)
            output.append(cleaned)
        return output

    @model_validator(mode="after")
    def _populate_normalized_aliases(self) -> "TaxonomyEntity":
        normalized = [normalize_taxonomy_alias(self.canonical_name), *[normalize_taxonomy_alias(a) for a in self.aliases]]
        deduped = _dedupe(normalized)
        object.__setattr__(self, "normalized_aliases", deduped)
        return self

    def entity_key(self) -> str:
        return f"{self.entity_type}:{normalize_taxonomy_name(self.canonical_name).replace(' ', '_')}"

    def to_entity_card(self) -> EntityCard:
        metadata = {
            "taxonomy_key": self.canonical_name,
            "review_status": self.review_status,
            "source_references": [ref.model_dump(mode="json") for ref in self.source_references],
        }
        payload = {
            "entity_type": self.entity_type,
            "canonical_name": self.canonical_name,
            "aliases": self.aliases,
            "source_ids": [ref.source_id for ref in self.source_references],
            "taxonomy_version": self.taxonomy_version,
            "entity_schema_version": self.entity_schema_version,
            "metadata": metadata,
        }
        return EntityCard(**payload)


class DrugProductEntity(TaxonomyEntity):
    entity_type: Literal["drug_product"]
    active_ingredients: list[str]
    drug_classes: list[str] = Field(default_factory=list)
    dosage_forms: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    route: str | None = None
    markets: list[str] = Field(default_factory=list)
    formulation_notes: str | None = None

    def to_entity_card(self) -> EntityCard:
        card = super().to_entity_card()
        return card.model_copy(update={"active_ingredients": self.active_ingredients, "drug_class": self.drug_classes})


class ActiveIngredientEntity(TaxonomyEntity):
    entity_type: Literal["active_ingredient"]
    drug_classes: list[str]
    indications: list[str] = Field(default_factory=list)
    safety_contexts: list[str] = Field(default_factory=list)

    def to_entity_card(self) -> EntityCard:
        card = super().to_entity_card()
        return card.model_copy(update={"drug_class": self.drug_classes, "used_for": self.indications, "safety_contexts": self.safety_contexts})


class DrugClassEntity(TaxonomyEntity):
    entity_type: Literal["drug_class"]
    description: str | None = None


class ConditionEntity(TaxonomyEntity):
    entity_type: Literal["condition"]


class SafetyContextEntity(TaxonomyEntity):
    entity_type: Literal["safety_context"]
    context_type: str | None = None


TaxonomyEntityUnion = (
    DrugProductEntity
    | ActiveIngredientEntity
    | DrugClassEntity
    | ConditionEntity
    | SafetyContextEntity
)
_ENTITY_ADAPTER = TypeAdapter(TaxonomyEntityUnion)


class AllowedAliasCollision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_alias: str
    entity_keys: list[str]
    reason: str

    @field_validator("normalized_alias")
    @classmethod
    def _normalize_alias(cls, value: str) -> str:
        return normalize_taxonomy_alias(value)


class TaxonomyCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    taxonomy_version: str
    entity_schema_version: str = "entity_schema_v2"
    taxonomy_schema_version: str = TAXONOMY_SCHEMA_VERSION
    entities: list[TaxonomyEntityUnion]
    allowed_alias_collisions: list[AllowedAliasCollision] = Field(default_factory=list)

    def verified_entities(self) -> list[TaxonomyEntityUnion]:
        return [entity for entity in self.entities if entity.review_status == "verified"]

    def to_entity_cards(self, *, verified_only: bool = True) -> list[EntityCard]:
        entities = self.verified_entities() if verified_only else list(self.entities)
        return [entity.to_entity_card() for entity in entities]

    def entity_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(entity.entity_type for entity in self.entities).items()))


class TaxonomyCheck(BaseModel):
    name: str
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class TaxonomyValidationReport(BaseModel):
    passed: bool
    taxonomy_version: str
    entity_counts: dict[str, int]
    verified_count: int
    draft_count: int
    rejected_count: int
    alias_count: int
    checks: list[TaxonomyCheck]
    warnings: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)


def load_taxonomy_catalog(path: str | Path = DEFAULT_TAXONOMY_V2_PATH) -> TaxonomyCatalog:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "entities" not in data:
        data = migrate_v1_taxonomy(data, source_path=path).model_dump(mode="json")
    entities = [_ENTITY_ADAPTER.validate_python(raw) for raw in data.get("entities", [])]
    return TaxonomyCatalog(
        taxonomy_version=data["taxonomy_version"],
        entity_schema_version=data.get("entity_schema_version", "entity_schema_v2"),
        taxonomy_schema_version=data.get("taxonomy_schema_version", TAXONOMY_SCHEMA_VERSION),
        entities=entities,
        allowed_alias_collisions=data.get("allowed_alias_collisions", []) or [],
    )


def migrate_v1_taxonomy(data: dict[str, Any], *, source_path: str | Path) -> TaxonomyCatalog:
    version = "drug_taxonomy_v2"
    schema_version = "entity_schema_v2"
    entities: list[TaxonomyEntityUnion] = []
    source_path = Path(source_path)

    def source(section: str, key: str, note: str | None = None) -> list[SourceReference]:
        return [
            SourceReference(
                source_id=f"drug_taxonomy_v1:{section}.{key}",
                source_type="existing_taxonomy",
                locator=f"{source_path.as_posix()}#{section}.{key}",
                note=note,
            )
        ]

    for key, entry in (data.get("drug_products") or {}).items():
        entities.append(
            DrugProductEntity(
                canonical_name=entry.get("canonical_name") or key,
                entity_type="drug_product",
                display_name=entry.get("canonical_name") or key,
                aliases=entry.get("aliases") or [],
                active_ingredients=entry.get("active_ingredients") or [],
                drug_classes=entry.get("drug_class") or [],
                source_references=source("drug_products", key),
                review_status="verified",
                taxonomy_version=version,
                entity_schema_version=schema_version,
            )
        )

    for key, entry in (data.get("active_ingredients") or {}).items():
        entities.append(
            ActiveIngredientEntity(
                canonical_name=entry.get("canonical_name") or key,
                entity_type="active_ingredient",
                display_name=entry.get("canonical_name") or key,
                aliases=entry.get("aliases") or [],
                drug_classes=entry.get("drug_class") or [],
                source_references=source("active_ingredients", key),
                review_status="verified",
                taxonomy_version=version,
                entity_schema_version=schema_version,
            )
        )

    drug_classes = dict(data.get("drug_classes") or {})
    referenced_classes = {
        class_name
        for section in ("drug_products", "active_ingredients")
        for entry in (data.get(section) or {}).values()
        for class_name in (entry.get("drug_class") or [])
    }
    for class_name in sorted(referenced_classes):
        drug_classes.setdefault(
            class_name,
            {
                "canonical_name": class_name,
                "aliases": [class_name.replace("_", " ")],
                "_note": "Added by v1 migration because an existing entity references this class.",
            },
        )

    for key, entry in drug_classes.items():
        entities.append(
            DrugClassEntity(
                canonical_name=entry.get("canonical_name") or key,
                entity_type="drug_class",
                display_name=entry.get("canonical_name") or key,
                aliases=entry.get("aliases") or [],
                source_references=source("drug_classes", key, entry.get("_note")),
                review_status="verified",
                taxonomy_version=version,
                entity_schema_version=schema_version,
            )
        )

    for key, entry in (data.get("conditions") or {}).items():
        entities.append(
            ConditionEntity(
                canonical_name=entry.get("canonical_name") or key,
                entity_type="condition",
                display_name=entry.get("canonical_name") or key,
                aliases=entry.get("aliases") or [],
                source_references=source("conditions", key),
                review_status="verified",
                taxonomy_version=version,
                entity_schema_version=schema_version,
            )
        )

    for key, entry in (data.get("safety_contexts") or {}).items():
        entities.append(
            SafetyContextEntity(
                canonical_name=entry.get("canonical_name") or key,
                entity_type="safety_context",
                display_name=entry.get("canonical_name") or key,
                aliases=entry.get("aliases") or [],
                context_type=entry.get("context_type"),
                source_references=source("safety_contexts", key),
                review_status="verified",
                taxonomy_version=version,
                entity_schema_version=schema_version,
            )
        )

    return TaxonomyCatalog(
        taxonomy_version=version,
        entity_schema_version=schema_version,
        entities=entities,
        allowed_alias_collisions=_default_allowed_alias_collisions(),
    )


def validate_taxonomy_catalog(catalog: TaxonomyCatalog, *, production_verified_only: bool = True) -> TaxonomyValidationReport:
    checks: list[TaxonomyCheck] = []
    warnings: list[str] = []
    failures: list[str] = []

    def add(name: str, passed: bool, details: dict[str, Any] | None = None, *, warning: str | None = None) -> None:
        checks.append(TaxonomyCheck(name=name, passed=passed, details=details or {}))
        if not passed:
            failures.append(name)
        if warning:
            warnings.append(warning)

    entities = catalog.entities
    entity_keys = [entity.entity_key() for entity in entities]
    key_counts = Counter(entity_keys)
    duplicate_keys = sorted(key for key, count in key_counts.items() if count > 1)
    add("duplicate_canonical_id", not duplicate_keys, {"duplicates": duplicate_keys})

    per_type_names = Counter((entity.entity_type, normalize_taxonomy_name(entity.canonical_name)) for entity in entities)
    duplicate_names = sorted(f"{etype}:{name}" for (etype, name), count in per_type_names.items() if count > 1)
    add("duplicate_canonical_name", not duplicate_names, {"duplicates": duplicate_names})

    missing_provenance = [entity.entity_key() for entity in entities if not entity.source_references]
    add("missing_provenance", not missing_provenance, {"entities": missing_provenance})

    missing_review = [entity.entity_key() for entity in entities if entity.review_status not in {"draft", "verified", "rejected"}]
    add("missing_review_status", not missing_review, {"entities": missing_review})

    class_names = {entity.canonical_name for entity in entities if entity.entity_type == "drug_class"}
    ingredient_names = {entity.canonical_name for entity in entities if entity.entity_type == "active_ingredient"}
    safety_names = {entity.canonical_name for entity in entities if entity.entity_type == "safety_context"}

    product_issues: list[str] = []
    ingredient_issues: list[str] = []
    safety_issues: list[str] = []
    for entity in entities:
        if isinstance(entity, DrugProductEntity):
            if not entity.active_ingredients:
                product_issues.append(f"{entity.entity_key()}:missing_ingredient")
            for ingredient in entity.active_ingredients:
                if ingredient not in ingredient_names:
                    product_issues.append(f"{entity.entity_key()}:unknown_ingredient:{ingredient}")
            for class_name in entity.drug_classes:
                if class_name not in class_names:
                    product_issues.append(f"{entity.entity_key()}:unknown_class:{class_name}")
        if isinstance(entity, ActiveIngredientEntity):
            canonical_key = normalize_taxonomy_name(entity.canonical_name).replace(" ", "_")
            if not entity.drug_classes and canonical_key not in ALLOWED_CLASSLESS_ACTIVE_INGREDIENTS:
                ingredient_issues.append(f"{entity.entity_key()}:missing_class")
            for class_name in entity.drug_classes:
                if class_name not in class_names:
                    ingredient_issues.append(f"{entity.entity_key()}:unknown_class:{class_name}")
            for safety_context in entity.safety_contexts:
                if safety_context not in safety_names:
                    safety_issues.append(f"{entity.entity_key()}:unknown_safety_context:{safety_context}")
    add("drug_product_relationship_integrity", not product_issues, {"issues": product_issues})
    add("active_ingredient_relationship_integrity", not ingredient_issues, {"issues": ingredient_issues})
    add("safety_context_integrity", not safety_issues, {"issues": safety_issues})

    collisions = _alias_collisions(catalog)
    unresolved = [collision for collision in collisions if not _collision_allowed(catalog, collision)]
    add("alias_collision", not unresolved, {"unresolved": unresolved})
    if collisions and not unresolved:
        warnings.append(f"Allowed alias collisions present: {len(collisions)}")

    if production_verified_only:
        draft_or_rejected = [entity.entity_key() for entity in entities if entity.review_status != "verified"]
        add("production_entities_verified", not draft_or_rejected, {"non_verified": draft_or_rejected})

    status_counts = Counter(entity.review_status for entity in entities)
    alias_count = sum(len(entity.aliases) for entity in entities)
    return TaxonomyValidationReport(
        passed=not failures,
        taxonomy_version=catalog.taxonomy_version,
        entity_counts=catalog.entity_counts(),
        verified_count=status_counts.get("verified", 0),
        draft_count=status_counts.get("draft", 0),
        rejected_count=status_counts.get("rejected", 0),
        alias_count=alias_count,
        checks=checks,
        warnings=warnings,
        failures=failures,
    )


def catalog_payload_hash(card: EntityCard) -> str:
    payload = card.to_payload()
    payload.pop("metadata", None)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _alias_collisions(catalog: TaxonomyCatalog) -> list[dict[str, Any]]:
    alias_to_entities: dict[str, set[str]] = defaultdict(set)
    for entity in catalog.entities:
        entity_key = entity.entity_key()
        values = [entity.canonical_name, *entity.aliases]
        for value in values:
            alias_to_entities[normalize_taxonomy_alias(value)].add(entity_key)
            alias_to_entities[accentless_taxonomy_alias(value)].add(entity_key)
    return [
        {"normalized_alias": alias, "entity_keys": sorted(entity_keys)}
        for alias, entity_keys in sorted(alias_to_entities.items())
        if len(entity_keys) > 1
    ]


def _collision_allowed(catalog: TaxonomyCatalog, collision: dict[str, Any]) -> bool:
    alias = collision["normalized_alias"]
    keys = set(collision["entity_keys"])
    for allowed in catalog.allowed_alias_collisions:
        if allowed.normalized_alias == alias and keys.issubset(set(allowed.entity_keys)):
            return True
    return False


def _default_allowed_alias_collisions() -> list[AllowedAliasCollision]:
    bp_keys = ["active_ingredient:benzoyl_peroxide", "drug_class:benzoyl_peroxide"]
    azelaic_keys = ["active_ingredient:azelaic_acid", "drug_class:azelaic_acid"]
    collisions = [
        AllowedAliasCollision(
            normalized_alias=alias,
            entity_keys=bp_keys,
            reason="Benzoyl peroxide is modeled as both active ingredient and mechanism-specific class in taxonomy v1.",
        )
        for alias in ("benzoyl peroxide", "benzoyl_peroxide", "bpo", "bp")
    ]
    collisions.extend(
        AllowedAliasCollision(
            normalized_alias=alias,
            entity_keys=azelaic_keys,
            reason="Azelaic acid class is added by migration to close an existing v1 relationship.",
        )
        for alias in ("azelaic acid", "azelaic_acid")
    )
    return collisions


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


__all__ = [
    "DEFAULT_TAXONOMY_V2_PATH",
    "TAXONOMY_SCHEMA_VERSION",
    "ActiveIngredientEntity",
    "ALLOWED_CLASSLESS_ACTIVE_INGREDIENTS",
    "AllowedAliasCollision",
    "ConditionEntity",
    "DrugClassEntity",
    "DrugProductEntity",
    "SafetyContextEntity",
    "SourceReference",
    "TaxonomyCatalog",
    "TaxonomyEntity",
    "TaxonomyValidationReport",
    "accentless_taxonomy_alias",
    "catalog_payload_hash",
    "load_taxonomy_catalog",
    "migrate_v1_taxonomy",
    "normalize_taxonomy_alias",
    "normalize_taxonomy_name",
    "validate_taxonomy_catalog",
]
