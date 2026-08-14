"""Pydantic contracts for Phase 2A entity-aware retrieval."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        item = value.strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


class NormalizedQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_query: str
    normalized_text: str
    intent: str
    drug_product: list[str] = Field(default_factory=list)
    active_ingredient: list[str] = Field(default_factory=list)
    drug_class: list[str] = Field(default_factory=list)
    condition: list[str] = Field(default_factory=list)
    safety_context: list[str] = Field(default_factory=list)
    query_intent_hint: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "drug_product",
        "active_ingredient",
        "drug_class",
        "condition",
        "safety_context",
        "query_intent_hint",
        "aliases",
        mode="before",
    )
    @classmethod
    def _normalize_lists(cls, value: Any) -> Any:
        if isinstance(value, list):
            return _dedupe_strings(value)
        return value


class RetrievedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    source: Literal["entity", "chunk"]
    collection: str
    text: str
    score: float | None = None
    fused_score: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    matched_metadata: dict[str, Any] = Field(default_factory=dict)
    rank: int | None = None
    debug: dict[str, Any] = Field(default_factory=dict)


class ContextItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    source: Literal["entity", "chunk"]
    role: str
    text: str
    payload: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    fused_score: float | None = None
    rank: int | None = None
    matched_metadata: dict[str, Any] = Field(default_factory=dict)
    reason: str


class ContextPackTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str
    selected_entity_ids: list[str] = Field(default_factory=list)
    selected_chunk_ids: list[str] = Field(default_factory=list)
    selection_reasons: list[str] = Field(default_factory=list)
    dropped_candidates: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PackedContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_query: str
    intent: str
    items: list[ContextItem] = Field(default_factory=list)
    context_text: str
    entity_items_count: int = 0
    chunk_items_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ContextItem",
    "ContextPackTrace",
    "NormalizedQuery",
    "PackedContext",
    "RetrievedCandidate",
]
