"""Typed query, chunk-candidate, and context-packing contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class NormalizedQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_query: str
    normalized_text: str


class RetrievedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
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
    role: str
    text: str
    payload: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    fused_score: float | None = None
    rank: int | None = None
    matched_metadata: dict[str, Any] = Field(default_factory=dict)
    reason: str


class PackedContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_query: str
    items: list[ContextItem] = Field(default_factory=list)
    context_text: str
    chunk_items_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ContextItem",
    "NormalizedQuery",
    "PackedContext",
    "RetrievedCandidate",
]
