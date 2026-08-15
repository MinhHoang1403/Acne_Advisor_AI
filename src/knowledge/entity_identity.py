"""Tạo canonical identity và deterministic Qdrant UUID cho EntityCards.

Identity dùng ``entity_type`` cùng taxonomy key/canonical name đã normalize.
Point ID là UUIDv5 trong URL namespace trên chuỗi ``acne_entity:<identity>``;
cùng input luôn cho cùng UUID. ``kb_version`` không tham gia để entity không đổi
point identity chỉ vì build version thay đổi.
"""

from __future__ import annotations

import uuid
from typing import Any

from src.knowledge.normalizer import normalize_text_key
from src.knowledge.schemas import EntityCard


def entity_identity_key(card_or_payload: EntityCard | dict[str, Any]) -> str:
    """Trả canonical identity không phụ thuộc knowledge build version."""

    if isinstance(card_or_payload, EntityCard):
        entity_type = card_or_payload.entity_type
        metadata = card_or_payload.metadata
        canonical_name = card_or_payload.canonical_name
    else:
        entity_type = str(card_or_payload.get("entity_type") or "").strip()
        metadata = card_or_payload.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        canonical_name = str(card_or_payload.get("canonical_name") or "").strip()

    taxonomy_key = str(metadata.get("taxonomy_key") or "").strip()
    identity_source = taxonomy_key or canonical_name
    normalized = normalize_text_key(identity_source).replace(" ", "_")
    if not entity_type or not normalized:
        raise ValueError("Entity identity requires entity_type and taxonomy_key/canonical_name.")
    return f"{entity_type}:{normalized}"


def entity_point_id(card: EntityCard, kb_version: str = "frozen_phase1_build") -> str:
    """Trả Qdrant UUID ổn định cho EntityCard identity.

    ``kb_version`` vẫn được nhận để giữ compatibility nhưng cố ý không tham gia
    canonical identity.
    """

    _ = kb_version
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"acne_entity:{entity_identity_key(card)}"))


__all__ = ["entity_identity_key", "entity_point_id"]
