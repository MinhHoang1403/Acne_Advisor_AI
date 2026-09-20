"""Enrich knowledge chunks with deterministic taxonomy metadata.

The active implementation maps aliases through ``DrugEntityNormalizer``. It
does not call an LLM or participate in runtime retrieval and reranking.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from src.knowledge import DrugEntityNormalizer


logger = logging.getLogger(__name__)

NEW_DOMAIN_METADATA_LIST_FIELDS = (
    "drug_product",
    "active_ingredient",
    "drug_class",
    "condition",
    "safety_context",
    "query_intent_hint",
)


def _dedupe(values: list[Any]) -> list[str]:
    """Loại item rỗng/trùng và giữ thứ tự xuất hiện của string values."""
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value is None:
            continue
        item = str(value).strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _empty_taxonomy_metadata() -> dict[str, Any]:
    metadata = {field_name: [] for field_name in NEW_DOMAIN_METADATA_LIST_FIELDS}
    metadata["taxonomy_version"] = _get_drug_entity_normalizer().taxonomy_version
    metadata["entity_schema_version"] = "source_backed_entity_card"
    return metadata


@lru_cache(maxsize=1)
def _get_drug_entity_normalizer() -> DrugEntityNormalizer:
    return DrugEntityNormalizer()


def _safe_expand_query(text: str) -> dict[str, Any]:
    try:
        return _get_drug_entity_normalizer().expand_query(text)
    except Exception as exc:  # pragma: no cover - exercised via monkeypatch tests if needed
        logger.warning(
            "Drug taxonomy normalizer failed; returning empty taxonomy metadata: %s",
            exc,
        )
        return {
            "normalized_entities": [],
            "active_ingredients": [],
            "drug_class": [],
            "condition": [],
            "safety_context": [],
        }


def _infer_query_intent_hint(
    text: str,
    active_ingredient: list[str],
    condition: list[str],
) -> list[str]:
    text_lower = text.lower()
    hints: list[str] = []

    def add_if(intent: str, keywords: list[str]) -> None:
        if any(keyword in text_lower for keyword in keywords):
            hints.append(intent)

    add_if(
        "side_effect",
        [
            "side effect",
            "side effects",
            "adverse",
            "tác dụng phụ",
            "kích ứng",
            "khô",
            "đỏ",
            "redness",
            "dryness",
            "peeling",
            "bong tróc",
            "irritation",
        ],
    )
    add_if(
        "contraindication",
        [
            "contraindicated",
            "not for use",
            "avoid",
            "chống chỉ định",
            "không dùng",
            "không sử dụng",
            "không nên dùng",
        ],
    )
    add_if(
        "pregnancy_safety",
        [
            "pregnancy",
            "pregnant",
            "thai kỳ",
            "mang thai",
            "breastfeeding",
            "cho con bú",
        ],
    )
    add_if(
        "referral",
        [
            "refer",
            "referral",
            "dermatologist",
            "bác sĩ da liễu",
            "chuyển tuyến",
            "khám",
        ],
    )
    add_if(
        "skincare",
        [
            "cleanser",
            "syndet",
            "moisturiser",
            "moisturizer",
            "sunscreen",
            "make-up",
            "makeup",
            "rửa mặt",
            "dưỡng ẩm",
            "chống nắng",
            "trang điểm",
        ],
    )
    add_if(
        "dosage_request",
        [
            "dose",
            "dosage",
            "mg/kg",
            "liều",
            "uống bao nhiêu",
        ],
    )
    add_if(
        "comparison",
        [
            "compare",
            "versus",
            " vs ",
            "khác nhau",
            "so sánh",
        ],
    )

    if active_ingredient:
        hints.append("ingredient_info")
    if condition or any(keyword in text_lower for keyword in ["acne", "mụn", "trứng cá"]):
        hints.append("condition_advice")

    return _dedupe(hints)


def enrich_domain_metadata(
    text: str,
    existing_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gộp chunk metadata hiện có với entity metadata dựa trên taxonomy.

    Hàm chủ ý dùng rule deterministic và giữ ingestion tiếp tục khi taxonomy
    loading lỗi; lỗi đó chỉ được ghi warning.
    """
    enriched = dict(existing_metadata or {})
    taxonomy_metadata = _empty_taxonomy_metadata()
    expanded = _safe_expand_query(text or "")
    entities = expanded.get("normalized_entities", [])

    drug_product: list[str] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        if entity.get("entity_type") == "drug_product":
            drug_product.append(str(entity.get("canonical_name") or ""))

    active_ingredient = list(expanded.get("active_ingredients") or [])
    drug_class = list(expanded.get("drug_class") or [])
    condition = list(expanded.get("condition") or [])
    safety_context = list(expanded.get("safety_context") or [])

    # Giữ ingredient metadata cũ nhưng chỉ map active ingredient đã biết.
    try:
        normalizer = _get_drug_entity_normalizer()
        for old_ingredient in enriched.get("ingredient", []) or []:
            card = normalizer.get_entity_card("active_ingredient", str(old_ingredient))
            if card:
                active_ingredient.append(str(card.metadata.get("taxonomy_key") or card.canonical_name))
    except Exception:
        pass

    # Extractor trước đã dùng safety_context cho irritation/dryness; giữ metadata
    # đó và bổ sung taxonomy context như pregnancy/breastfeeding.
    safety_context.extend(enriched.get("safety_context", []) or [])

    taxonomy_metadata["drug_product"] = _dedupe(drug_product)
    taxonomy_metadata["active_ingredient"] = _dedupe(active_ingredient)
    taxonomy_metadata["drug_class"] = _dedupe(drug_class)
    taxonomy_metadata["condition"] = _dedupe(condition)
    taxonomy_metadata["safety_context"] = _dedupe(safety_context)
    taxonomy_metadata["query_intent_hint"] = _infer_query_intent_hint(
        text or "",
        taxonomy_metadata["active_ingredient"],
        taxonomy_metadata["condition"],
    )

    enriched.update(taxonomy_metadata)
    return enriched
