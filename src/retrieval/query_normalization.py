"""Taxonomy-backed query normalization used by retrieval and answer policy."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from src.ingestion.domain_metadata import enrich_domain_metadata
from src.knowledge import DrugEntityNormalizer
from src.knowledge.normalizer import normalize_text_key
from src.retrieval.contracts import NormalizedQuery


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


@lru_cache(maxsize=1)
def _normalizer() -> DrugEntityNormalizer:
    return DrugEntityNormalizer()


def normalize_query(
    query: str,
    normalizer: DrugEntityNormalizer | None = None,
) -> NormalizedQuery:
    """Normalize a user query into taxonomy-backed retrieval fields."""

    normalizer = normalizer or _normalizer()
    original = query or ""
    normalized_text = normalize_text_key(original)
    expanded = normalizer.expand_query(original)
    metadata = enrich_domain_metadata(original, existing_metadata={})
    entities = [
        entity for entity in expanded.get("normalized_entities", [])
        if isinstance(entity, dict)
    ]

    drug_product = [
        str(entity.get("canonical_name") or "")
        for entity in entities
        if entity.get("entity_type") == "drug_product"
    ]

    aliases: list[str] = []
    for entity in entities:
        aliases.extend(str(alias) for alias in entity.get("aliases", []) or [])

    condition = list(expanded.get("condition") or [])
    safety_context = list(expanded.get("safety_context") or [])
    if _has_negated_pregnancy_context(normalized_text):
        safety_context = [
            value for value in safety_context
            if normalize_text_key(value) not in {"pregnancy", "thai ky", "mang thai"}
        ]
        metadata["safety_context"] = [
            value for value in metadata.get("safety_context", [])
            if normalize_text_key(str(value)) not in {"pregnancy", "thai ky", "mang thai"}
        ]

    acne_metadata = _detect_acne_type_metadata(normalized_text)
    if acne_metadata:
        condition.append("acne_vulgaris")
        metadata.update(acne_metadata)

    intent = _infer_intent(normalized_text, metadata)
    query_intent_hint = list(metadata.get("query_intent_hint") or [])
    if intent not in query_intent_hint:
        query_intent_hint.append(intent)

    confidence = 0.9 if entities else 0.65
    if intent == "general_acne_question" and not entities:
        confidence = 0.5

    return NormalizedQuery(
        original_query=original,
        normalized_text=normalized_text,
        intent=intent,
        drug_product=_dedupe(drug_product),
        active_ingredient=_dedupe(list(expanded.get("active_ingredients") or [])),
        drug_class=_dedupe(list(expanded.get("drug_class") or [])),
        condition=_dedupe(condition),
        safety_context=_dedupe(safety_context),
        query_intent_hint=_dedupe(query_intent_hint),
        aliases=_dedupe(aliases),
        confidence=confidence,
        metadata=metadata,
    )


def _infer_intent(normalized_text: str, metadata: dict[str, Any]) -> str:
    if _contains_any(normalized_text, ["mun dau den", "mụn đầu đen", "blackhead"]):
        return "acne_type"
    if _contains_any(normalized_text, ["mun dau trang", "mụn đầu trắng", "whitehead"]):
        return "acne_type"
    if _contains_any(normalized_text, ["mun viem", "mụn viêm", "mụn bọc", "mụn nang"]):
        return "acne_type"
    if _is_class_group_question(normalized_text):
        return "class_check"
    if _contains_any(normalized_text, ["thuoc nhom", "thuộc nhóm", "nhom gi", "nhóm gì"]):
        return "class_check"
    if _contains_any(normalized_text, ["co phai khang sinh", "có phải kháng sinh", "antibiotic"]):
        return "class_check"
    if _contains_any(normalized_text, ["co phai retinoid", "có phải retinoid", "retinoid"]):
        if "co phai" in normalized_text or "có phải" in normalized_text:
            return "class_check"
    if _is_comparison_question(normalized_text) and _has_drug_metadata(metadata):
        return "comparison"
    if _contains_any(
        normalized_text,
        ["tac dung phu", "tác dụng phụ", "kich ung", "kích ứng", "kho da", "khô da", "do da", "đỏ da"],
    ):
        return "side_effect"
    if _contains_any(
        normalized_text,
        [
            "mang thai",
            "dang mang thai",
            "đang mang thai",
            "co thai",
            "có thai",
            "dang co thai",
            "đang có thai",
            "co bau",
            "có bầu",
            "dang bau",
            "đang bầu",
            "thai ky",
            "thai kỳ",
            "cho con bu",
            "cho con bú",
        ],
    ) and not _has_negated_pregnancy_context(normalized_text):
        return "safety"
    if _contains_any(
        normalized_text,
        [
            "thanh phan",
            "thành phần",
            "hoat chat",
            "hoạt chất",
            "gom nhung hoat chat",
            "gồm những hoạt chất",
            "co bpo",
            "có bpo",
        ],
    ):
        return "ingredient_question"
    if _contains_any(normalized_text, ["la gi", "là gì", "thuoc gi", "thuốc gì"]):
        return "drug_identity" if metadata.get("drug_product") or metadata.get("active_ingredient") else "general_acne_question"
    return "general_acne_question"


def _has_drug_metadata(metadata: dict[str, Any]) -> bool:
    return any(
        metadata.get(field)
        for field in ("drug_product", "active_ingredient", "drug_class")
    )


def _is_comparison_question(normalized_text: str) -> bool:
    return _contains_any(
        normalized_text,
        [
            "so sanh",
            "so sánh",
            "khac nhau",
            "khác nhau",
            "khac gi",
            "khác gì",
            "khac the nao",
            "khác thế nào",
            "khac nhau o",
            "khác nhau ở",
            " vs ",
            "versus",
        ],
    )


def _is_class_group_question(normalized_text: str) -> bool:
    return _contains_any(
        normalized_text,
        [
            "cung nhom",
            "cùng nhóm",
            "cung mot nhom",
            "cùng một nhóm",
            "co cung nhom",
            "có cùng nhóm",
            "thuoc cung nhom",
            "thuộc cùng nhóm",
        ],
    )


def _has_negated_pregnancy_context(normalized_text: str) -> bool:
    ascii_text = _ascii_fold(normalized_text)
    negation_patterns = [
        r"\bkhong\s+(?:co\s+)?(?:mang\s+thai|thai|bau)\b",
        r"\bkhong\s+phai\s+(?:dang\s+)?(?:mang\s+thai|co\s+thai|co\s+bau)\b",
        r"\bnoi\s+nham\b.{0,60}\bkhong\b.{0,20}\b(?:mang\s+thai|co\s+thai|co\s+bau|bau)\b",
    ]
    return any(re.search(pattern, ascii_text) for pattern in negation_patterns)


def _detect_acne_type_metadata(normalized_text: str) -> dict[str, Any]:
    if _contains_any(normalized_text, ["mun dau den", "mụn đầu đen", "blackhead"]):
        return {"concern": ["blackheads", "comedonal_acne"], "content_type": ["acne_type"]}
    if _contains_any(normalized_text, ["mun dau trang", "mụn đầu trắng", "whitehead"]):
        return {"concern": ["whiteheads", "comedonal_acne"], "content_type": ["acne_type"]}
    if _contains_any(normalized_text, ["mun viem", "mụn viêm"]):
        return {"concern": ["inflammatory_acne"], "content_type": ["acne_type", "treatment"]}
    if _contains_any(normalized_text, ["mụn bọc", "mụn nang", "mun boc", "mun nang"]):
        return {"concern": ["severe_acne", "inflammatory_acne"], "content_type": ["acne_type", "treatment"]}
    return {}


def _contains_any(text: str, needles: list[str]) -> bool:
    ascii_text = _ascii_fold(text)
    for needle in needles:
        normalized = normalize_text_key(needle)
        if normalized in text or _ascii_fold(normalized) in ascii_text:
            return True
    return False


def _ascii_fold(text: str) -> str:
    replacements = {"đ": "d", "Đ": "D"}
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    try:
        import unicodedata

        text = "".join(
            char for char in unicodedata.normalize("NFD", text)
            if unicodedata.category(char) != "Mn"
        )
    except Exception:
        pass
    return re.sub(r"\s+", " ", text.lower()).strip()


__all__ = ["normalize_query"]
