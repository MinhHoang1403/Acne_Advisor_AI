"""
domain_metadata.py
==================
Rule-based dermatology metadata extractor.

Provides :func:`extract_dermatology_metadata` which scans a chunk's text
(and optional header path) for domain-specific keywords and returns a
structured metadata dict suitable for enriching ``SemanticChunk.metadata``.

No LLM calls – pure keyword / regex matching.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

from src.ingestion.dermatology_taxonomy import (
    BODY_AREA_KEYWORDS,
    CONCERN_KEYWORDS,
    CONTENT_TYPE_KEYWORDS,
    DOMAIN_TOPIC_KEYWORDS,
    INGREDIENT_KEYWORDS,
    SAFETY_CONTEXT_KEYWORDS,
    SKIN_TYPE_KEYWORDS,
)
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


# ─────────────────────────────────────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DermatologyChunkMetadata:
    """Structured metadata for a single dermatology content chunk."""

    domain_topic: list[str] = field(default_factory=list)
    content_type: list[str] = field(default_factory=list)
    concern: list[str] = field(default_factory=list)
    ingredient: list[str] = field(default_factory=list)
    skin_type: list[str] = field(default_factory=list)
    body_area: list[str] = field(default_factory=list)
    safety_context: list[str] = field(default_factory=list)
    evidence_type: str | None = None
    confidence: float = 0.0
    extraction_method: str = "rule_based"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _match_keywords(
    text_lower: str,
    keyword_dict: dict[str, list[str]],
) -> list[str]:
    """Return deduplicated, stable-ordered list of canonical values matched."""
    matched: list[str] = []
    for canonical, keywords in keyword_dict.items():
        for kw in keywords:
            if kw in text_lower:
                if canonical not in matched:
                    matched.append(canonical)
                break  # one keyword is enough per canonical value
    return matched


def _detect_evidence_type(text_lower: str) -> str | None:
    """Heuristic detection of evidence type from text keywords."""
    evidence_patterns: list[tuple[str, list[str]]] = [
        ("clinical_study", [
            "clinical trial", "thử nghiệm lâm sàng",
            "randomized", "rct", "double-blind", "placebo",
            "nghiên cứu lâm sàng",
        ]),
        ("systematic_review", [
            "systematic review", "meta-analysis", "meta analysis",
            "tổng quan hệ thống",
        ]),
        ("guideline", [
            "guideline", "hướng dẫn điều trị", "phác đồ",
            "consensus", "recommendation",
        ]),
        ("expert_opinion", [
            "expert opinion", "ý kiến chuyên gia",
            "according to dermatologist", "bác sĩ khuyên",
        ]),
        ("in_vitro", [
            "in vitro", "in-vitro", "cell culture", "nuôi cấy tế bào",
        ]),
    ]

    for evidence_type, patterns in evidence_patterns:
        for pattern in patterns:
            if pattern in text_lower:
                return evidence_type
    return None


def _compute_confidence(meta: DermatologyChunkMetadata) -> float:
    """
    Compute a confidence score based on how many metadata fields were populated.

    Strategy:
    - 0.0 if nothing matched at all.
    - 0.3 base if at least one field matched.
    - +0.1 for each additional populated field (up to 7 fields).
    - Capped at 1.0.
    """
    populated_fields = 0
    for field_name in (
        "domain_topic",
        "content_type",
        "concern",
        "ingredient",
        "skin_type",
        "body_area",
        "safety_context",
    ):
        values = getattr(meta, field_name)
        if values:
            populated_fields += 1

    if populated_fields == 0:
        return 0.0

    # Base 0.3 + 0.1 per populated field
    confidence = 0.3 + (populated_fields * 0.1)
    return min(confidence, 1.0)


def _dedupe(values: list[Any]) -> list[str]:
    """Return stable ordered string values with empty items removed."""
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
    """Merge existing chunk metadata with taxonomy-backed entity metadata.

    This function is intentionally rule-based and safe for ingestion: taxonomy
    loading failures are logged as warnings and do not abort the pipeline.
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

    # Preserve old ingredient metadata by mapping only known active ingredients.
    try:
        normalizer = _get_drug_entity_normalizer()
        for old_ingredient in enriched.get("ingredient", []) or []:
            card = normalizer.get_entity_card("active_ingredient", str(old_ingredient))
            if card:
                active_ingredient.append(str(card.metadata.get("taxonomy_key") or card.canonical_name))
    except Exception:
        pass

    # The previous extractor already used safety_context for irritation/dryness.
    # Keep those values and add taxonomy contexts such as pregnancy/breastfeeding.
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


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_dermatology_metadata(
    text: str,
    header_path: str | list[str] = "",
) -> dict[str, Any]:
    """
    Extract dermatology-specific metadata from *text* using rule-based
    keyword matching.

    Parameters
    ----------
    text : str
        The chunk text to analyse.
    header_path : str | list[str]
        Optional Markdown header path (e.g. ``"Acne Treatment > Retinoids"``
        or ``["Acne Treatment", "Retinoids"]``). Will be concatenated and
        matched alongside *text*.

    Returns
    -------
    dict[str, Any]
        A flat dictionary matching the ``DermatologyChunkMetadata`` fields.
        All list values are deduplicated, stable-ordered, lowercase snake_case.
    """
    # Normalise header_path to a string
    if isinstance(header_path, list):
        header_str = " ".join(header_path)
    else:
        header_str = header_path or ""

    # Combine text and header for matching (lowercased)
    combined = f"{header_str} {text}".lower()

    meta = DermatologyChunkMetadata(
        domain_topic=_match_keywords(combined, DOMAIN_TOPIC_KEYWORDS),
        content_type=_match_keywords(combined, CONTENT_TYPE_KEYWORDS),
        concern=_match_keywords(combined, CONCERN_KEYWORDS),
        ingredient=_match_keywords(combined, INGREDIENT_KEYWORDS),
        skin_type=_match_keywords(combined, SKIN_TYPE_KEYWORDS),
        body_area=_match_keywords(combined, BODY_AREA_KEYWORDS),
        safety_context=_match_keywords(combined, SAFETY_CONTEXT_KEYWORDS),
        evidence_type=_detect_evidence_type(combined),
        extraction_method="rule_based",
    )

    meta.confidence = _compute_confidence(meta)

    return meta.to_dict()
