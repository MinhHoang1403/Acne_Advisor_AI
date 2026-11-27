"""Xây dựng và đồng bộ đồ thị tri thức hai tầng (Lexical + Clinical Facet Graph) trên Neo4j.

Module này mở rộng Neo4j từ một bảng tra cứu taxonomy tĩnh (32 thực thể) thành
tầng truy xuất GraphRAG thực thụ cho toàn bộ kho tri thức y khoa đã biên dịch:

1. Tầng Cấu trúc & Provenance (Lexical Graph):
   - ``(:KnowledgeSource)-[:PUBLISHED_DOCUMENT]->(:KnowledgeDocument)``
   - ``(:KnowledgeDocument)-[:HAS_CHUNK]->(:KnowledgeChunk)``
   - ``(:KnowledgeChunk)-[:NEXT_CHUNK]->(:KnowledgeChunk)`` (bảo toàn mạch bảng biểu
     và đoạn văn liền kề trong cùng tài liệu nguồn).

2. Tầng Thực thể & Khía cạnh Lâm sàng (Clinical Entity & Facet Graph):
   - ``(:KnowledgeChunk)-[:MENTIONS_ENTITY]->(:DrugProduct|:ActiveIngredient|:DrugClass|:SafetyContext)``
   - ``(:KnowledgeChunk)-[:HAS_CLINICAL_FACET]->(:ClinicalFacet)``
   - ``(:ActiveIngredient|:DrugClass)-[:CLINICAL_INTERACTION]->(:ActiveIngredient|:DrugClass)``

Thiết kế tuân thủ nghiêm ngặt hợp đồng bất biến của dự án:
- Tách biệt hoàn toàn tập nhãn ``LEXICAL_GRAPH_LABELS`` và quan hệ
  ``LEXICAL_GRAPH_RELATIONSHIPS`` khỏi ``ENTITY_GRAPH_LABELS`` /
  ``ENTITY_GRAPH_RELATIONSHIPS`` để ``validate_entity_graph_records()`` luôn giữ
  nguyên 32 nút / 27 quan hệ chuẩn.
- Không neo đồ thị vào thực thể chung ``Condition:acne_vulgaris`` nhằm triệt tiêu
  nhiễu toàn cục khi truy xuất các câu hỏi tổng quát.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
import re
from typing import Any, Iterable, Mapping
import unicodedata

from src.knowledge.graph_schema import NEO4J_SCHEMA_VERSION
from src.knowledge.normalizer import (
    DrugEntityNormalizer,
    normalize_text_key,
)

GRAPH_CONTRACT_VERSION = NEO4J_SCHEMA_VERSION
_DEFAULT_NORMALIZER: DrugEntityNormalizer | None = None


def _get_default_normalizer() -> DrugEntityNormalizer:
    global _DEFAULT_NORMALIZER
    if _DEFAULT_NORMALIZER is None:
        _DEFAULT_NORMALIZER = DrugEntityNormalizer()
    return _DEFAULT_NORMALIZER


def normalize_lookup_text(value: str | None) -> str:
    """Chuẩn hóa văn bản tra cứu giữ dấu tiếng Việt nhất quán với DrugEntityNormalizer."""
    return normalize_text_key(value or "")

LEXICAL_GRAPH_LABELS: tuple[str, ...] = (
    "KnowledgeSource",
    "KnowledgeDocument",
    "KnowledgeChunk",
    "ClinicalFacet",
)

LEXICAL_GRAPH_RELATIONSHIPS: tuple[str, ...] = (
    "PUBLISHED_DOCUMENT",
    "HAS_CHUNK",
    "NEXT_CHUNK",
    "MENTIONS_ENTITY",
    "HAS_CLINICAL_FACET",
    "CLINICAL_INTERACTION",
)

SPECIFIC_ENTITY_GROUPS: tuple[str, ...] = (
    "drug_product",
    "active_ingredient",
    "drug_class",
    "safety_context",
)

ENTITY_GROUP_TO_LABEL: dict[str, str] = {
    "drug_product": "DrugProduct",
    "active_ingredient": "ActiveIngredient",
    "drug_class": "DrugClass",
    "safety_context": "SafetyContext",
}


@dataclass(frozen=True)
class ClinicalFacetSpec:
    """Đặc tả tất định cho một nút khía cạnh lâm sàng (ClinicalFacet)."""

    facet_key: str
    facet_group: str
    display_name: str
    aliases: tuple[str, ...]
    metadata_matches: tuple[tuple[str, str], ...] = ()


# Danh mục khía cạnh lâm sàng song ngữ (Việt - Anh) bám sát metadata và thuật ngữ
# xuất hiện thực tế trong 53 nguồn y khoa đã kiểm duyệt (AAD 2024, NICE NG198, Bộ Y tế).
CLINICAL_FACET_CATALOG: tuple[ClinicalFacetSpec, ...] = (
    # --- Phân thể mụn & biến thể đặc hiệu (acne_subtype) ---
    ClinicalFacetSpec(
        facet_key="comedonal_acne",
        facet_group="acne_subtype",
        display_name="Mụn nhân (đầu trắng / đầu đen)",
        aliases=(
            "comedonal acne",
            "mụn đầu đen",
            "mụn đầu trắng",
            "mụn ẩn",
            "nhân mụn đóng",
            "nhân mụn mở",
            "open comedone",
            "closed comedone",
        ),
        metadata_matches=(("acne_type", "Comedonal acne"),),
    ),
    ClinicalFacetSpec(
        facet_key="papular_pustular_acne",
        facet_group="acne_subtype",
        display_name="Mụn sẩn viêm và mụn mủ",
        aliases=(
            "papular/pustular acne",
            "papulopustular acne",
            "mụn sẩn",
            "mụn mủ",
            "sẩn mủ",
        ),
        metadata_matches=(("acne_type", "Papular/Pustular acne"),),
    ),
    ClinicalFacetSpec(
        facet_key="nodulocystic_acne",
        facet_group="acne_subtype",
        display_name="Mụn cục / mụn bọc / mụn nang nặng",
        aliases=(
            "nodular acne",
            "nodulocystic acne",
            "mụn bọc",
            "mụn nang",
            "mụn cục",
            "nốt nang",
        ),
        metadata_matches=(("acne_type", "Nodular/Nodulocystic acne"),),
    ),
    ClinicalFacetSpec(
        facet_key="acne_conglobata",
        facet_group="acne_subtype",
        display_name="Trứng cá mạch lươn (Acne conglobata)",
        aliases=(
            "acne conglobata",
            "trứng cá mạch lươn",
            "mụn mạch lươn",
        ),
        metadata_matches=(("acne_type", "Acne conglobata"),),
    ),
    ClinicalFacetSpec(
        facet_key="acne_fulminans",
        facet_group="acne_subtype",
        display_name="Trứng cá tối cấp (Acne fulminans)",
        aliases=(
            "acne fulminans",
            "trứng cá tối cấp",
            "mụn tối cấp",
        ),
        metadata_matches=(("acne_type", "Acne fulminans"),),
    ),
    ClinicalFacetSpec(
        facet_key="steroid_acne",
        facet_group="acne_subtype",
        display_name="Trứng cá do thuốc / Steroid acne",
        aliases=(
            "steroid acne",
            "trứng cá do steroid",
            "mụn do steroid",
            "mụn do corticoid",
            "trứng cá do thuốc",
            "drug-induced acne",
            "corticosteroid đường toàn thân",
            "hội chứng cushing",
            "cushing",
            "steroid đồng hóa",
            "anabolic steroid",
        ),
    ),
    ClinicalFacetSpec(
        facet_key="gram_negative_folliculitis",
        facet_group="acne_subtype",
        display_name="Viêm nang lông do vi khuẩn Gram âm",
        aliases=(
            "gram-negative folliculitis",
            "viêm nang lông do vi khuẩn gram âm",
            "viêm nang lông gram âm",
            "vi khuẩn gram âm",
        ),
        metadata_matches=(("differential", "Folliculitis"),),
    ),
    ClinicalFacetSpec(
        facet_key="acne_excoriee",
        facet_group="acne_subtype",
        display_name="Trứng cá do cào gãi (Acne excoriée)",
        aliases=(
            "acne excoriee",
            "acne excoriée",
            "trứng cá do cào gãi",
            "tự cào gãi mụn",
        ),
    ),
    ClinicalFacetSpec(
        facet_key="neonatal_pediatric_acne",
        facet_group="acne_subtype",
        display_name="Mụn sơ sinh, nhũ nhi và trẻ nhỏ",
        aliases=(
            "neonatal acne",
            "infantile acne",
            "mid-childhood acne",
            "preadolescent acne",
            "pediatric acne",
            "trứng cá sơ sinh",
            "mụn sơ sinh",
            "trứng cá nhũ nhi",
            "mụn mủ đầu mặt sơ sinh",
            "neonatal cephalic pustulosis",
            "malassezia",
        ),
        metadata_matches=(
            ("acne_type", "Pediatric acne"),
            ("special_group", "Pediatric"),
        ),
    ),
    ClinicalFacetSpec(
        facet_key="truncal_acne",
        facet_group="acne_subtype",
        display_name="Mụn thân mình (lưng, ngực, vai)",
        aliases=(
            "truncal acne",
            "mụn lưng",
            "mụn ngực",
            "mụn thân mình",
            "vùng lưng và ngực",
        ),
        metadata_matches=(("acne_type", "Truncal acne"),),
    ),
    # --- Chăm sóc da hỗ trợ & lối sống (care_topic) ---
    ClinicalFacetSpec(
        facet_key="cleansing_care",
        facet_group="care_topic",
        display_name="Làm sạch da và sữa rửa mặt dịu nhẹ",
        aliases=(
            "sữa rửa mặt",
            "rửa mặt",
            "làm sạch da",
            "gentle cleanser",
            "cân bằng ph",
            "chà xát mạnh",
        ),
        metadata_matches=(("care_topic", "cleaning"),),
    ),
    ClinicalFacetSpec(
        facet_key="moisturizing_barrier",
        facet_group="care_topic",
        display_name="Dưỡng ẩm không gây bít tắc (Non-comedogenic / Oil-free)",
        aliases=(
            "kem dưỡng ẩm",
            "dưỡng ẩm",
            "moisturizer",
            "non-comedogenic",
            "không gây nhân mụn",
            "không gây bít tắc",
            "oil-free",
            "không chứa dầu",
            "hàng rào bảo vệ da",
        ),
        metadata_matches=(
            ("care_topic", "moisturizing"),
            ("care_topic", "recommended_ingredients"),
        ),
    ),
    ClinicalFacetSpec(
        facet_key="care_during_acne_therapy",
        facet_group="care_topic",
        display_name="Chăm sóc da khô / kích ứng khi đang dùng thuốc trị mụn",
        aliases=(
            "đang điều trị mụn",
            "khi điều trị mụn",
            "trong quá trình điều trị mụn",
            "khô da bong tróc",
            "giảm kích ứng khi bôi",
            "thoa kem dưỡng ẩm sau",
        ),
        metadata_matches=(("care_topic", "with_acne_treatment"),),
    ),
    ClinicalFacetSpec(
        facet_key="comedogenic_ingredients_avoid",
        facet_group="care_topic",
        display_name="Thành phần và mỹ phẩm gây bít tắc cần tránh",
        aliases=(
            "thành phần cần tránh",
            "mỹ phẩm gây mụn",
            "acne cosmetica",
            "dầu khoáng đặc",
            "lanolin",
            "isopropyl myristate",
            "dầu dừa",
        ),
        metadata_matches=(("care_topic", "ingredients_to_avoid"),),
    ),
    ClinicalFacetSpec(
        facet_key="sun_protection_photosensitivity",
        facet_group="care_topic",
        display_name="Chống nắng và nhạy cảm ánh sáng",
        aliases=(
            "kem chống nắng",
            "chống nắng",
            "sunscreen",
            "nhạy cảm ánh sáng",
            "nhạy cảm với ánh nắng",
            "photosensitivity",
            "tia uv",
            "spf 30",
        ),
        metadata_matches=(("care_topic", "sun_protection"),),
    ),
    ClinicalFacetSpec(
        facet_key="comedone_extraction_picking",
        facet_group="care_topic",
        display_name="Lấy nhân mụn và tác hại tự nặn mụn",
        aliases=(
            "lấy nhân mụn",
            "nặn mụn",
            "tự nặn mụn",
            "comedone extraction",
            "dụng cụ lấy nhân mụn",
        ),
    ),
    ClinicalFacetSpec(
        facet_key="sensitive_skin_fragrance_free",
        facet_group="care_topic",
        display_name="Chăm sóc da nhạy cảm / không hương liệu",
        aliases=(
            "da nhạy cảm",
            "sensitive skin",
            "không hương liệu",
            "fragrance-free",
            "không chứa cồn",
        ),
    ),
    ClinicalFacetSpec(
        facet_key="diet_glycemic_dairy",
        facet_group="care_topic",
        display_name="Chế độ ăn, chỉ số đường huyết và sữa",
        aliases=(
            "chỉ số đường huyết",
            "glycemic index",
            "high glycemic",
            "sô cô la",
            "sô-cô-la",
            "chocolate",
            "sữa tách béo",
            "skim milk",
            "whey protein",
            "đồ ngọt",
        ),
    ),
    # --- Nhóm bệnh nhân đặc biệt & nội tiết (special_group) ---
    ClinicalFacetSpec(
        facet_key="hormonal_pcos_androgen",
        facet_group="special_group",
        display_name="Mụn nội tiết, PCOS và cường androgen",
        aliases=(
            "mụn nội tiết",
            "hormonal acne",
            "buồng trứng đa nang",
            "đa nang buồng trứng",
            "pcos",
            "cường androgen",
            "hyperandrogenism",
            "chu kỳ kinh nguyệt",
            "rậm lông",
        ),
        metadata_matches=(
            ("acne_type", "Adult/Hormonal acne"),
            ("special_group", "Hormonal"),
            ("special_group", "Adult female"),
        ),
    ),
    ClinicalFacetSpec(
        facet_key="pregnancy_teratogenicity",
        facet_group="special_group",
        display_name="An toàn thai kỳ và nguy cơ quái thai",
        aliases=(
            "mang thai",
            "thai kỳ",
            "phụ nữ có thai",
            "dự định mang thai",
            "quái thai",
            "dị tật thai nhi",
            "teratogenic",
            "ipledge",
            "biện pháp tránh thai",
        ),
        metadata_matches=(("special_group", "Pregnancy"),),
    ),
    ClinicalFacetSpec(
        facet_key="breastfeeding_lactation",
        facet_group="special_group",
        display_name="An toàn thời kỳ cho con bú",
        aliases=(
            "cho con bú",
            "phụ nữ đang cho con bú",
            "sữa mẹ",
            "breastfeeding",
            "lactation",
        ),
        metadata_matches=(("special_group", "Breastfeeding"),),
    ),
    # --- Khái niệm lâm sàng chuyên sâu (clinical_concept) ---
    ClinicalFacetSpec(
        facet_key="antibiotic_stewardship",
        facet_group="clinical_concept",
        display_name="Phòng ngừa kháng kháng sinh và giới hạn đơn trị liệu",
        aliases=(
            "kháng kháng sinh",
            "kháng thuốc",
            "antibiotic resistance",
            "đơn trị liệu",
            "monotherapy",
            "tối đa 3-4 tháng",
            "12 tuần",
        ),
    ),
    ClinicalFacetSpec(
        facet_key="intracranial_hypertension_interaction",
        facet_group="clinical_concept",
        display_name="Tăng áp lực nội sọ lành tính (Pseudotumor cerebri)",
        aliases=(
            "tăng áp lực nội sọ",
            "u giả não",
            "pseudotumor cerebri",
            "intracranial hypertension",
        ),
    ),
    ClinicalFacetSpec(
        facet_key="isotretinoin_lab_monitoring",
        facet_group="clinical_concept",
        display_name="Theo dõi xét nghiệm gan và lipid máu khi dùng Isotretinoin",
        aliases=(
            "chức năng gan",
            "men gan",
            "alt",
            "ast",
            "mỡ máu",
            "triglyceride",
            "cholesterol",
            "lipid máu",
            "công thức máu",
        ),
    ),
    ClinicalFacetSpec(
        facet_key="acne_scarring_sequelae",
        facet_group="clinical_concept",
        display_name="Sẹo mụn (Ice pick, Boxcar, Rolling, Sẹo lồi) và tăng sắc tố sau viêm",
        aliases=(
            "sẹo rỗ",
            "sẹo lõm",
            "ice pick",
            "sẹo đáy nhọn",
            "boxcar",
            "sẹo đáy vuông",
            "sẹo đáy phẳng",
            "rolling",
            "sẹo lượn sóng",
            "sẹo lòng chảo",
            "sẹo phì đại",
            "sẹo lồi",
            "keloid",
            "tăng sắc tố sau viêm",
            "thâm mụn",
            "pih",
        ),
    ),
    ClinicalFacetSpec(
        facet_key="pathogenesis_four_pillars",
        facet_group="clinical_concept",
        display_name="Bốn cơ chế bệnh sinh cốt lõi của mụn trứng cá",
        aliases=(
            "cơ chế bệnh sinh",
            "sinh bệnh học",
            "bốn cơ chế",
            "4 cơ chế",
            "tăng tiết bã nhờn",
            "dày sừng cổ nang lông",
            "sừng hóa cổ nang lông",
            "cutibacterium acnes",
            "propionibacterium acnes",
            "c. acnes",
            "p. acnes",
        ),
    ),
)

FACET_BY_KEY: dict[str, ClinicalFacetSpec] = {
    spec.facet_key: spec for spec in CLINICAL_FACET_CATALOG
}


# Các tương tác / phối hợp thuốc chuẩn hóa theo guideline lâm sàng (AAD / NICE / Bộ Y tế).
# Dùng kiểu quan hệ riêng CLINICAL_INTERACTION để không làm thay đổi bộ đếm 27 quan hệ gốc.
CLINICAL_INTERACTION_CATALOG: tuple[dict[str, str], ...] = (
    {
        "source_label": "ActiveIngredient",
        "source_name": "isotretinoin",
        "target_label": "ActiveIngredient",
        "target_name": "doxycycline",
        "interaction_type": "contraindicated_cotherapy",
        "clinical_note": "Chống chỉ định phối hợp do nguy cơ tăng áp lực nội sọ lành tính (pseudotumor cerebri).",
    },
    {
        "source_label": "ActiveIngredient",
        "source_name": "isotretinoin",
        "target_label": "ActiveIngredient",
        "target_name": "minocycline",
        "interaction_type": "contraindicated_cotherapy",
        "clinical_note": "Chống chỉ định phối hợp do nguy cơ tăng áp lực nội sọ lành tính (pseudotumor cerebri).",
    },
    {
        "source_label": "ActiveIngredient",
        "source_name": "isotretinoin",
        "target_label": "ActiveIngredient",
        "target_name": "sarecycline",
        "interaction_type": "contraindicated_cotherapy",
        "clinical_note": "Chống chỉ định phối hợp kháng sinh nhóm tetracycline (sarecycline) đường uống với isotretinoin do nguy cơ tăng áp lực nội sọ lành tính.",
    },
    {
        "source_label": "ActiveIngredient",
        "source_name": "isotretinoin",
        "target_label": "DrugClass",
        "target_name": "oral_antibiotic",
        "interaction_type": "contraindicated_cotherapy",
        "clinical_note": "Tránh phối hợp isotretinoin với kháng sinh nhóm cyclin đường uống.",
    },
    {
        "source_label": "ActiveIngredient",
        "source_name": "benzoyl_peroxide",
        "target_label": "ActiveIngredient",
        "target_name": "clindamycin",
        "interaction_type": "antiresistance_synergy",
        "clinical_note": "Phối hợp bắt buộc để ngăn ngừa kháng kháng sinh của C. acnes.",
    },
    {
        "source_label": "ActiveIngredient",
        "source_name": "benzoyl_peroxide",
        "target_label": "ActiveIngredient",
        "target_name": "erythromycin",
        "interaction_type": "antiresistance_synergy",
        "clinical_note": "Phối hợp bắt buộc để giảm đề kháng kháng sinh tại chỗ.",
    },
    {
        "source_label": "ActiveIngredient",
        "source_name": "benzoyl_peroxide",
        "target_label": "DrugClass",
        "target_name": "topical_antibiotic",
        "interaction_type": "antiresistance_synergy",
        "clinical_note": "Không dùng kháng sinh bôi đơn trị liệu; luôn phối hợp Benzoyl Peroxide.",
    },
    {
        "source_label": "ActiveIngredient",
        "source_name": "benzoyl_peroxide",
        "target_label": "ActiveIngredient",
        "target_name": "doxycycline",
        "interaction_type": "antiresistance_synergy",
        "clinical_note": "Kháng sinh đường uống cần phối hợp Benzoyl Peroxide hoặc retinoid bôi tại chỗ.",
    },
    {
        "source_label": "ActiveIngredient",
        "source_name": "benzoyl_peroxide",
        "target_label": "ActiveIngredient",
        "target_name": "adapalene",
        "interaction_type": "fixed_dose_synergy",
        "clinical_note": "Adapalene bền vững hóa học khi dùng chung với Benzoyl Peroxide.",
    },
    {
        "source_label": "ActiveIngredient",
        "source_name": "tretinoin",
        "target_label": "ActiveIngredient",
        "target_name": "benzoyl_peroxide",
        "interaction_type": "oxidation_timing_caution",
        "clinical_note": "Benzoyl Peroxide có thể oxy hóa tretinoin thường nếu bôi cùng thời điểm.",
    },
)


def fold_ascii_text(value: str | None) -> str:
    """Bỏ dấu tiếng Việt an toàn để bắt khớp truy vấn thiếu dấu hoặc gõ tắt."""

    normalized = normalize_lookup_text(value)
    if not normalized:
        return ""
    replaced = normalized.replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", replaced)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return " ".join(stripped.split())


def _compile_phrase_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase)
    return re.compile(rf"(?<![0-9a-zA-ZÀ-ỹ_]){escaped}(?![0-9a-zA-ZÀ-ỹ_])")


def _build_compiled_facet_patterns() -> list[tuple[ClinicalFacetSpec, tuple[re.Pattern[str], ...], tuple[re.Pattern[str], ...]]]:
    compiled: list[tuple[ClinicalFacetSpec, tuple[re.Pattern[str], ...], tuple[re.Pattern[str], ...]]] = []
    for spec in CLINICAL_FACET_CATALOG:
        accented_aliases = sorted(
            {normalize_lookup_text(alias) for alias in spec.aliases if normalize_lookup_text(alias)},
            key=lambda item: (-len(item), item),
        )
        folded_aliases = sorted(
            {
                fold_ascii_text(alias)
                for alias in spec.aliases
                if len(fold_ascii_text(alias)) >= 4
            },
            key=lambda item: (-len(item), item),
        )
        compiled.append(
            (
                spec,
                tuple(_compile_phrase_pattern(alias) for alias in accented_aliases),
                tuple(_compile_phrase_pattern(alias) for alias in folded_aliases),
            )
        )
    return compiled


_COMPILED_FACET_PATTERNS = _build_compiled_facet_patterns()


# Cầu nối biệt dược thương mại phổ biến tại Việt Nam & quốc tế sang thực thể chuẩn
# trên Neo4j mà KHÔNG sửa đổi file đóng băng data/taxonomy/drug_aliases.yaml
# (bảo toàn tuyệt đối mã định danh build b768fae999391f47760a).
EXTENDED_CLINICAL_BRAND_SYNONYMS: tuple[tuple[str, tuple[str, ...], tuple[tuple[str, str], ...]], ...] = (
    (
        "Klenzit-C",
        ("klenzit-c", "klenzit c"),
        (("active_ingredient", "adapalene"), ("active_ingredient", "clindamycin")),
    ),
    (
        "Klenzit-MS",
        ("klenzit-ms", "klenzit ms"),
        (("active_ingredient", "adapalene"),),
    ),
    (
        "Azaduo",
        ("azaduo",),
        (("active_ingredient", "adapalene"), ("active_ingredient", "benzoyl_peroxide")),
    ),
    (
        "Dermaforte",
        ("dermaforte", "derma forte"),
        (("active_ingredient", "azelaic_acid"),),
    ),
    (
        "Skinoren",
        ("skinoren", "finacea"),
        (("active_ingredient", "azelaic_acid"),),
    ),
    (
        "Roaccutane",
        ("roaccutane", "accutane", "acnotin", "nimegen", "curacne"),
        (("active_ingredient", "isotretinoin"),),
    ),
    (
        "Retacnyl",
        ("retacnyl", "retin-a", "retin a", "acnelyse", "tretiheal", "locacid"),
        (("active_ingredient", "tretinoin"),),
    ),
    (
        "Benzac AC",
        ("benzac ac", "benzac", "panoxyl", "oxy 5", "oxy 10"),
        (("active_ingredient", "benzoyl_peroxide"),),
    ),
    (
        "Erylik",
        ("erylik",),
        (("active_ingredient", "erythromycin"), ("active_ingredient", "tretinoin")),
    ),
    (
        "Zineryt",
        ("zineryt", "eryfluid"),
        (("active_ingredient", "erythromycin"),),
    ),
    (
        "Tazorac",
        ("tazorac", "tazret"),
        (("active_ingredient", "tazarotene"),),
    ),
    (
        "Aldactone",
        ("aldactone",),
        (("active_ingredient", "spironolactone"),),
    ),
    (
        "Diane-35",
        ("diane-35", "diane 35", "yaz", "yasmin"),
        (("drug_class", "combined_oral_contraceptive"),),
    ),
)


@dataclass(frozen=True)
class QueryGraphAnchors:
    """Tập điểm neo (anchors) được trích xuất từ câu hỏi để truy vấn đồ thị Neo4j."""

    entities_by_group: dict[str, list[str]] = field(default_factory=dict)
    facets_by_group: dict[str, list[str]] = field(default_factory=dict)
    brand_bridge_paths: tuple[str, ...] = ()

    @property
    def entity_names(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for group in SPECIFIC_ENTITY_GROUPS:
            for name in self.entities_by_group.get(group, ()):
                if name not in seen:
                    seen.add(name)
                    ordered.append(name)
        return ordered

    @property
    def facet_keys(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for keys in self.facets_by_group.values():
            for key in keys:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        return ordered

    @property
    def total_anchor_count(self) -> int:
        return len(self.entity_names) + len(self.facet_keys)

    @property
    def has_specific_anchors(self) -> bool:
        return self.total_anchor_count > 0

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "entities": self.entity_names,
            "entities_by_group": dict(self.entities_by_group),
            "facets": self.facet_keys,
            "facets_by_group": dict(self.facets_by_group),
            "brand_bridge_paths": list(self.brand_bridge_paths),
            "anchor_count": self.total_anchor_count,
        }


def extract_clinical_facets_from_text(
    text: str | None,
    *,
    allow_ascii_fallback: bool = False,
) -> dict[str, list[str]]:
    """Trích xuất các khía cạnh lâm sàng (ClinicalFacet) từ văn bản tiếng Việt / Anh."""

    normalized = normalize_lookup_text(text)
    if not normalized:
        return {}
    folded = fold_ascii_text(text) if allow_ascii_fallback else ""
    has_non_ascii = any(ord(ch) > 127 for ch in normalized)

    matched_by_group: dict[str, list[str]] = defaultdict(list)
    for spec, accented_patterns, folded_patterns in _COMPILED_FACET_PATTERNS:
        matched = any(pattern.search(normalized) for pattern in accented_patterns)
        # Chỉ kích hoạt ASCII-folding cho truy vấn người dùng gõ không dấu (tránh
        # bắt nhầm từ đồng âm khi đánh chỉ mục tài liệu chuẩn có đầy đủ dấu).
        if not matched and allow_ascii_fallback and not has_non_ascii and folded:
            matched = any(pattern.search(folded) for pattern in folded_patterns)
        if matched:
            matched_by_group[spec.facet_group].append(spec.facet_key)

    return {group: sorted(set(keys)) for group, keys in matched_by_group.items() if keys}


def _build_compiled_brand_patterns() -> list[tuple[str, tuple[tuple[str, re.Pattern[str]], ...], tuple[tuple[str, str], ...]]]:
    compiled: list[tuple[str, tuple[tuple[str, re.Pattern[str]], ...], tuple[tuple[str, str], ...]]] = []
    for brand_display, aliases, targets in EXTENDED_CLINICAL_BRAND_SYNONYMS:
        alias_patterns: list[tuple[str, re.Pattern[str]]] = []
        for alias in aliases:
            norm_alias = normalize_lookup_text(alias)
            if norm_alias:
                display_label = brand_display if norm_alias in normalize_lookup_text(brand_display) else alias.title()
                alias_patterns.append((display_label, _compile_phrase_pattern(norm_alias)))
        compiled.append((brand_display, tuple(alias_patterns), targets))
    return compiled


_COMPILED_BRAND_PATTERNS = _build_compiled_brand_patterns()


def _extract_extended_brand_bridges(
    text: str,
    normalizer: DrugEntityNormalizer,
) -> tuple[dict[str, set[str]], list[str]]:
    """Ánh xạ biệt dược thương mại mở rộng sang hoạt chất/nhóm thuốc chuẩn trên Neo4j."""

    normalized = normalize_lookup_text(text)
    if not normalized:
        return {}, []
    matched_entities: dict[str, set[str]] = defaultdict(set)
    bridge_paths: list[str] = []
    for _default_brand, alias_patterns, targets in _COMPILED_BRAND_PATTERNS:
        matched_label: str | None = None
        for display_label, pattern in alias_patterns:
            if pattern.search(normalized):
                matched_label = display_label
                break
        if matched_label is not None:
            for target_group, target_key in targets:
                card = normalizer.get_entity_card(target_group, target_key)
                if card is not None:
                    matched_entities[target_group].add(card.canonical_name)
                    rel_name = "BELONGS_TO_CLASS" if target_group == "drug_class" else "HAS_ACTIVE_INGREDIENT"
                    path = f"{matched_label}-[{rel_name}]->{card.canonical_name}"
                    if path not in bridge_paths:
                        bridge_paths.append(path)
    return matched_entities, bridge_paths


def _extract_specific_entities(
    text: str,
    normalizer: DrugEntityNormalizer,
    *,
    filter_contrast: bool = False,
) -> tuple[dict[str, list[str]], list[str]]:
    """Trích xuất các thực thể y khoa đặc hiệu và chuẩn hóa sang đúng card.canonical_name."""

    cards = normalizer.match_alias(text)
    if filter_contrast:
        cards = normalizer._filter_contrast_class_mentions(text, cards)
    matched: dict[str, set[str]] = defaultdict(set)
    for card in cards:
        if card.entity_type in SPECIFIC_ENTITY_GROUPS:
            matched[card.entity_type].add(card.canonical_name)

    bridged, bridge_paths = _extract_extended_brand_bridges(text, normalizer)
    for group, names in bridged.items():
        matched[group].update(names)

    return {group: sorted(names) for group, names in matched.items() if names}, bridge_paths


def extract_query_graph_anchors(
    query: str | None,
    *,
    taxonomy: DrugEntityNormalizer | None = None,
) -> QueryGraphAnchors:
    """Trích xuất điểm neo thực thể & khía cạnh lâm sàng từ câu hỏi của người dùng.

    Quy tắc chống nhiễu:
    - Loại bỏ hoàn toàn thực thể chung ``Condition:acne_vulgaris``.
    - Nếu câu hỏi quá chung (vd: "mụn là gì") và không khớp bất kỳ hoạt chất,
      biệt dược, đối tượng an toàn hay phân thể/chủ đề cụ thể nào, hàm trả về
      ``has_specific_anchors == False`` để kênh Graph trả rỗng tức thì (0 nhiễu).
    """

    clean_query = (query or "").strip()
    if not clean_query:
        return QueryGraphAnchors()

    normalizer = taxonomy or _get_default_normalizer()
    entities_by_group, bridge_paths = _extract_specific_entities(
        clean_query,
        normalizer,
        filter_contrast=True,
    )

    facets_by_group = extract_clinical_facets_from_text(
        clean_query,
        allow_ascii_fallback=True,
    )
    return QueryGraphAnchors(
        entities_by_group=entities_by_group,
        facets_by_group=facets_by_group,
        brand_bridge_paths=tuple(bridge_paths),
    )


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, Iterable):
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text)
        return out
    return []


def _normalize_point_record(raw: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Chuẩn hóa bản ghi đầu vào từ Qdrant point hoặc compiled build record."""

    if "payload" in raw and isinstance(raw["payload"], Mapping):
        payload = dict(raw["payload"])
        point_id = str(raw.get("id") or payload.get("chunk_id") or "").strip()
    else:
        payload = dict(raw)
        point_id = str(raw.get("point_id") or raw.get("id") or raw.get("chunk_id") or "").strip()

    if not point_id:
        raise ValueError("Knowledge chunk record is missing a stable point_id/chunk_id")
    payload.setdefault("chunk_id", point_id)
    return point_id, payload


def build_lexical_graph_records(
    chunk_records: Iterable[Mapping[str, Any]],
    *,
    build_id: str,
    taxonomy: DrugEntityNormalizer | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Biên dịch toàn bộ nút và cạnh của Lexical + Clinical Facet Graph từ tập chunks.

    Hàm hoàn toàn tất định (deterministic), không gọi LLM hay API ngoài.
    """

    clean_build_id = str(build_id or "").strip()
    if not clean_build_id:
        raise ValueError("build_id is required when building lexical graph records")

    normalizer = taxonomy or _get_default_normalizer()
    label_to_group = {val: key for key, val in ENTITY_GROUP_TO_LABEL.items()}

    sources_by_id: dict[str, dict[str, Any]] = {}
    documents_by_id: dict[str, dict[str, Any]] = {}
    chunks: list[dict[str, Any]] = []
    chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)

    published_pairs: set[tuple[str, str]] = set()
    has_chunk_edges: list[dict[str, Any]] = []
    mentions_entity_edges: list[dict[str, Any]] = []
    has_facet_edges: list[dict[str, Any]] = []

    normalized_inputs = [_normalize_point_record(item) for item in chunk_records]
    normalized_inputs.sort(key=lambda pair: (str(pair[1].get("document_id") or ""), int(pair[1].get("chunk_index") or 0), pair[0]))

    for point_id, payload in normalized_inputs:
        source_id = str(payload.get("source_id") or payload.get("document_id") or "unknown_source").strip()
        document_id = str(payload.get("document_id") or source_id).strip()
        source_title = str(payload.get("source_title") or payload.get("document_title") or source_id).strip()
        document_title = str(payload.get("document_title") or source_title).strip()
        chunk_index = int(payload.get("chunk_index") or 0)
        content_kind = str(payload.get("content_kind") or "text").strip()
        section_title = str(payload.get("section_title") or payload.get("header") or "").strip()
        text = str(payload.get("text") or "").strip()

        if source_id not in sources_by_id:
            sources_by_id[source_id] = {
                "source_id": source_id,
                "title": source_title,
                "source_authority": str(payload.get("source_authority") or "").strip(),
                "source_type": str(payload.get("source_type") or "").strip(),
                "source_url": str(payload.get("source_url") or "").strip(),
                "scope": str(payload.get("scope") or "").strip(),
                "category": str(payload.get("category") or "").strip(),
                "kb_version": clean_build_id,
                "schema_version": GRAPH_CONTRACT_VERSION,
            }

        if document_id not in documents_by_id:
            documents_by_id[document_id] = {
                "document_id": document_id,
                "source_id": source_id,
                "title": document_title,
                "source_title": source_title,
                "kb_version": clean_build_id,
                "schema_version": GRAPH_CONTRACT_VERSION,
            }

        published_pairs.add((source_id, document_id))

        # Lưu nguyên bản JSON payload để Neo4j trả về candidate đồng nhất 100% với Qdrant.
        payload_copy = dict(payload)
        payload_copy["chunk_id"] = point_id
        payload_copy["build_id"] = clean_build_id

        chunk_node = {
            "point_id": point_id,
            "chunk_id": str(payload.get("chunk_id") or point_id),
            "document_id": document_id,
            "source_id": source_id,
            "source_title": source_title,
            "document_title": document_title,
            "section_title": section_title,
            "content_kind": content_kind,
            "chunk_index": chunk_index,
            "table_index": int(payload["table_index"]) if payload.get("table_index") is not None else -1,
            "text": text,
            "payload_json": json.dumps(payload_copy, ensure_ascii=False, sort_keys=True),
            "kb_version": clean_build_id,
            "schema_version": GRAPH_CONTRACT_VERSION,
        }
        chunks.append(chunk_node)
        chunks_by_doc[document_id].append(chunk_node)

        has_chunk_edges.append(
            {
                "document_id": document_id,
                "point_id": point_id,
                "chunk_index": chunk_index,
                "kb_version": clean_build_id,
            }
        )

        # 1. Liên kết MENTIONS_ENTITY từ cả metadata đã chuẩn hóa lẫn khớp từ vựng văn bản
        searchable_chunk_text = " ".join(
            part for part in (section_title, str(payload.get("header") or ""), text) if part
        )
        lexical_entities, _ = _extract_specific_entities(searchable_chunk_text, normalizer)

        for group in SPECIFIC_ENTITY_GROUPS:
            target_label = ENTITY_GROUP_TO_LABEL[group]
            meta_names: set[str] = set()
            for raw_name in _normalize_string_list(payload.get(group)):
                resolved_card = normalizer.get_entity_card(group, raw_name)
                if resolved_card is not None:
                    meta_names.add(resolved_card.canonical_name)
            lex_names = set(lexical_entities.get(group, ()))
            for canonical_name in sorted(meta_names | lex_names):
                in_meta = canonical_name in meta_names
                weight = 1.25 if in_meta else 1.0
                if group in {"drug_product", "active_ingredient"}:
                    weight += 0.15
                mentions_entity_edges.append(
                    {
                        "point_id": point_id,
                        "target_label": target_label,
                        "canonical_name": canonical_name,
                        "entity_group": group,
                        "match_source": "metadata" if in_meta else "lexical",
                        "weight": round(weight, 3),
                        "kb_version": clean_build_id,
                    }
                )

        # 2. Liên kết HAS_CLINICAL_FACET từ cả trường metadata lâm sàng lẫn thuật ngữ chuyên môn
        meta_matched_facets: set[str] = set()
        for spec in CLINICAL_FACET_CATALOG:
            for field_name, expected_val in spec.metadata_matches:
                actual_values = {val.casefold() for val in _normalize_string_list(payload.get(field_name))}
                if expected_val.casefold() in actual_values:
                    meta_matched_facets.add(spec.facet_key)
                    break

        lexical_facets_by_group = extract_clinical_facets_from_text(
            searchable_chunk_text,
            allow_ascii_fallback=False,
        )
        lex_matched_facets = {
            facet_key
            for keys in lexical_facets_by_group.values()
            for facet_key in keys
        }

        for facet_key in sorted(meta_matched_facets | lex_matched_facets):
            in_meta = facet_key in meta_matched_facets
            spec = FACET_BY_KEY[facet_key]
            has_facet_edges.append(
                {
                    "point_id": point_id,
                    "facet_key": facet_key,
                    "facet_group": spec.facet_group,
                    "match_source": "metadata" if in_meta else "lexical",
                    "weight": 1.3 if in_meta else 1.05,
                    "kb_version": clean_build_id,
                }
            )

    # 3. Liên kết NEXT_CHUNK giữa các đoạn liền kề trong cùng một tài liệu
    next_chunk_edges: list[dict[str, Any]] = []
    for document_id, doc_chunks in sorted(chunks_by_doc.items()):
        ordered_doc_chunks = sorted(
            doc_chunks,
            key=lambda item: (int(item["chunk_index"]), str(item["point_id"])),
        )
        for idx in range(len(ordered_doc_chunks) - 1):
            curr_chunk = ordered_doc_chunks[idx]
            nxt_chunk = ordered_doc_chunks[idx + 1]
            same_section = bool(
                curr_chunk["section_title"]
                and curr_chunk["section_title"] == nxt_chunk["section_title"]
            )
            same_table = (
                curr_chunk["content_kind"] == "table"
                and nxt_chunk["content_kind"] == "table"
                and curr_chunk["table_index"] >= 0
                and curr_chunk["table_index"] == nxt_chunk["table_index"]
            )
            next_chunk_edges.append(
                {
                    "from_point_id": curr_chunk["point_id"],
                    "to_point_id": nxt_chunk["point_id"],
                    "document_id": document_id,
                    "same_section": same_section,
                    "same_table": same_table,
                    "kb_version": clean_build_id,
                }
            )

    facet_nodes = [
        {
            "facet_key": spec.facet_key,
            "facet_group": spec.facet_group,
            "display_name": spec.display_name,
            "aliases": list(spec.aliases),
            "kb_version": clean_build_id,
            "schema_version": GRAPH_CONTRACT_VERSION,
        }
        for spec in CLINICAL_FACET_CATALOG
    ]

    published_edges = [
        {
            "source_id": source_id,
            "document_id": document_id,
            "kb_version": clean_build_id,
        }
        for source_id, document_id in sorted(published_pairs)
    ]

    interaction_edges: list[dict[str, Any]] = []
    for item in CLINICAL_INTERACTION_CATALOG:
        src_group = label_to_group.get(item["source_label"], "")
        tgt_group = label_to_group.get(item["target_label"], "")
        src_card = normalizer.get_entity_card(src_group, item["source_name"])
        tgt_card = normalizer.get_entity_card(tgt_group, item["target_name"])
        if src_card is None or tgt_card is None:
            continue
        interaction_edges.append(
            {
                **item,
                "source_name": src_card.canonical_name,
                "target_name": tgt_card.canonical_name,
                "kb_version": clean_build_id,
                "schema_version": GRAPH_CONTRACT_VERSION,
            }
        )

    return {
        "sources": [sources_by_id[key] for key in sorted(sources_by_id)],
        "documents": [documents_by_id[key] for key in sorted(documents_by_id)],
        "chunks": chunks,
        "facets": facet_nodes,
        "published_edges": published_edges,
        "has_chunk_edges": has_chunk_edges,
        "next_chunk_edges": next_chunk_edges,
        "mentions_entity_edges": mentions_entity_edges,
        "has_facet_edges": has_facet_edges,
        "interaction_edges": interaction_edges,
    }


def _iter_batches(items: list[dict[str, Any]], batch_size: int = 250) -> Iterable[list[dict[str, Any]]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


async def ensure_lexical_graph_constraints(driver: Any) -> list[str]:
    """Khởi tạo ràng buộc duy nhất và chỉ mục tra cứu nhanh trên Neo4j."""

    statements = [
        (
            "CREATE CONSTRAINT lexical_source_id_unique IF NOT EXISTS "
            "FOR (s:KnowledgeSource) REQUIRE s.source_id IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT lexical_document_id_unique IF NOT EXISTS "
            "FOR (d:KnowledgeDocument) REQUIRE d.document_id IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT lexical_chunk_point_id_unique IF NOT EXISTS "
            "FOR (c:KnowledgeChunk) REQUIRE c.point_id IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT clinical_facet_key_unique IF NOT EXISTS "
            "FOR (f:ClinicalFacet) REQUIRE f.facet_key IS UNIQUE"
        ),
        (
            "CREATE INDEX lexical_chunk_kb_version_idx IF NOT EXISTS "
            "FOR (c:KnowledgeChunk) ON (c.kb_version)"
        ),
        (
            "CREATE INDEX lexical_chunk_doc_idx IF NOT EXISTS "
            "FOR (c:KnowledgeChunk) ON (c.document_id, c.chunk_index)"
        ),
    ]
    async with driver.session() as session:
        for statement in statements:
            result = await session.run(statement)
            await result.consume()
    return statements


async def sync_lexical_graph_to_neo4j(
    driver: Any,
    lexical_records: Mapping[str, list[dict[str, Any]]],
    *,
    build_id: str,
) -> dict[str, Any]:
    """Đồng bộ hàng loạt (UNWIND batch) toàn bộ Lexical + Clinical Graph vào Neo4j.

    Không xóa hay thay đổi bất kỳ nút/cạnh nào thuộc ``ENTITY_GRAPH_LABELS`` và
    ``ENTITY_GRAPH_RELATIONSHIPS``. Dọn sạch các bản ghi lexical cũ có ``kb_version``
    lệch với ``build_id`` hiện hành.
    """

    clean_build_id = str(build_id or "").strip()
    if not clean_build_id:
        raise ValueError("build_id is required when syncing lexical graph to Neo4j")

    await ensure_lexical_graph_constraints(driver)

    sources = list(lexical_records.get("sources", []))
    documents = list(lexical_records.get("documents", []))
    chunks = list(lexical_records.get("chunks", []))
    facets = list(lexical_records.get("facets", []))
    published_edges = list(lexical_records.get("published_edges", []))
    has_chunk_edges = list(lexical_records.get("has_chunk_edges", []))
    next_chunk_edges = list(lexical_records.get("next_chunk_edges", []))
    mentions_entity_edges = list(lexical_records.get("mentions_entity_edges", []))
    has_facet_edges = list(lexical_records.get("has_facet_edges", []))
    interaction_edges = list(lexical_records.get("interaction_edges", []))

    async with driver.session() as session:
        # 1. Merge các nút thuộc tầng Lexical & Facet
        for batch in _iter_batches(sources):
            res = await session.run(
                "UNWIND $rows AS row "
                "MERGE (s:KnowledgeSource {source_id: row.source_id}) "
                "SET s.title = row.title, "
                "    s.source_authority = row.source_authority, "
                "    s.source_type = row.source_type, "
                "    s.source_url = row.source_url, "
                "    s.scope = row.scope, "
                "    s.category = row.category, "
                "    s.kb_version = row.kb_version, "
                "    s.schema_version = row.schema_version",
                rows=batch,
            )
            await res.consume()

        for batch in _iter_batches(documents):
            res = await session.run(
                "UNWIND $rows AS row "
                "MERGE (d:KnowledgeDocument {document_id: row.document_id}) "
                "SET d.source_id = row.source_id, "
                "    d.title = row.title, "
                "    d.source_title = row.source_title, "
                "    d.kb_version = row.kb_version, "
                "    d.schema_version = row.schema_version",
                rows=batch,
            )
            await res.consume()

        for batch in _iter_batches(chunks, batch_size=150):
            res = await session.run(
                "UNWIND $rows AS row "
                "MERGE (c:KnowledgeChunk {point_id: row.point_id}) "
                "SET c.chunk_id = row.chunk_id, "
                "    c.document_id = row.document_id, "
                "    c.source_id = row.source_id, "
                "    c.source_title = row.source_title, "
                "    c.document_title = row.document_title, "
                "    c.section_title = row.section_title, "
                "    c.content_kind = row.content_kind, "
                "    c.chunk_index = row.chunk_index, "
                "    c.table_index = row.table_index, "
                "    c.text = row.text, "
                "    c.payload_json = row.payload_json, "
                "    c.kb_version = row.kb_version, "
                "    c.schema_version = row.schema_version",
                rows=batch,
            )
            await res.consume()

        for batch in _iter_batches(facets):
            res = await session.run(
                "UNWIND $rows AS row "
                "MERGE (f:ClinicalFacet {facet_key: row.facet_key}) "
                "SET f.facet_group = row.facet_group, "
                "    f.display_name = row.display_name, "
                "    f.aliases = row.aliases, "
                "    f.kb_version = row.kb_version, "
                "    f.schema_version = row.schema_version",
                rows=batch,
            )
            await res.consume()

        # Xóa trước các quan hệ lexical cũ để đảm bảo idempotent tuyệt đối khi chạy lại cùng build_id
        for rel_type in LEXICAL_GRAPH_RELATIONSHIPS:
            clean_res = await session.run(f"MATCH ()-[r:{rel_type}]->() DELETE r")
            await clean_res.consume()

        # 2. Merge các quan hệ cấu trúc và ngữ nghĩa lâm sàng
        for batch in _iter_batches(published_edges):
            res = await session.run(
                "UNWIND $rows AS row "
                "MATCH (s:KnowledgeSource {source_id: row.source_id}) "
                "MATCH (d:KnowledgeDocument {document_id: row.document_id}) "
                "MERGE (s)-[r:PUBLISHED_DOCUMENT]->(d) "
                "SET r.kb_version = row.kb_version",
                rows=batch,
            )
            await res.consume()

        for batch in _iter_batches(has_chunk_edges):
            res = await session.run(
                "UNWIND $rows AS row "
                "MATCH (d:KnowledgeDocument {document_id: row.document_id}) "
                "MATCH (c:KnowledgeChunk {point_id: row.point_id}) "
                "MERGE (d)-[r:HAS_CHUNK]->(c) "
                "SET r.chunk_index = row.chunk_index, r.kb_version = row.kb_version",
                rows=batch,
            )
            await res.consume()

        for batch in _iter_batches(next_chunk_edges):
            res = await session.run(
                "UNWIND $rows AS row "
                "MATCH (c1:KnowledgeChunk {point_id: row.from_point_id}) "
                "MATCH (c2:KnowledgeChunk {point_id: row.to_point_id}) "
                "MERGE (c1)-[r:NEXT_CHUNK]->(c2) "
                "SET r.document_id = row.document_id, "
                "    r.same_section = row.same_section, "
                "    r.same_table = row.same_table, "
                "    r.kb_version = row.kb_version",
                rows=batch,
            )
            await res.consume()

        # Nhóm MENTIONS_ENTITY theo nhãn đích an toàn
        edges_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in mentions_entity_edges:
            label = str(edge["target_label"])
            if label in ENTITY_GROUP_TO_LABEL.values():
                edges_by_label[label].append(edge)

        for label, label_edges in sorted(edges_by_label.items()):
            for batch in _iter_batches(label_edges):
                res = await session.run(
                    "UNWIND $rows AS row "
                    "MATCH (c:KnowledgeChunk {point_id: row.point_id}) "
                    f"MATCH (e:{label} {{canonical_name: row.canonical_name}}) "
                    "MERGE (c)-[r:MENTIONS_ENTITY]->(e) "
                    "SET r.entity_group = row.entity_group, "
                    "    r.match_source = row.match_source, "
                    "    r.weight = row.weight, "
                    "    r.kb_version = row.kb_version",
                    rows=batch,
                )
                await res.consume()

        for batch in _iter_batches(has_facet_edges):
            res = await session.run(
                "UNWIND $rows AS row "
                "MATCH (c:KnowledgeChunk {point_id: row.point_id}) "
                "MATCH (f:ClinicalFacet {facet_key: row.facet_key}) "
                "MERGE (c)-[r:HAS_CLINICAL_FACET]->(f) "
                "SET r.facet_group = row.facet_group, "
                "    r.match_source = row.match_source, "
                "    r.weight = row.weight, "
                "    r.kb_version = row.kb_version",
                rows=batch,
            )
            await res.consume()

        for edge in interaction_edges:
            src_label = edge["source_label"]
            tgt_label = edge["target_label"]
            if src_label not in ENTITY_GROUP_TO_LABEL.values() or tgt_label not in ENTITY_GROUP_TO_LABEL.values():
                continue
            res = await session.run(
                f"MATCH (src:{src_label} {{canonical_name: $source_name}}) "
                f"MATCH (tgt:{tgt_label} {{canonical_name: $target_name}}) "
                "MERGE (src)-[r:CLINICAL_INTERACTION]->(tgt) "
                "SET r.interaction_type = $interaction_type, "
                "    r.clinical_note = $clinical_note, "
                "    r.kb_version = $kb_version, "
                "    r.schema_version = $schema_version",
                source_name=edge["source_name"],
                target_name=edge["target_name"],
                interaction_type=edge["interaction_type"],
                clinical_note=edge["clinical_note"],
                kb_version=edge["kb_version"],
                schema_version=edge["schema_version"],
            )
            await res.consume()

        # 3. Thu dọn các nút lexical thuộc build cũ (không chạm tới ENTITY_GRAPH_LABELS)
        for label in LEXICAL_GRAPH_LABELS:
            del_res = await session.run(
                f"MATCH (n:{label}) "
                "WHERE coalesce(n.kb_version, '') <> $build_id "
                "DETACH DELETE n",
                build_id=clean_build_id,
            )
            await del_res.consume()

    validation = await validate_lexical_graph_in_neo4j(
        driver,
        lexical_records,
        build_id=clean_build_id,
    )
    if not validation["passed"]:
        raise RuntimeError(f"Lexical GraphRAG reconciliation failed: {validation['errors']}")

    return {
        "build_id": clean_build_id,
        "counts": {
            "sources": len(sources),
            "documents": len(documents),
            "chunks": len(chunks),
            "facets": len(facets),
            "published_edges": len(published_edges),
            "has_chunk_edges": len(has_chunk_edges),
            "next_chunk_edges": len(next_chunk_edges),
            "mentions_entity_edges": len(mentions_entity_edges),
            "has_facet_edges": len(has_facet_edges),
            "interaction_edges": len(interaction_edges),
        },
        "validation": validation,
    }


async def validate_lexical_graph_in_neo4j(
    driver: Any,
    lexical_records: Mapping[str, list[dict[str, Any]]],
    *,
    build_id: str,
) -> dict[str, Any]:
    """Đối chiếu số lượng nút và quan hệ của Lexical + Clinical Graph trong Neo4j."""

    expected_nodes = {
        "KnowledgeSource": len(lexical_records.get("sources", [])),
        "KnowledgeDocument": len(lexical_records.get("documents", [])),
        "KnowledgeChunk": len(lexical_records.get("chunks", [])),
        "ClinicalFacet": len(lexical_records.get("facets", [])),
    }
    expected_rels = {
        "PUBLISHED_DOCUMENT": len(lexical_records.get("published_edges", [])),
        "HAS_CHUNK": len(lexical_records.get("has_chunk_edges", [])),
        "NEXT_CHUNK": len(lexical_records.get("next_chunk_edges", [])),
        "MENTIONS_ENTITY": len(lexical_records.get("mentions_entity_edges", [])),
        "HAS_CLINICAL_FACET": len(lexical_records.get("has_facet_edges", [])),
        "CLINICAL_INTERACTION": len(lexical_records.get("interaction_edges", [])),
    }

    actual_nodes: dict[str, int] = {}
    actual_rels: dict[str, int] = {}
    errors: list[str] = []

    async with driver.session() as session:
        for label, exp_count in expected_nodes.items():
            res = await session.run(
                f"MATCH (n:{label} {{kb_version: $build_id}}) RETURN count(n) AS count",
                build_id=build_id,
            )
            record = await res.single()
            count = int(record["count"]) if record else 0
            actual_nodes[label] = count
            if count != exp_count:
                errors.append(f"lexical node mismatch for {label}: expected {exp_count}, got {count}")

        for rel_type, exp_count in expected_rels.items():
            res = await session.run(
                f"MATCH ()-[r:{rel_type} {{kb_version: $build_id}}]->() RETURN count(r) AS count",
                build_id=build_id,
            )
            record = await res.single()
            count = int(record["count"]) if record else 0
            actual_rels[rel_type] = count
            if count != exp_count:
                errors.append(
                    f"lexical relationship mismatch for {rel_type}: expected {exp_count}, got {count}"
                )

    rel_counter = Counter()
    for rel_type, count in actual_rels.items():
        rel_counter[rel_type] = count

    return {
        "layer": "neo4j_lexical_graph",
        "passed": not errors,
        "build_id": build_id,
        "expected_nodes": expected_nodes,
        "actual_nodes": actual_nodes,
        "expected_relationships": expected_rels,
        "actual_relationships": dict(rel_counter),
        "errors": errors,
    }


__all__ = [
    "CLINICAL_FACET_CATALOG",
    "CLINICAL_INTERACTION_CATALOG",
    "ClinicalFacetSpec",
    "FACET_BY_KEY",
    "LEXICAL_GRAPH_LABELS",
    "LEXICAL_GRAPH_RELATIONSHIPS",
    "QueryGraphAnchors",
    "build_lexical_graph_records",
    "ensure_lexical_graph_constraints",
    "extract_clinical_facets_from_text",
    "extract_query_graph_anchors",
    "fold_ascii_text",
    "sync_lexical_graph_to_neo4j",
    "validate_lexical_graph_in_neo4j",
]
