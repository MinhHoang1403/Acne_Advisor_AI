"""
dermatology_taxonomy.py
=======================
Dermatology domain vocabulary for acne-agent-system.

Defines canonical categories, keyword dictionaries, and Vietnamese → English
mappings used by the rule-based metadata extractor in ``domain_metadata.py``.

All canonical values are lowercase snake_case strings.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Domain Topics
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_TOPICS: list[str] = [
    "acne_causes",
    "acne_symptoms",
    "acne_severity",
    "acne_treatment",
    "ingredient_mechanism",
    "side_effect",
    "contraindication",
    "routine_advice",
    "when_to_see_doctor",
    "pregnancy_safety",
    "pediatric_safety",
]

# ─────────────────────────────────────────────────────────────────────────────
# Content Types
# ─────────────────────────────────────────────────────────────────────────────

CONTENT_TYPES: list[str] = [
    "definition",
    "cause",
    "symptom",
    "diagnosis",
    "severity",
    "treatment",
    "mechanism",
    "side_effect",
    "warning",
    "contraindication",
    "routine",
    "doctor_visit",
]

# ─────────────────────────────────────────────────────────────────────────────
# Ingredients
# ─────────────────────────────────────────────────────────────────────────────

INGREDIENTS: list[str] = [
    "benzoyl_peroxide",
    "retinoid",
    "retinol",
    "adapalene",
    "tretinoin",
    "isotretinoin",
    "salicylic_acid",
    "azelaic_acid",
    "clindamycin",
    "erythromycin",
    "niacinamide",
]

# ─────────────────────────────────────────────────────────────────────────────
# Skin Types
# ─────────────────────────────────────────────────────────────────────────────

SKIN_TYPES: list[str] = [
    "oily",
    "dry",
    "sensitive",
    "combination",
    "normal",
]

# ─────────────────────────────────────────────────────────────────────────────
# Concerns
# ─────────────────────────────────────────────────────────────────────────────

CONCERNS: list[str] = [
    "acne",
    "inflammatory_acne",
    "blackheads",
    "whiteheads",
    "comedonal_acne",
    "pustules",
    "nodules",
    "acne_scars",
    "post_inflammatory_hyperpigmentation",
]

# ─────────────────────────────────────────────────────────────────────────────
# Body Areas
# ─────────────────────────────────────────────────────────────────────────────

BODY_AREAS: list[str] = [
    "face",
    "cheek",
    "chin",
    "forehead",
    "nose",
    "back",
    "chest",
]

# ─────────────────────────────────────────────────────────────────────────────
# Safety Contexts
# ─────────────────────────────────────────────────────────────────────────────

SAFETY_CONTEXTS: list[str] = [
    "irritation",
    "dryness",
    "peeling",
    "burning",
    "allergy",
    "pregnancy_safety",
    "breastfeeding_safety",
    "pediatric_safety",
    "photosensitivity",
    "contraindication",
]

# ─────────────────────────────────────────────────────────────────────────────
# Keyword dictionaries for rule-based matching
# ─────────────────────────────────────────────────────────────────────────────
# Each dict maps a canonical value → list of keyword patterns (lowercased).
# Both English and Vietnamese keywords are included.
#
# The extractor converts the input text to lowercase and checks for substring
# matches, so these should be reasonably specific to avoid false positives.

DOMAIN_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "acne_causes": [
        "nguyên nhân", "nguyên nhân mụn", "gây mụn", "gây ra mụn",
        "cause of acne", "causes of acne", "acne cause",
        "nội tiết tố", "hormone", "hormonal", "androgen",
        "tắc nghẽn lỗ chân lông", "clogged pore", "excess sebum",
        "bã nhờn",
    ],
    "acne_symptoms": [
        "triệu chứng", "biểu hiện", "dấu hiệu",
        "symptom", "sign", "manifestation",
    ],
    "acne_severity": [
        "mức độ", "phân độ", "độ nặng", "cấp độ",
        "severity", "grade", "mild", "moderate", "severe",
        "nhẹ", "trung bình", "nặng",
    ],
    "acne_treatment": [
        "điều trị", "trị mụn", "chữa mụn", "thuốc trị",
        "treatment", "therapy", "therapeutic",
        "bôi", "uống", "topical", "oral",
    ],
    "ingredient_mechanism": [
        "cơ chế", "tác dụng", "hoạt động",
        "mechanism", "how it works", "mode of action",
        "ức chế", "inhibit", "reduce", "giảm",
    ],
    "side_effect": [
        "tác dụng phụ", "phản ứng phụ", "tác dụng không mong muốn",
        "side effect", "adverse effect", "adverse reaction",
    ],
    "contraindication": [
        "chống chỉ định", "không nên dùng", "không được dùng",
        "contraindication", "contraindicated", "do not use",
    ],
    "routine_advice": [
        "quy trình", "routine", "skincare routine",
        "bước chăm sóc", "chăm sóc da", "hướng dẫn",
        "cách dùng", "cách sử dụng",
    ],
    "when_to_see_doctor": [
        "gặp bác sĩ", "đi khám", "khám bác sĩ", "đến bác sĩ",
        "see a doctor", "see doctor", "consult a dermatologist",
        "bác sĩ da liễu",
    ],
    "pregnancy_safety": [
        "mang thai", "thai kỳ", "bầu",
        "pregnancy", "pregnant", "prenatal",
    ],
    "pediatric_safety": [
        "trẻ em", "trẻ nhỏ", "con tôi", "bé nhà", "thiếu niên",
        "pediatric", "children", "child", "adolescent", "teen",
    ],
}

CONTENT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "definition": [
        "là gì", "định nghĩa", "khái niệm",
        "what is", "definition", "defined as",
    ],
    "cause": [
        "nguyên nhân", "do", "gây ra", "gây nên",
        "cause", "caused by", "etiology", "aetiology",
    ],
    "symptom": [
        "triệu chứng", "biểu hiện", "dấu hiệu",
        "symptom", "sign", "present with",
    ],
    "diagnosis": [
        "chẩn đoán", "xét nghiệm",
        "diagnosis", "diagnose", "diagnostic",
    ],
    "severity": [
        "mức độ", "phân loại mức", "cấp độ",
        "severity", "grading", "classification",
    ],
    "treatment": [
        "điều trị", "trị", "chữa", "thuốc",
        "treatment", "therapy", "manage", "therapeutic",
    ],
    "mechanism": [
        "cơ chế", "cách hoạt động", "tác dụng",
        "mechanism", "how it works", "mode of action", "pharmacology",
    ],
    "side_effect": [
        "tác dụng phụ", "phản ứng phụ", "tác dụng không mong muốn",
        "gây khô", "gây kích ứng", "gây bong tróc",
        "side effect", "adverse", "unwanted effect",
    ],
    "warning": [
        "cảnh báo", "lưu ý", "thận trọng", "cần chú ý",
        "warning", "caution", "precaution", "be careful",
    ],
    "contraindication": [
        "chống chỉ định", "không nên", "tránh dùng",
        "contraindication", "contraindicated", "avoid",
    ],
    "routine": [
        "quy trình", "routine", "skincare",
        "bước chăm sóc", "hướng dẫn sử dụng",
    ],
    "doctor_visit": [
        "gặp bác sĩ", "khám bác sĩ", "đi khám",
        "see a doctor", "consult", "dermatologist",
    ],
}

INGREDIENT_KEYWORDS: dict[str, list[str]] = {
    "benzoyl_peroxide": [
        "benzoyl peroxide", "benzoyl_peroxide", "bpo",
    ],
    "retinoid": [
        "retinoid",
    ],
    "retinol": [
        "retinol",
    ],
    "adapalene": [
        "adapalene", "differin",
    ],
    "tretinoin": [
        "tretinoin", "retin-a", "retin a",
    ],
    "isotretinoin": [
        "isotretinoin", "accutane", "roaccutane",
    ],
    "salicylic_acid": [
        "salicylic acid", "salicylic_acid", "bha",
        "axit salicylic", "acid salicylic",
    ],
    "azelaic_acid": [
        "azelaic acid", "azelaic_acid",
        "axit azelaic", "acid azelaic",
    ],
    "clindamycin": [
        "clindamycin",
    ],
    "erythromycin": [
        "erythromycin",
    ],
    "niacinamide": [
        "niacinamide", "vitamin b3", "nicotinamide",
    ],
}

SKIN_TYPE_KEYWORDS: dict[str, list[str]] = {
    "oily": [
        "da dầu", "da nhờn",
        "oily skin", "oily",
    ],
    "dry": [
        "da khô ",  # trailing space to avoid matching "da không"
        "da khô,", "da khô.", "da khô?", "da khô!",  # punctuation variants
        "dry skin",
    ],
    "sensitive": [
        "da nhạy cảm", "da mẫn cảm",
        "sensitive skin", "sensitive",
    ],
    "combination": [
        "da hỗn hợp",
        "combination skin", "combination",
    ],
    "normal": [
        "da thường", "da bình thường",
        "normal skin",
    ],
}

CONCERN_KEYWORDS: dict[str, list[str]] = {
    "acne": [
        "mụn trứng cá", "mụn", "acne",
    ],
    "inflammatory_acne": [
        "mụn viêm", "mụn sưng", "mụn đỏ",
        "inflammatory acne", "inflamed acne",
    ],
    "blackheads": [
        "mụn đầu đen", "đầu đen",
        "blackhead", "blackheads", "open comedone",
    ],
    "whiteheads": [
        "mụn đầu trắng", "đầu trắng", "mụn ẩn",
        "whitehead", "whiteheads", "closed comedone",
    ],
    "comedonal_acne": [
        "mụn không viêm", "mụn comedone",
        "comedonal acne", "comedone",
    ],
    "pustules": [
        "mụn mủ", "mụn có mủ",
        "pustule", "pustules",
    ],
    "nodules": [
        "mụn cục", "mụn nang", "mụn bọc",
        "nodule", "nodules", "nodular acne", "cystic acne",
    ],
    "acne_scars": [
        "sẹo mụn", "sẹo rỗ", "sẹo lồi",
        "acne scar", "acne scars", "scarring",
    ],
    "post_inflammatory_hyperpigmentation": [
        "thâm mụn", "thâm sau mụn", "vết thâm",
        "post inflammatory hyperpigmentation", "post-inflammatory hyperpigmentation",
        "pih", "dark spot", "dark spots",
    ],
}

BODY_AREA_KEYWORDS: dict[str, list[str]] = {
    "face": [
        "mặt", "khuôn mặt", "vùng mặt",
        "face", "facial",
    ],
    "cheek": [
        "má", "hai má", "vùng má",
        "cheek", "cheeks",
    ],
    "chin": [
        "cằm", "vùng cằm",
        "chin",
    ],
    "forehead": [
        "trán", "vùng trán",
        "forehead",
    ],
    "nose": [
        "mũi", "vùng mũi",
        "nose", "nasal",
    ],
    "back": [
        "lưng", "vùng lưng",
        "back",
    ],
    "chest": [
        "ngực", "vùng ngực",
        "chest",
    ],
}

SAFETY_CONTEXT_KEYWORDS: dict[str, list[str]] = {
    "irritation": [
        "kích ứng", "kích ứng da",
        "irritation", "irritate", "irritating",
    ],
    "dryness": [
        "khô da", "gây khô", "bị khô",
        "dryness", "skin dryness", "drying", "xerosis",
    ],
    "peeling": [
        "bong tróc", "tróc da", "bong da",
        "peeling", "flaking", "desquamation",
    ],
    "burning": [
        "nóng rát", "châm chích", "rát", "bỏng rát",
        "burning", "stinging",
    ],
    "allergy": [
        "dị ứng", "phản ứng dị ứng",
        "allergy", "allergic", "allergic reaction",
    ],
    "pregnancy_safety": [
        "mang thai", "thai kỳ", "phụ nữ mang thai", "bầu",
        "pregnancy", "pregnant",
    ],
    "breastfeeding_safety": [
        "cho con bú", "đang cho bú", "nuôi con bằng sữa mẹ",
        "breastfeeding", "lactation", "nursing",
    ],
    "pediatric_safety": [
        "trẻ em", "trẻ nhỏ", "con tôi", "bé nhà", "thiếu niên",
        "pediatric", "children", "child",
    ],
    "photosensitivity": [
        "nhạy cảm ánh sáng", "sợ nắng", "tránh nắng",
        "photosensitivity", "photosensitive", "sun sensitivity",
    ],
    "contraindication": [
        "chống chỉ định", "không nên dùng", "không được dùng",
        "contraindication", "contraindicated",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Consolidated Vietnamese → English mapping
# ─────────────────────────────────────────────────────────────────────────────
# This flat dict is provided for convenience / documentation.
# It maps a Vietnamese keyword to its canonical (category, value) pair.

VIETNAMESE_MAPPINGS: dict[str, tuple[str, str]] = {
    # Skin types
    "da dầu": ("skin_type", "oily"),
    "da khô": ("skin_type", "dry"),
    "da nhạy cảm": ("skin_type", "sensitive"),
    # Concerns
    "mụn viêm": ("concern", "inflammatory_acne"),
    "mụn đầu đen": ("concern", "blackheads"),
    "mụn đầu trắng": ("concern", "whiteheads"),
    "mụn mủ": ("concern", "pustules"),
    "mụn cục": ("concern", "nodules"),
    "mụn nang": ("concern", "nodules"),
    "thâm mụn": ("concern", "post_inflammatory_hyperpigmentation"),
    "sẹo mụn": ("concern", "acne_scars"),
    # Safety contexts
    "kích ứng": ("safety_context", "irritation"),
    "khô da": ("safety_context", "dryness"),
    "bong tróc": ("safety_context", "peeling"),
    "nóng rát": ("safety_context", "burning"),
    "châm chích": ("safety_context", "burning"),
    "dị ứng": ("safety_context", "allergy"),
    "mang thai": ("safety_context", "pregnancy_safety"),
    "cho con bú": ("safety_context", "breastfeeding_safety"),
    "trẻ em": ("safety_context", "pediatric_safety"),
    "con tôi": ("safety_context", "pediatric_safety"),
    "bé nhà": ("safety_context", "pediatric_safety"),
    # Body areas
    "má": ("body_area", "cheek"),
    "cằm": ("body_area", "chin"),
    "trán": ("body_area", "forehead"),
    "mũi": ("body_area", "nose"),
    "lưng": ("body_area", "back"),
    "ngực": ("body_area", "chest"),
}
