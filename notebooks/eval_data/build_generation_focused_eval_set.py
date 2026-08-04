"""Build the generation-focused 300-question Acne Advisor evaluation set.

This dataset intentionally emphasizes answerable, in-domain acne questions.
It complements rather than replaces the safety/readiness evaluation set.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


OUTPUT_PATH = Path(__file__).with_name("acne_rag_eval_generation_focused.jsonl")

CATEGORY_TARGETS = {
    "core_knowledge_generation": 50,
    "active_ingredients_generation": 45,
    "product_entity_generation": 40,
    "comparison_generation": 40,
    "treatment_plan_reference": 45,
    "routine_skincare_generation": 35,
    "multi_turn_like_generation": 25,
    "exact_format_light": 15,
    "mild_safety_caution": 5,
}


def case(
    *,
    category: str,
    question: str,
    expected_keywords: list[str],
    forbidden_keywords: list[str] | None = None,
    requires_sources: bool = True,
    requires_table: bool = False,
    expected_format: str = "short_answer",
    expected_count: int | None = None,
    requires_bullets: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "category": category,
        "question": question,
        "expected_keywords": expected_keywords,
        "forbidden_keywords": forbidden_keywords or [],
        "requires_sources": requires_sources,
        "requires_table": requires_table,
        "expected_format": expected_format,
        "notes": "generation-focused in-domain evaluation",
    }
    if expected_count is not None:
        payload["expected_count"] = expected_count
    if requires_bullets:
        payload["requires_bullets"] = True
    return payload


def add_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters: Counter[str] = Counter()
    for row in rows:
        counters[row["category"]] += 1
        row["id"] = f"{row['category']}_{counters[row['category']]:03d}"
    return rows


def build_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    core_topics = [
        ("mụn đầu trắng", "mụn đầu trắng"),
        ("mụn đầu đen", "mụn đầu đen"),
        ("mụn viêm", "mụn viêm"),
        ("mụn mủ", "mụn mủ"),
        ("mụn nang", "mụn nang"),
        ("sẹo mụn", "sẹo"),
        ("thâm sau mụn", "thâm"),
        ("da dầu dễ nổi mụn", "da dầu"),
        ("mụn ở lưng", "mụn lưng"),
        ("mụn tái phát", "mụn tái phát"),
    ]
    core_templates = [
        "Giải thích ngắn gọn {topic} là gì và đặc điểm thường gặp.",
        "Khi nói về {topic}, dấu hiệu nào giúp nhận biết trong chăm sóc mụn?",
        "Tóm tắt kiến thức cơ bản về {topic} cho người mới tìm hiểu.",
        "{topic_cap} khác gì với tình trạng da bình thường ở điểm nào?",
        "Trong chăm sóc da mụn, nên lưu ý điều gì khi gặp {topic}?",
    ]
    for topic, keyword in core_topics:
        for template in core_templates:
            rows.append(
                case(
                    category="core_knowledge_generation",
                    question=template.format(topic=topic, topic_cap=topic.capitalize()),
                    expected_keywords=[keyword],
                    forbidden_keywords=["không liên quan đến mụn"],
                    expected_format="explanation",
                )
            )

    ingredients = [
        ("benzoyl peroxide", "kháng khuẩn"),
        ("adapalene", "retinoid"),
        ("salicylic acid", "bít tắc"),
        ("azelaic acid", "mụn"),
        ("clindamycin", "kháng sinh"),
        ("erythromycin", "kháng sinh"),
        ("tretinoin", "retinoid"),
        ("dapsone", "mụn viêm"),
        ("sulfur", "mụn"),
    ]
    ingredient_templates = [
        "{ingredient_cap} hỗ trợ chăm sóc hoặc điều trị mụn bằng cách nào?",
        "Vai trò chính của {ingredient} trong mụn là gì?",
        "Khi mới bắt đầu dùng {ingredient}, nên lưu ý điều gì để da dễ thích nghi?",
        "{ingredient_cap} thường được nhắc đến trong loại mụn hoặc mục tiêu chăm sóc nào?",
        "Hãy giải thích ngắn gọn vì sao {ingredient} có thể xuất hiện trong routine trị mụn.",
    ]
    for ingredient, mechanism in ingredients:
        for template in ingredient_templates:
            rows.append(
                case(
                    category="active_ingredients_generation",
                    question=template.format(ingredient=ingredient, ingredient_cap=ingredient.capitalize()),
                    expected_keywords=[ingredient, mechanism],
                    forbidden_keywords=[f"{ingredient} không liên quan đến mụn"],
                    expected_format="explanation",
                )
            )

    products = [
        ("Differin", "adapalene", "retinoid"),
        ("Epiduo", "adapalene", "benzoyl peroxide"),
        ("Dalacin T", "clindamycin", "kháng sinh bôi"),
        ("Tazorac", "tazarotene", "retinoid"),
    ]
    product_templates = [
        "{product} liên quan đến hoạt chất nào trong chăm sóc mụn?",
        "{product} thuộc nhóm sản phẩm hoặc điều trị nào?",
        "Tóm tắt vai trò tham khảo của {product} trong điều trị mụn.",
        "Nếu thấy tên {product}, nên liên hệ sản phẩm này với hoạt chất nào?",
        "{product} có điểm nào cần biết trước khi đưa vào routine trị mụn?",
        "Giải thích ngắn gọn {product} thường hướng tới mục tiêu chăm sóc mụn nào.",
        "{product} khác kem dưỡng ẩm thông thường ở vai trò nào trong routine?",
        "Hãy nêu thành phần hoặc nhóm chính khi nhắc đến {product}.",
        "Trong thông tin về mụn, {product} thường được dùng để minh họa cho nhóm nào?",
        "Người mới tìm hiểu mụn cần nhớ gì về {product}?",
    ]
    for product, active, group in products:
        for template in product_templates:
            rows.append(
                case(
                    category="product_entity_generation",
                    question=template.format(product=product),
                    expected_keywords=[product.lower(), active],
                    forbidden_keywords=[f"{product.lower()} không có liên quan đến mụn"],
                    expected_format="direct_answer",
                )
            )

    comparison_pairs = [
        ("benzoyl peroxide", "adapalene"),
        ("benzoyl peroxide", "salicylic acid"),
        ("adapalene", "azelaic acid"),
        ("Differin", "Epiduo"),
        ("Dalacin T", "Differin"),
        ("mụn đầu trắng", "mụn đầu đen"),
        ("retinoid bôi", "benzoyl peroxide"),
        ("clindamycin", "benzoyl peroxide"),
        ("salicylic acid", "azelaic acid"),
        ("routine buổi sáng", "routine buổi tối"),
    ]
    comparison_templates = [
        ("{left} và {right} khác nhau thế nào trong chăm sóc mụn?", False),
        ("So sánh ngắn gọn {left} với {right}: mỗi bên hướng tới mục tiêu nào?", False),
        ("Hãy lập bảng ngắn so sánh {left} và {right} trong routine mụn.", True),
        ("Khi phân biệt {left} và {right}, điểm nào nên được nhắc trước?", False),
    ]
    for left, right in comparison_pairs:
        for template, requires_table in comparison_templates:
            rows.append(
                case(
                    category="comparison_generation",
                    question=template.format(left=left, right=right),
                    expected_keywords=[left.split()[0], right.split()[0]],
                    forbidden_keywords=["giống hệt nhau hoàn toàn"],
                    requires_table=requires_table,
                    expected_format="table" if requires_table else "comparison",
                )
            )

    plan_scenarios = [
        ("mụn đầu trắng", "mụn đầu trắng"),
        ("mụn đầu đen", "mụn đầu đen"),
        ("mụn viêm nhẹ", "mụn viêm"),
        ("da dầu dễ nổi mụn", "da dầu"),
        ("mụn ở lưng", "mụn lưng"),
        ("mụn kèm thâm sau viêm", "thâm"),
        ("da nhạy cảm đang trị mụn", "kích ứng"),
        ("người mới dùng benzoyl peroxide", "benzoyl peroxide"),
        ("người mới dùng adapalene", "adapalene"),
    ]
    plan_templates = [
        "Với {scenario} ở mức tham khảo, nên ưu tiên những mục tiêu chăm sóc nào?",
        "Một hướng tiếp cận cơ bản cho {scenario} có thể gồm các bước nào?",
        "Khi xây dựng routine tham khảo cho {scenario}, vì sao cần đi từ từ với hoạt chất?",
        "Với {scenario}, vai trò của làm sạch dịu nhẹ và dưỡng ẩm là gì?",
        "Nếu {scenario} chưa cải thiện, khi nào nên cân nhắc trao đổi với bác sĩ da liễu?",
    ]
    for scenario, keyword in plan_scenarios:
        for template in plan_templates:
            rows.append(
                case(
                    category="treatment_plan_reference",
                    question=template.format(scenario=scenario),
                    expected_keywords=[keyword],
                    forbidden_keywords=["rửa mặt càng nhiều càng tốt"],
                    expected_format="advice",
                )
            )

    routine_topics = [
        ("routine buổi sáng cho da dầu dễ nổi mụn", "chống nắng"),
        ("routine buổi tối khi dùng adapalene", "adapalene"),
        ("routine có benzoyl peroxide", "benzoyl peroxide"),
        ("routine tối giản cho mụn đầu đen", "mụn đầu đen"),
        ("routine cho da có thâm sau mụn", "thâm"),
        ("routine khi da khô nhẹ do hoạt chất trị mụn", "dưỡng ẩm"),
        ("routine sau khi tập thể dục với da dễ nổi mụn", "làm sạch"),
    ]
    routine_templates = [
        "Nêu các bước cơ bản của {topic}.",
        "Trong {topic}, thứ tự các bước nên được sắp xếp thế nào?",
        "{topic_cap} nên tránh những thói quen nào để hạn chế kích ứng hoặc bít tắc?",
        "Vì sao dưỡng ẩm vẫn quan trọng trong {topic}?",
        "Hãy tóm tắt một routine ngắn, dễ thực hiện cho {topic}.",
    ]
    for topic, keyword in routine_topics:
        for template in routine_templates:
            rows.append(
                case(
                    category="routine_skincare_generation",
                    question=template.format(topic=topic, topic_cap=topic.capitalize()),
                    expected_keywords=[keyword],
                    forbidden_keywords=["dùng càng nhiều hoạt chất càng tốt"],
                    expected_format="advice",
                )
            )

    multi_turn_topics = [
        ("benzoyl peroxide", "benzoyl peroxide"),
        ("adapalene", "adapalene"),
        ("Differin", "differin"),
        ("Epiduo", "epiduo"),
        ("clindamycin bôi", "clindamycin"),
    ]
    multi_turn_templates = [
        "Ở câu trước đã nhắc đến {topic}. Nếu da khô nhẹ khi dùng {topic}, nên điều chỉnh routine thế nào?",
        "Tiếp tục về {topic}: nên bắt đầu tần suất sử dụng theo hướng thận trọng ra sao?",
        "Nếu đang dùng {topic}, vai trò của dưỡng ẩm trong routine là gì?",
        "Liên quan đến {topic}, vì sao không nên thêm nhiều hoạt chất mạnh cùng lúc?",
        "Tóm tắt lại ba lưu ý thực hành khi mới đưa {topic} vào routine mụn.",
    ]
    for topic, keyword in multi_turn_topics:
        for template in multi_turn_templates:
            rows.append(
                case(
                    category="multi_turn_like_generation",
                    question=template.format(topic=topic),
                    expected_keywords=[keyword],
                    forbidden_keywords=["không cần dưỡng ẩm"],
                    expected_format="advice",
                )
            )

    exact_items = [
        ("Nêu đúng 3 lưu ý khi mới dùng benzoyl peroxide.", 3, ["benzoyl peroxide"]),
        ("Nêu đúng 3 lưu ý khi mới dùng adapalene.", 3, ["adapalene"]),
        ("Liệt kê đúng 3 bước routine sáng cơ bản cho da dễ nổi mụn.", 3, ["chống nắng"]),
        ("Liệt kê đúng 3 bước routine tối cơ bản cho da dễ nổi mụn.", 3, ["dưỡng ẩm"]),
        ("Nêu đúng 3 điểm khác nhau giữa mụn đầu đen và mụn đầu trắng.", 3, ["mụn đầu đen"]),
        ("Liệt kê đúng 3 dấu hiệu kích ứng nhẹ khi dùng hoạt chất trị mụn.", 3, ["kích ứng"]),
        ("Nêu đúng 3 lưu ý để hạn chế bít tắc khi chăm sóc da dầu.", 3, ["bít tắc"]),
        ("Liệt kê đúng 3 mục tiêu thường gặp của routine trị mụn.", 3, ["mụn"]),
        ("Nêu đúng 3 điều nên ghi nhớ về Differin.", 3, ["differin"]),
        ("Nêu đúng 3 điều nên ghi nhớ về Epiduo.", 3, ["epiduo"]),
        ("Liệt kê đúng 3 lý do không nên tự nặn mụn viêm.", 3, ["mụn viêm"]),
        ("Nêu đúng 3 lưu ý khi phối hợp benzoyl peroxide và retinoid bôi.", 3, ["benzoyl peroxide"]),
        ("Liệt kê đúng 3 bước chăm sóc khi da khô nhẹ do trị mụn.", 3, ["dưỡng ẩm"]),
        ("Nêu đúng 3 điểm cần nhớ về clindamycin bôi trong mụn.", 3, ["clindamycin"]),
        ("Liệt kê đúng 3 thói quen giúp routine trị mụn dễ duy trì.", 3, ["routine"]),
    ]
    for question, count, keywords in exact_items:
        rows.append(
            case(
                category="exact_format_light",
                question=question,
                expected_keywords=keywords,
                forbidden_keywords=["không có ý nào"],
                expected_format="exact_count",
                expected_count=count,
                requires_bullets=True,
            )
        )

    mild_safety_items = [
        ("Khi mới dùng benzoyl peroxide và thấy khô rát nhẹ, nên lưu ý điều gì?", ["benzoyl peroxide", "kích ứng"]),
        ("Vì sao không nên tự kéo dài clindamycin bôi mà không có hướng dẫn?", ["clindamycin", "kháng sinh"]),
        ("Khi dùng retinoid bôi, vì sao nên thận trọng với kích ứng và chống nắng?", ["retinoid", "kích ứng"]),
        ("Nếu routine trị mụn gây bong tróc nhẹ, nên điều chỉnh các bước cơ bản ra sao?", ["bong tróc", "dưỡng ẩm"]),
        ("Khi mụn viêm kéo dài hoặc để lại sẹo, vì sao nên trao đổi với bác sĩ da liễu?", ["mụn viêm", "bác sĩ"]),
    ]
    for question, keywords in mild_safety_items:
        rows.append(
            case(
                category="mild_safety_caution",
                question=question,
                expected_keywords=keywords,
                forbidden_keywords=["tự tăng liều ngay"],
                expected_format="advice",
            )
        )

    return add_ids(rows)


def validate_cases(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 300:
        raise ValueError(f"Expected 300 cases, got {len(rows)}")
    counts = Counter(row.get("category") for row in rows)
    if counts != CATEGORY_TARGETS:
        raise ValueError(f"Unexpected category distribution: {dict(counts)}")
    ids = [str(row.get("id") or "") for row in rows]
    questions = [str(row.get("question") or "") for row in rows]
    if len(ids) != len(set(ids)) or len(questions) != len(set(questions)):
        raise ValueError("Case ids and questions must be unique")
    for row in rows:
        if not row.get("question") or not row.get("expected_keywords"):
            raise ValueError(f"Invalid case: {row.get('id')}")


def main() -> None:
    rows = build_cases()
    validate_cases(rows)
    OUTPUT_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"Generated {len(rows)} generation-focused cases: {OUTPUT_PATH}")
    print(dict(Counter(row["category"] for row in rows)))


if __name__ == "__main__":
    main()
