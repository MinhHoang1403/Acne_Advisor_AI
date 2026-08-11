from src.retrieval.query_normalization import normalize_query


def test_drug_queries_normalize_taxonomy_entities():
    dalacin = normalize_query("Dalacin T là gì?")
    assert dalacin.intent == "drug_identity"
    assert "Dalacin T" in dalacin.drug_product
    assert "clindamycin" in dalacin.active_ingredient
    assert "topical_antibiotic" in dalacin.drug_class

    epiduo = normalize_query("Epiduo có BPO không?")
    assert epiduo.intent == "ingredient_question"
    assert "Epiduo" in epiduo.drug_product
    assert {"adapalene", "benzoyl_peroxide"}.issubset(epiduo.active_ingredient)

    differin = normalize_query("Differin thuộc nhóm gì?")
    assert differin.intent == "class_check"
    assert "adapalene" in differin.active_ingredient
    assert "topical_retinoid" in differin.drug_class

    differin_ingredient = normalize_query("Differin có hoạt chất chính là gì?")
    assert differin_ingredient.intent == "ingredient_question"
    assert "Differin" in differin_ingredient.drug_product
    assert "adapalene" in differin_ingredient.active_ingredient

    epiduo_ingredients = normalize_query("Epiduo gồm những hoạt chất nào?")
    assert epiduo_ingredients.intent == "ingredient_question"
    assert {"adapalene", "benzoyl_peroxide"}.issubset(epiduo_ingredients.active_ingredient)

    tazorac = normalize_query("Tazorac chứa hoạt chất gì?")
    assert tazorac.intent == "ingredient_question"
    assert "Tazorac" in tazorac.drug_product
    assert "tazarotene" in tazorac.active_ingredient
    assert "topical_retinoid" in tazorac.drug_class

    tazarotene = normalize_query("Tazarotene thuộc nhóm thuốc nào?")
    assert tazarotene.intent == "class_check"
    assert "tazarotene" in tazarotene.active_ingredient
    assert "topical_retinoid" in tazarotene.drug_class


def test_class_contrast_queries_do_not_assign_negative_class():
    bp = normalize_query("Benzoyl peroxide có phải kháng sinh không?")
    assert bp.intent == "class_check"
    assert "benzoyl_peroxide" in bp.active_ingredient
    assert "benzoyl_peroxide" not in bp.drug_class
    assert "topical_antibiotic" not in bp.drug_class

    adapalene = normalize_query("Adapalene có phải kháng sinh không?")
    assert adapalene.intent == "class_check"
    assert "adapalene" in adapalene.active_ingredient
    assert "topical_retinoid" in adapalene.drug_class
    assert "topical_antibiotic" not in adapalene.drug_class


def test_acne_type_query_is_not_classified_as_drug_query():
    blackheads = normalize_query("Mụn đầu đen là gì?")
    assert blackheads.intent == "acne_type"
    assert "acne_vulgaris" in blackheads.condition
    assert "blackheads" in blackheads.metadata["concern"]
    assert blackheads.drug_product == []

    inflammatory = normalize_query("Mụn viêm nên xử lý thế nào?")
    assert inflammatory.intent == "acne_type"
    assert "inflammatory_acne" in inflammatory.metadata["concern"]
    assert inflammatory.drug_product == []


def test_comparison_and_group_queries_keep_all_primary_entities():
    products = normalize_query("Differin và Epiduo khác nhau ở thành phần nào?")
    assert products.intent == "comparison"
    assert {"Differin", "Epiduo"}.issubset(products.drug_product)
    assert {"adapalene", "benzoyl_peroxide"}.issubset(products.active_ingredient)

    tazorac_products = normalize_query(
        "Tazorac, Differin và Epiduo khác nhau về hoạt chất như thế nào?"
    )
    assert tazorac_products.intent == "comparison"
    assert {"Tazorac", "Differin", "Epiduo"}.issubset(tazorac_products.drug_product)
    assert {"tazarotene", "adapalene", "benzoyl_peroxide"}.issubset(
        tazorac_products.active_ingredient
    )

    ingredients = normalize_query("Adapalene và benzoyl peroxide khác nhau thế nào?")
    assert ingredients.intent == "comparison"
    assert {"adapalene", "benzoyl_peroxide"}.issubset(ingredients.active_ingredient)

    shared_class = normalize_query("Adapalene, tretinoin và isotretinoin có cùng nhóm thuốc không?")
    assert shared_class.intent == "class_check"
    assert {"adapalene", "tretinoin", "isotretinoin"}.issubset(shared_class.active_ingredient)
    assert {"topical_retinoid", "oral_retinoid"}.issubset(shared_class.drug_class)


def test_common_typos_and_no_diacritic_mentions_resolve_to_taxonomy_entities():
    differin = normalize_query("diferin co hoat chat gi")
    assert "Differin" in differin.drug_product
    assert "adapalene" in differin.active_ingredient

    adapalene = normalize_query("adapalen thuộc nhóm thuốc gì")
    assert "adapalene" in adapalene.active_ingredient
    assert "topical_retinoid" in adapalene.drug_class

    bp = normalize_query("benzoyl peroxid co phai khang sinh khong")
    assert bp.intent == "class_check"
    assert "benzoyl_peroxide" in bp.active_ingredient

    tazorac = normalize_query("tazorac co hoat chat gi")
    assert "Tazorac" in tazorac.drug_product
    assert "tazarotene" in tazorac.active_ingredient

    tazarotene = normalize_query("tazaroten thuoc nhom nao")
    assert "tazarotene" in tazarotene.active_ingredient
    assert "topical_retinoid" in tazarotene.drug_class


def test_negated_pregnancy_context_does_not_keep_safety_intent():
    corrected = normalize_query("Tôi nói nhầm, tôi không mang thai, chỉ có da nhạy cảm.")
    assert corrected.intent != "safety"
    assert "pregnancy" not in corrected.safety_context


def test_multi_medication_pregnancy_query_is_safety_and_keeps_all_entities():
    query = normalize_query(
        "Tôi đang có thai và hiện dùng adapalene, tazarotene và doxycycline để trị mụn. Tôi nên làm gì?"
    )

    assert query.intent == "safety"
    assert {"adapalene", "tazarotene", "doxycycline"}.issubset(query.active_ingredient)
    assert {"topical_retinoid", "oral_antibiotic"}.issubset(query.drug_class)
