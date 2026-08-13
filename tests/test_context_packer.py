from src.retrieval.context_packer import pack_context, packed_context_to_legacy_contexts
from src.retrieval.contracts import RetrievedCandidate
from src.retrieval.query_normalization import normalize_query


def _entity(
    candidate_id: str,
    canonical_name: str,
    entity_type: str = "drug_product",
    **payload,
) -> RetrievedCandidate:
    base_payload = {
        "entity_id": candidate_id,
        "entity_type": entity_type,
        "canonical_name": canonical_name,
        "active_ingredients": payload.pop("active_ingredients", []),
        "drug_class": payload.pop("drug_class", []),
        "text": f"Entity card for {canonical_name}",
        **payload,
    }
    return RetrievedCandidate(
        candidate_id=candidate_id,
        source="entity",
        collection="acne_entities",
        text=str(base_payload["text"]),
        score=1.0,
        fused_score=1.2,
        payload=base_payload,
        matched_metadata={"canonical_name": [canonical_name]},
    )


def _chunk(candidate_id: str, text: str, **payload) -> RetrievedCandidate:
    base_payload = {
        "chunk_id": candidate_id,
        "source_file": "test.md",
        "header": "Evidence",
        "text": text,
        **payload,
    }
    return RetrievedCandidate(
        candidate_id=candidate_id,
        source="chunk",
        collection="acne_knowledge",
        text=text,
        score=0.3,
        fused_score=0.4,
        payload=base_payload,
        matched_metadata={
            key: value
            for key, value in payload.items()
            if key in {"drug_product", "active_ingredient", "drug_class", "condition", "concern", "content_type", "safety_context", "query_intent_hint"}
        },
    )


def test_drug_identity_packs_entity_and_chunk_evidence():
    normalized = normalize_query("Dalacin T là gì?")
    candidates = [
        _entity(
            "entity:dalacin",
            "Dalacin T",
            active_ingredients=["clindamycin"],
            drug_class=["topical_antibiotic"],
        ),
        _chunk(
            "chunk:dalacin",
            "Dalacin T contains clindamycin and is a topical_antibiotic.",
            drug_product=["Dalacin T"],
            active_ingredient=["clindamycin"],
            drug_class=["topical_antibiotic"],
        ),
    ]

    packed = pack_context(normalized, candidates, max_items=4)

    assert packed.entity_items_count == 1
    assert packed.chunk_items_count == 1
    assert "[ENTITY CARD #1]" in packed.context_text
    assert "[EVIDENCE CHUNK #1]" in packed.context_text
    assert "clindamycin" in packed.context_text
    assert "topical_antibiotic" in packed.context_text


def test_forced_source_does_not_include_empty_text_candidate():
    normalized = normalize_query("Dalacin T là gì?")
    candidates = [
        _entity("entity:dalacin", "Dalacin T", active_ingredients=["clindamycin"]),
        _chunk(
            "chunk:empty",
            "",
            drug_product=["Dalacin T"],
            active_ingredient=["clindamycin"],
        ),
    ]

    packed = pack_context(normalized, candidates, max_items=4)

    assert packed.entity_items_count == 1
    assert packed.chunk_items_count == 0
    assert all(item.text.strip() for item in packed.items)
    assert any(drop["reason"] == "empty_text" for drop in packed.debug["pack_trace"]["dropped_candidates"])
    assert "Drug intent packed context has entity card but no chunk evidence." in packed.warnings


def test_acne_type_prioritizes_chunk_over_drug_entity():
    normalized = normalize_query("Mụn đầu đen là gì?")
    candidates = [
        _entity("entity:dalacin", "Dalacin T", active_ingredients=["clindamycin"]),
        _chunk(
            "chunk:blackheads",
            "Blackheads are comedonal acne.",
            condition=["acne_vulgaris"],
            concern=["blackheads"],
            content_type=["acne_type"],
        ),
    ]

    packed = pack_context(normalized, candidates, max_items=3)

    assert packed.chunk_items_count == 1
    assert all(item.payload.get("entity_type") != "drug_product" for item in packed.items)
    assert "blackheads" in packed.context_text


def test_side_effect_and_safety_roles_use_relevant_metadata():
    side_effect_query = normalize_query("Adapalene có tác dụng phụ gì?")
    side_effect = pack_context(
        side_effect_query,
        [
            _entity(
                "entity:adapalene",
                "adapalene",
                "active_ingredient",
                side_effects=["dryness", "irritation"],
                drug_class=["topical_retinoid"],
            )
        ],
    )
    assert side_effect.items[0].role == "primary_entity"
    assert "Side effects: dryness, irritation" in side_effect.context_text

    safety_query = normalize_query("Retinoid có dùng khi mang thai không?")
    safety = pack_context(
        safety_query,
        [
            _chunk(
                "chunk:pregnancy",
                "Retinoids need pregnancy safety caution.",
                safety_context=["pregnancy"],
                drug_class=["topical_retinoid"],
            )
        ],
    )
    assert safety.items[0].role == "safety_context"
    assert "pregnancy" in safety.context_text


def test_max_items_max_chars_missing_fields_and_dedupe_are_safe():
    normalized = normalize_query("Differin thuộc nhóm gì?")
    duplicate_a = _chunk("same", "Adapalene evidence " * 100, active_ingredient=["adapalene"])
    duplicate_b = _chunk("same", "Duplicate evidence", active_ingredient=["adapalene"])
    extra = _chunk("extra", "Extra evidence", drug_class=["topical_retinoid"])
    missing_optional = _entity("entity:differin", "Differin")

    packed = pack_context(
        normalized,
        [missing_optional, duplicate_a, duplicate_b, extra],
        max_items=3,
        max_chars=220,
    )

    assert len(packed.items) == 3
    assert packed.context_text.endswith("...[truncated]")
    assert any(drop["reason"] == "duplicate" for drop in packed.debug["pack_trace"]["dropped_candidates"])


def test_bridge_to_legacy_contexts_preserves_prompt_fields():
    normalized = normalize_query("Epiduo có BPO không?")
    packed = pack_context(
        normalized,
        [
            _entity("entity:epiduo", "Epiduo", active_ingredients=["adapalene", "benzoyl_peroxide"]),
            _chunk("chunk:epiduo", "Epiduo contains BPO.", active_ingredient=["benzoyl_peroxide"]),
        ],
    )

    contexts = packed_context_to_legacy_contexts(packed)

    assert contexts[0]["text"]
    assert contexts[0]["context_role"] == "primary_entity"
    assert contexts[1]["context_role"] == "supporting_evidence"


def test_comparison_context_keeps_all_primary_entity_cards():
    normalized = normalize_query("Differin và Epiduo khác nhau ở thành phần nào?")
    candidates = [
        _chunk(
            f"chunk:{index}",
            f"General comparison evidence {index} about adapalene and benzoyl peroxide.",
            active_ingredient=["adapalene", "benzoyl_peroxide"],
            query_intent_hint=["comparison"],
        )
        for index in range(5)
    ]
    candidates.extend(
        [
            _entity(
                "drug_product:differin",
                "Differin",
                active_ingredients=["adapalene"],
                aliases=["differin"],
            ),
            _entity(
                "drug_product:epiduo",
                "Epiduo",
                active_ingredients=["adapalene", "benzoyl_peroxide"],
                aliases=["epiduo"],
            ),
        ]
    )

    packed = pack_context(normalized, candidates, max_items=5)
    selected_entity_ids = packed.debug["pack_trace"]["selected_entity_ids"]

    assert "drug_product:differin" in selected_entity_ids
    assert "drug_product:epiduo" in selected_entity_ids


def test_safety_context_keeps_all_named_medication_entity_cards():
    normalized = normalize_query(
        "Tôi đang có thai và hiện dùng adapalene, tazarotene và doxycycline để trị mụn. Tôi nên làm gì?"
    )
    candidates = [
        _chunk(
            f"chunk:pregnancy-{index}",
            f"General pregnancy safety evidence {index}.",
            safety_context=["pregnancy"],
            query_intent_hint=["safety"],
        )
        for index in range(4)
    ]
    candidates.extend(
        [
            _entity(
                "active_ingredient:adapalene",
                "adapalene",
                "active_ingredient",
                drug_class=["topical_retinoid"],
                safety_contexts=["pregnancy"],
            ),
            _entity(
                "active_ingredient:tazarotene",
                "tazarotene",
                "active_ingredient",
                drug_class=["topical_retinoid"],
                safety_contexts=["pregnancy"],
            ),
            _entity(
                "active_ingredient:doxycycline",
                "doxycycline",
                "active_ingredient",
                drug_class=["oral_antibiotic"],
                safety_contexts=["pregnancy"],
            ),
        ]
    )

    packed = pack_context(normalized, candidates, max_items=3)
    selected_entity_ids = set(packed.debug["pack_trace"]["selected_entity_ids"])

    assert normalized.intent == "safety"
    assert selected_entity_ids == {
        "active_ingredient:adapalene",
        "active_ingredient:tazarotene",
        "active_ingredient:doxycycline",
    }
    assert "adapalene" in packed.context_text
    assert "tazarotene" in packed.context_text
    assert "doxycycline" in packed.context_text


def test_pack_context_drops_near_duplicate_long_text_but_keeps_safety_evidence():
    normalized = normalize_query("Retinoid có dùng khi mang thai không?")
    duplicate_text = "Retinoids require pregnancy safety counseling and contraception. " * 8
    packed = pack_context(
        normalized,
        [
            _chunk(
                "chunk:pregnancy-a",
                duplicate_text,
                safety_context=["pregnancy"],
                drug_class=["topical_retinoid"],
            ),
            _chunk(
                "chunk:pregnancy-b",
                duplicate_text + "Minor trailing difference.",
                safety_context=["pregnancy"],
                drug_class=["topical_retinoid"],
            ),
            _chunk(
                "chunk:irritation",
                "Retinoids can irritate skin.",
                safety_context=["irritation"],
                drug_class=["topical_retinoid"],
            ),
        ],
        max_items=3,
        max_chars=900,
    )

    assert packed.chunk_items_count == 2
    assert "pregnancy" in packed.context_text
    assert any(
        drop["reason"] == "near_duplicate_text"
        for drop in packed.debug["pack_trace"]["dropped_candidates"]
    )
