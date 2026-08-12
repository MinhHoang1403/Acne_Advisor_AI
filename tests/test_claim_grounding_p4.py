from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agent.nodes import claim_grounding as node_module
from src.quality.claim_grounding import (
    AnswerClaim,
    ClaimCriticality,
    ClaimEntailmentVerdict,
    ClaimEvidenceLink,
    ClaimGroundingResult,
    EntailmentStatus,
    EvidenceScope,
    P4Mode,
    ShadowPolicyAction,
    compact_claim_grounding,
    evaluate_claim_grounding,
    extract_answer_claims,
    map_claims_to_evidence,
    verify_claim_entailment,
)
from src.retrieval.contracts import ContextItem, PackedContext
from src.observability.versioning import build_pipeline_version_manifest, compute_pipeline_fingerprint


def _context(*items: tuple[str, str, str | None]) -> PackedContext:
    contexts = []
    for evidence_id, text, source_path in items:
        payload = {"chunk_id": evidence_id}
        if source_path is not None:
            payload["source_path"] = source_path
        contexts.append(
            ContextItem(
                item_id=evidence_id,
                source="chunk",
                role="primary",
                text=text,
                payload=payload,
                reason="fixture",
            )
        )
    return PackedContext(
        original_query="fixture",
        intent="definition",
        items=contexts,
        context_text="\n".join(item.text for item in contexts),
    )


def _single_result(claim: str, evidence: str, *, query: str = "Mụn là gì?") -> ClaimGroundingResult:
    return evaluate_claim_grounding(
        answer=claim,
        query=query,
        packed_context=_context(("chunk-1", evidence, "guideline.pdf")),
        p3_status="SUFFICIENT",
    )


def test_claim_contract_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        AnswerClaim(
            claim_id="claim:1",
            text="Adapalene là retinoid bôi.",
            normalized_text="adapalene la retinoid boi",
            sentence_index=0,
            claim_index=0,
            claim_type="DEFINITION",
            criticality="NON_CRITICAL",
            hidden_reasoning="not allowed",
        )


def test_extraction_handles_headings_bullets_unicode_and_non_claim_text():
    answer = """## Tổng quan
- Adapalene là một retinoid bôi trị mụn.
- Benzoyl peroxide có tác dụng kháng khuẩn với C. acnes.

Tôi có thể giúp gì thêm?"""
    claims, flags = extract_answer_claims(answer, query="So sánh adapalene và benzoyl peroxide")
    assert [claim.text for claim in claims] == [
        "Adapalene là một retinoid bôi trị mụn.",
        "Benzoyl peroxide có tác dụng kháng khuẩn với C. acnes.",
    ]
    assert flags == {"extraction_truncated": False, "critical_claim_overflow": False}


def test_extraction_retains_critical_claims_when_soft_bound_is_reached():
    answer = "\n".join(
        [
            "Adapalene là một retinoid bôi trị mụn.",
            "Benzoyl peroxide có tác dụng kháng khuẩn trên da mụn.",
            "Khi mang thai không nên tự dùng tazarotene trị mụn.",
        ]
    )
    claims, flags = extract_answer_claims(answer, query="Thuốc trị mụn", max_claims=2)
    assert len(claims) == 2
    assert any(claim.criticality == ClaimCriticality.CRITICAL for claim in claims)
    assert flags["extraction_truncated"] is True


def test_extraction_handles_markdown_table_without_treating_header_as_claim():
    answer = """| Hoạt chất | Nhóm | Vai trò |
|---|---|---|
| Adapalene | Retinoid bôi | Giảm bít tắc và nhân mụn |
| Benzoyl peroxide | Không phải kháng sinh | Kháng khuẩn với C. acnes |"""
    claims, _ = extract_answer_claims(answer, query="So sánh adapalene và benzoyl peroxide")
    assert len(claims) == 2
    assert "Adapalene" in claims[0].text
    assert "Benzoyl peroxide" in claims[1].text


def test_extraction_splits_two_medical_propositions_in_one_sentence_conservatively():
    claims, _ = extract_answer_claims(
        "Adapalene là retinoid bôi và benzoyl peroxide không phải là kháng sinh.",
        query="So sánh adapalene và benzoyl peroxide",
    )
    assert [claim.text for claim in claims] == [
        "Adapalene là retinoid bôi",
        "benzoyl peroxide không phải là kháng sinh.",
    ]


def test_nonmedical_conversation_produces_no_claims():
    claims, _ = extract_answer_claims("Cảm ơn bạn. Tôi có thể giúp gì thêm?")
    assert claims == ()


def test_mapping_accepts_only_packed_chunk_with_valid_provenance():
    claim = extract_answer_claims("Adapalene là một retinoid bôi trị mụn.")[0][0]
    context = _context(
        ("valid", "Adapalene là một retinoid bôi trị mụn.", "guide.pdf"),
        ("invalid", "Adapalene là retinoid.", None),
    )
    context = context.model_copy(
        update={
            "items": context.items
            + [
                ContextItem(
                    item_id="entity:adapalene",
                    source="entity",
                    role="entity",
                    text="Adapalene",
                    payload={"entity_id": "adapalene"},
                    reason="EntitySignal",
                ),
                ContextItem(
                    item_id="graph:adapalene",
                    source="entity",
                    role="graph_signal",
                    text="Adapalene BELONGS_TO_CLASS retinoid",
                    payload={"source_chunk_ids": ["valid"]},
                    reason="GraphSignal",
                ),
            ]
        }
    )
    mapped, links = map_claims_to_evidence((claim,), context)
    assert mapped[0].mapped_evidence_ids == ("valid",)
    assert [link.evidence_id for link in links] == ["valid"]
    assert all(link.scope == EvidenceScope.GENERATION_CONTEXT_EVIDENCE for link in links)


def test_direct_citation_mapping_has_priority():
    claim = extract_answer_claims("Theo guide.pdf, adapalene là retinoid bôi trị mụn.")[0][0]
    mapped, links = map_claims_to_evidence(
        (claim,),
        _context(("chunk-1", "Adapalene là retinoid bôi trị mụn.", "guide.pdf")),
    )
    assert mapped[0].mapped_evidence_ids == ("chunk-1",)
    assert links[0].mapping_reason == "DIRECT_CITATION"


@pytest.mark.parametrize(
    ("claim", "evidence", "expected"),
    [
        (
            "Tazarotene là một retinoid bôi dùng để điều trị mụn.",
            "Tazarotene là một retinoid bôi dùng để điều trị mụn.",
            EntailmentStatus.SUPPORTED,
        ),
        (
            "Retinoid bôi luôn luôn chữa khỏi hoàn toàn mụn đầu đen trong một tuần.",
            "Retinoid bôi có thể cải thiện mụn đầu đen.",
            EntailmentStatus.PARTIALLY_SUPPORTED,
        ),
        (
            "Adapalene làm bạc màu tóc và vải.",
            "Adapalene là retinoid bôi có thể gây khô và kích ứng da.",
            EntailmentStatus.UNSUPPORTED,
        ),
        (
            "Benzoyl peroxide là kháng sinh.",
            "Benzoyl peroxide không phải là kháng sinh.",
            EntailmentStatus.CONTRADICTED,
        ),
    ],
)
def test_entailment_statuses_are_distinct(claim: str, evidence: str, expected: EntailmentStatus):
    result = _single_result(claim, evidence, query=claim)
    assert len(result.verdicts) == 1
    assert result.verdicts[0].verdict == expected


def test_no_candidate_evidence_is_no_evidence_not_verifier_error():
    result = evaluate_claim_grounding(
        answer="Adapalene là một retinoid bôi trị mụn.",
        query="Adapalene là gì?",
        packed_context=_context(("chunk-x", "Benzoyl peroxide có tác dụng kháng khuẩn.", "bp.pdf")),
        p3_status="SUFFICIENT",
    )
    assert result.verdicts[0].verdict == EntailmentStatus.NO_EVIDENCE


def test_invalid_evidence_schema_degrades_to_verifier_error(monkeypatch):
    claim = extract_answer_claims("Adapalene là một retinoid bôi trị mụn.")[0][0]
    link = ClaimEvidenceLink(
        claim_id=claim.claim_id,
        evidence_id="chunk-1",
        source_id="guide.pdf",
        mapping_reason="LEXICAL_OVERLAP",
        provenance_valid=True,
        rank=1,
    )
    monkeypatch.setattr("src.quality.claim_grounding._packed_context", lambda _: (_ for _ in ()).throw(ValueError("bad")))
    verdict = verify_claim_entailment(claim, (link,), {})
    assert verdict.verdict == EntailmentStatus.VERIFIER_ERROR
    assert verdict.reason_code == "INVALID_VERIFIER_OUTPUT"


def test_critical_failure_is_would_block_critical():
    result = evaluate_claim_grounding(
        answer="Tazarotene an toàn khi mang thai.",
        query="Tazarotene có dùng khi mang thai không?",
        packed_context=_context(("pregnancy", "Tazarotene không được dùng trong thai kỳ.", "label.pdf")),
        p3_status="SUFFICIENT",
    )
    assert result.claims[0].criticality == ClaimCriticality.CRITICAL
    assert result.verdicts[0].verdict == EntailmentStatus.CONTRADICTED
    assert result.shadow_action == ShadowPolicyAction.WOULD_BLOCK_CRITICAL
    assert result.verified_claims.critical_failures == (result.claims[0].claim_id,)


def test_noncritical_partial_and_unsupported_shadow_actions():
    partial = _single_result(
        "Retinoid bôi luôn luôn chữa khỏi hoàn toàn mụn đầu đen trong một tuần.",
        "Retinoid bôi có thể cải thiện mụn đầu đen.",
    )
    unsupported = _single_result(
        "Adapalene làm bạc màu tóc và vải.",
        "Adapalene là retinoid bôi có thể gây khô da.",
    )
    assert partial.shadow_action == ShadowPolicyAction.WOULD_REWRITE_PARTIAL
    assert unsupported.shadow_action in {
        ShadowPolicyAction.WOULD_ABSTAIN,
        ShadowPolicyAction.WOULD_DROP_NONCRITICAL,
    }


def test_p3_insufficiency_has_precedence_and_p4_does_not_claim_grounding():
    result = evaluate_claim_grounding(
        answer="Adapalene là retinoid bôi.",
        query="Adapalene là gì?",
        packed_context=None,
        p3_status="CRITICAL_EVIDENCE_MISSING",
    )
    assert result.status == "skipped_p3_precedence"
    assert result.degraded is True
    assert result.claims == ()
    assert result.shadow_action == ShadowPolicyAction.WOULD_ABSTAIN


@pytest.mark.asyncio
async def test_shadow_node_preserves_draft_byte_for_byte(monkeypatch):
    monkeypatch.setenv("P4_MODE", "shadow")
    draft = "Adapalene là một retinoid bôi trị mụn."
    result = await node_module.claim_grounding_node(
        {
            "draft_answer": draft,
            "user_question": "Adapalene là gì?",
            "packed_context": _context(("chunk-1", draft, "guide.pdf")).model_dump(mode="json"),
            "evidence_sufficiency": {"status": "SUFFICIENT"},
            "pipeline_manifest": {"p4_mode": "shadow"},
            "performance_timings": {},
            "retrieval_diagnostics": {},
        }
    )
    assert result["draft_answer"] == draft
    assert result["p4_answer_modified"] is False
    assert result["claim_grounding"]["production_answer_modified"] is False


@pytest.mark.asyncio
async def test_shadow_node_failure_returns_original_draft(monkeypatch):
    monkeypatch.setattr(node_module, "evaluate_claim_grounding", lambda **_: (_ for _ in ()).throw(RuntimeError("provider timeout")))
    result = await node_module.claim_grounding_node(
        {
            "draft_answer": "Adapalene là retinoid bôi.",
            "user_question": "Adapalene là gì?",
            "pipeline_manifest": {"p4_mode": "shadow"},
        }
    )
    assert result["draft_answer"] == "Adapalene là retinoid bôi."
    assert result["p4_degraded"] is True
    assert result["p4_answer_modified"] is False
    assert result["p4_shadow_policy"] == "VERIFIER_UNAVAILABLE"
    assert result["p4_trace"][0]["event"] == "CLAIM_VERIFIER_FAILED"


def test_compact_metadata_contains_counts_but_no_claim_or_evidence_text():
    result = _single_result(
        "Tazarotene là một retinoid bôi dùng để điều trị mụn.",
        "Tazarotene là một retinoid bôi dùng để điều trị mụn.",
    )
    compact = compact_claim_grounding(result)
    assert compact["claim_count"] == 1
    assert compact["supported_count"] == 1
    assert "claims" not in compact
    assert "evidence_links" not in compact


def test_trace_is_bounded_and_contains_required_shadow_events():
    result = _single_result(
        "Tazarotene là một retinoid bôi dùng để điều trị mụn.",
        "Tazarotene là một retinoid bôi dùng để điều trị mụn.",
    )
    events = [item.event for item in result.trace]
    assert events == [
        "CLAIM_EXTRACTION_STARTED",
        "CLAIM_EXTRACTION_COMPLETED",
        "CLAIM_MAPPED",
        "CLAIM_VERIFICATION_STARTED",
        "CLAIM_VERIFIED",
        "SHADOW_POLICY_EVALUATED",
    ]
    assert len(result.trace) <= 96


def test_p4_mode_and_contract_versions_participate_in_fingerprint_without_cache_bump():
    shadow = build_pipeline_version_manifest({"P4_MODE": "shadow", "CACHE_ANSWER_VERSION": "v5"})
    disabled = build_pipeline_version_manifest({"P4_MODE": "disabled", "CACHE_ANSWER_VERSION": "v5"})
    assert shadow["p4_mode"] == "shadow"
    assert shadow["p4_claim_grounding_version"] == "claim_level_grounding_v1"
    assert shadow["answer_cache_version"] == disabled["answer_cache_version"] == "v5"
    assert compute_pipeline_fingerprint(shadow) != compute_pipeline_fingerprint(disabled)


def test_enforcement_modes_are_interlocked_until_calibration_is_explicitly_ready():
    blocked = build_pipeline_version_manifest({"P4_MODE": "enforce_critical"})
    critical_ready = build_pipeline_version_manifest(
        {"P4_MODE": "enforce_critical", "P4_CRITICAL_ENFORCEMENT_READY": "true"}
    )
    all_blocked = build_pipeline_version_manifest(
        {"P4_MODE": "enforce_all", "P4_CRITICAL_ENFORCEMENT_READY": "true"}
    )
    assert blocked["p4_requested_mode"] == "enforce_critical"
    assert blocked["p4_mode"] == "shadow"
    assert critical_ready["p4_mode"] == "enforce_critical"
    assert all_blocked["p4_mode"] == "shadow"


def test_verifier_error_cannot_be_constructed_as_supported_by_default():
    verdict = ClaimEntailmentVerdict(
        claim_id="claim:1",
        verdict=EntailmentStatus.VERIFIER_ERROR,
        reason_code="INVALID_VERIFIER_OUTPUT",
    )
    assert verdict.verdict != EntailmentStatus.SUPPORTED


def test_disabled_mode_has_immediate_no_data_migration_rollback():
    result = evaluate_claim_grounding(
        answer="Adapalene là retinoid bôi.",
        query="Adapalene là gì?",
        packed_context=None,
        mode=P4Mode.DISABLED,
    )
    assert result.status == "disabled"
    assert result.claims == ()
