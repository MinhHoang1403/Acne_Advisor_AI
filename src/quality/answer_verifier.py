"""Rule-based answer quality verifier and guard.

The verifier is deterministic and offline-only. It never calls LLMs or
external APIs.
"""

from __future__ import annotations

import re
from typing import Any

from src.agent.answer_formatting import assess_structural_quality, infer_response_profile
from src.agent.requested_structure import canonical_column_name, parse_requested_structure
from src.retrieval.contracts import NormalizedQuery, PackedContext
from src.retrieval.query_normalization import normalize_query
from src.quality.contracts import (
    AnswerGuardResult,
    AnswerQualityIssue,
    AnswerVerificationReport,
    DomainProposition,
)
from src.quality.proposition_detector import (
    extract_domain_propositions,
    proposition_exists,
)
from src.quality.vietnamese_text import build_matching_views


CRITICAL = "critical"
ERROR = "error"
WARNING = "warning"


def verify_answer_quality(
    query: str,
    answer: str,
    normalized_query: NormalizedQuery | None = None,
    packed_context: PackedContext | None = None,
    retrieval_trace: Any | None = None,
) -> AnswerVerificationReport:
    """Verify answer quality with deterministic medical/domain rules."""

    normalized_query = normalized_query or normalize_query(query)
    answer = answer or ""
    query_norm = _norm(query)
    answer_norm = _norm(answer)
    propositions = extract_domain_propositions(answer, query_context=query)
    issues: list[AnswerQualityIssue] = []
    required: list[str] = []
    detected: list[str] = []
    contradictions: list[str] = []
    safety_warnings: list[str] = []

    _check_bp_antibiotic(query_norm, answer_norm, propositions, issues, required, detected, contradictions)
    _check_clindamycin_retinoid(query_norm, answer_norm, propositions, issues, required, detected, contradictions)
    _check_adapalene_antibiotic(query_norm, answer_norm, propositions, issues, required, detected, contradictions)
    _check_dalacin_identity(query_norm, answer_norm, propositions, issues, required, detected, contradictions)
    _check_epiduo_ingredients(query_norm, answer_norm, propositions, issues, required, detected, contradictions)
    _check_differin_class(query_norm, answer_norm, propositions, issues, required, detected, contradictions)
    _check_topical_antibiotic_caution(query_norm, answer_norm, issues, safety_warnings)
    _check_retinoid_pregnancy_safety(query_norm, answer_norm, issues, contradictions, safety_warnings)
    _check_isotretinoin_safety(answer_norm, issues, contradictions, safety_warnings)
    _check_acne_type_answer(query_norm, answer_norm, issues, required, detected)
    _check_antibiotic_multi_intent(query_norm, answer_norm, issues, required, detected)
    _check_multi_entity_pregnancy_coverage(query_norm, answer_norm, normalized_query, issues, required, detected)
    _check_self_harm_crisis_response(query_norm, answer_norm, issues)
    _check_acne_fulminans_urgency(query_norm, answer_norm, issues)
    _check_requested_table_schema(query, answer, issues)
    _check_irrelevant_topical_warning(query_norm, answer_norm, issues)
    _check_structural_presentation(query, answer, issues)

    missing = [fact for fact in _dedupe(required) if fact not in _dedupe(detected)]
    for fact in missing:
        if _required_fact_is_important(fact, normalized_query.intent):
            issues.append(
                AnswerQualityIssue(
                    code="missing_required_fact",
                    severity=WARNING,
                    message=f"Answer may be missing required fact: {fact}.",
                    evidence={"fact": fact},
                    suggested_fix=f"Include: {fact}.",
                )
            )

    has_error = any(issue.severity in {ERROR, CRITICAL} for issue in issues)
    return AnswerVerificationReport(
        passed=not has_error,
        original_query=query,
        intent=normalized_query.intent,
        checked_answer=answer,
        issues=issues,
        required_facts=_dedupe(required),
        detected_facts=_dedupe(detected),
        missing_facts=missing,
        contradictions=_dedupe(contradictions),
        safety_warnings=_dedupe(safety_warnings),
        metadata={
            "packed_context_items": len(packed_context.items) if packed_context else 0,
            "retrieval_trace_available": retrieval_trace is not None,
            "answer_verifier": {
                "version": "answer_verifier_v2",
                "proposition_count": len(propositions),
                "proposition_rules": _proposition_rule_counts(propositions),
            },
            "normalized_entities": {
                "drug_product": normalized_query.drug_product,
                "active_ingredient": normalized_query.active_ingredient,
                "drug_class": normalized_query.drug_class,
                "condition": normalized_query.condition,
            },
        },
    )


def apply_answer_guard(
    query: str,
    answer: str,
    normalized_query: NormalizedQuery | None = None,
    packed_context: PackedContext | None = None,
    retrieval_trace: Any | None = None,
    mode: str = "metadata_only",
) -> AnswerGuardResult:
    """Apply answer guard in metadata-only, append-warnings, or strict-safe mode."""

    report = verify_answer_quality(
        query=query,
        answer=answer,
        normalized_query=normalized_query,
        packed_context=packed_context,
        retrieval_trace=retrieval_trace,
    )
    mode = (mode or "metadata_only").strip().lower()
    if mode == "metadata_only":
        return AnswerGuardResult(
            answer=answer,
            original_answer=answer,
            report=report,
            modified=False,
        )

    if mode == "append_warnings" and report.issues:
        warning_text = _short_warning(report)
        if warning_text and warning_text not in answer:
            return AnswerGuardResult(
                answer=answer.rstrip() + "\n\n**Lưu ý an toàn**\n" + warning_text,
                original_answer=answer,
                report=report,
                modified=True,
                modification_reason="appended_quality_warning",
            )

    if mode == "strict_safe" and any(issue.severity == CRITICAL for issue in report.issues):
        return AnswerGuardResult(
            answer=_strict_safe_answer(query, report),
            original_answer=answer,
            report=report,
            modified=True,
            modification_reason="critical_quality_contradiction",
        )

    return AnswerGuardResult(
        answer=answer,
        original_answer=answer,
        report=report,
        modified=False,
    )


def _check_bp_antibiotic(
    query: str,
    answer: str,
    propositions: list[DomainProposition],
    issues: list[AnswerQualityIssue],
    required: list[str],
    detected: list[str],
    contradictions: list[str],
) -> None:
    if not ("benzoyl peroxide" in query or "bpo" in query or re.search(r"\bbp\b", query)):
        return
    if "khang sinh" not in query and "antibiotic" not in query:
        return
    required.append("benzoyl peroxide is not an antibiotic")
    if _has_prop(propositions, "benzoyl_peroxide", "is_not_a", "antibiotic"):
        detected.append("benzoyl peroxide is not an antibiotic")
    if _has_prop(propositions, "benzoyl_peroxide", "is_a", "antibiotic"):
        contradictions.append("benzoyl peroxide incorrectly described as antibiotic")
        issues.append(
            _issue(
                "bp_antibiotic_contradiction",
                CRITICAL,
                "Benzoyl peroxide was described as an antibiotic.",
                evidence=_evidence_for(propositions, "benzoyl_peroxide", "is_a", "antibiotic"),
            )
        )
    elif "benzoyl peroxide" not in answer and "bpo" not in answer:
        if "benzoyl peroxide is not an antibiotic" not in detected:
            issues.append(_issue("bp_missing_direct_entity", ERROR, "Answer does not directly address benzoyl peroxide."))
    elif "benzoyl peroxide is not an antibiotic" not in detected:
        issues.append(_issue("bp_missing_not_antibiotic", ERROR, "Answer should say benzoyl peroxide is not an antibiotic."))


def _check_clindamycin_retinoid(query: str, answer: str, propositions: list[DomainProposition], issues: list[AnswerQualityIssue], required: list[str], detected: list[str], contradictions: list[str]) -> None:
    if not _contains_any(query, ["clindamycin", "dalacin"]) or "retinoid" not in query:
        return
    required.extend(["clindamycin is not a retinoid", "clindamycin is a topical antibiotic"])
    if _has_prop(propositions, "clindamycin", "is_not_a", "retinoid"):
        detected.append("clindamycin is not a retinoid")
    if _has_prop(propositions, "clindamycin", "is_a", "topical_antibiotic") or _contains_any(answer, ["khang sinh boi", "topical antibiotic", "topical_antibiotic"]):
        detected.append("clindamycin is a topical antibiotic")
    if _has_prop(propositions, "clindamycin", "is_a", "retinoid"):
        contradictions.append("clindamycin incorrectly described as retinoid")
        issues.append(
            _issue(
                "clindamycin_retinoid_contradiction",
                CRITICAL,
                "Clindamycin was described as a retinoid.",
                evidence=_evidence_for(propositions, "clindamycin", "is_a", "retinoid"),
            )
        )


def _check_adapalene_antibiotic(query: str, answer: str, propositions: list[DomainProposition], issues: list[AnswerQualityIssue], required: list[str], detected: list[str], contradictions: list[str]) -> None:
    if not _contains_any(query, ["adapalene", "differin"]) or ("khang sinh" not in query and "antibiotic" not in query):
        return
    required.extend(["adapalene is not an antibiotic", "adapalene is a topical retinoid"])
    if _has_prop(propositions, "adapalene", "is_not_a", "antibiotic"):
        detected.append("adapalene is not an antibiotic")
    if _has_prop(propositions, "adapalene", "is_a", "retinoid") or _contains_any(answer, ["retinoid boi", "topical retinoid", "topical_retinoid"]):
        detected.append("adapalene is a topical retinoid")
    if _has_prop(propositions, "adapalene", "is_a", "antibiotic"):
        contradictions.append("adapalene incorrectly described as antibiotic")
        issues.append(
            _issue(
                "adapalene_antibiotic_contradiction",
                CRITICAL,
                "Adapalene was described as an antibiotic.",
                evidence=_evidence_for(propositions, "adapalene", "is_a", "antibiotic"),
            )
        )


def _check_dalacin_identity(query: str, answer: str, propositions: list[DomainProposition], issues: list[AnswerQualityIssue], required: list[str], detected: list[str], contradictions: list[str]) -> None:
    if "dalacin" not in query:
        return
    required.extend(["Dalacin T contains clindamycin", "Dalacin T is a topical antibiotic"])
    if _has_prop(propositions, "Dalacin T", "contains", "clindamycin") or "clindamycin" in answer:
        detected.append("Dalacin T contains clindamycin")
    if _has_prop(propositions, "clindamycin", "is_a", "topical_antibiotic") or _contains_any(answer, ["khang sinh boi", "topical antibiotic", "topical_antibiotic"]):
        detected.append("Dalacin T is a topical antibiotic")
    if _has_prop(propositions, "clindamycin", "is_a", "retinoid"):
        contradictions.append("Dalacin T incorrectly described as retinoid")
        issues.append(_issue("dalacin_retinoid_contradiction", CRITICAL, "Dalacin T was described as a retinoid."))


def _check_epiduo_ingredients(query: str, answer: str, propositions: list[DomainProposition], issues: list[AnswerQualityIssue], required: list[str], detected: list[str], contradictions: list[str]) -> None:
    if "epiduo" not in query:
        return
    if not _contains_any(query, ["bpo", "thanh phan", "co benzoyl", "co bpo", "ingredient"]):
        return
    required.extend(["Epiduo contains adapalene", "Epiduo contains benzoyl peroxide"])
    if _has_prop(propositions, "Epiduo", "contains", "adapalene") or "adapalene" in answer:
        detected.append("Epiduo contains adapalene")
    if _has_prop(propositions, "Epiduo", "contains", "benzoyl_peroxide") or "benzoyl peroxide" in answer or "bpo" in answer:
        detected.append("Epiduo contains benzoyl peroxide")
    if _has_prop(propositions, "benzoyl_peroxide", "is_a", "antibiotic"):
        contradictions.append("benzoyl peroxide incorrectly described as antibiotic")
        issues.append(_issue("epiduo_bp_antibiotic_contradiction", CRITICAL, "Epiduo answer calls benzoyl peroxide an antibiotic."))
    if ("adapalene" in answer) ^ ("benzoyl peroxide" in answer or "bpo" in answer):
        issues.append(_issue("epiduo_incomplete_ingredients", WARNING, "Epiduo ingredient answer mentions only one key ingredient."))


def _check_differin_class(query: str, answer: str, propositions: list[DomainProposition], issues: list[AnswerQualityIssue], required: list[str], detected: list[str], contradictions: list[str]) -> None:
    if "differin" not in query:
        return
    required.extend(["Differin contains adapalene", "Differin is a topical retinoid"])
    if _has_prop(propositions, "Differin", "contains", "adapalene") or "adapalene" in answer:
        detected.append("Differin contains adapalene")
    if _has_prop(propositions, "adapalene", "is_a", "retinoid") or _contains_any(answer, ["retinoid boi", "topical retinoid", "topical_retinoid"]):
        detected.append("Differin is a topical retinoid")
    if _has_prop(propositions, "adapalene", "is_a", "antibiotic"):
        contradictions.append("Differin incorrectly described as antibiotic")
        issues.append(_issue("differin_antibiotic_contradiction", CRITICAL, "Differin was described as an antibiotic."))


def _check_topical_antibiotic_caution(query: str, answer: str, issues: list[AnswerQualityIssue], safety_warnings: list[str]) -> None:
    mentions_topical_antibiotic = _contains_any(answer, ["khang sinh boi", "topical antibiotic", "clindamycin", "erythromycin"])
    recommends = _contains_any(answer, ["nen dung", "co the dung", "su dung", "dung clindamycin", "apply", "use"])
    has_caution = _contains_any(answer, ["khong nen dung don doc", "khong dung don doc", "khong nen keo dai", "khong dung keo dai", "giam nguy co khang khang sinh", "antibiotic resistance"])
    if mentions_topical_antibiotic and recommends and not has_caution:
        safety_warnings.append("topical antibiotic caution missing")
        issues.append(_issue("topical_antibiotic_caution_missing", WARNING, "Topical antibiotic recommendation should caution against monotherapy or prolonged use."))


def _check_retinoid_pregnancy_safety(query: str, answer: str, issues: list[AnswerQualityIssue], contradictions: list[str], safety_warnings: list[str]) -> None:
    pregnancy_context = _contains_any(query + " " + answer, ["mang thai", "thai ky", "co thai", "pregnancy", "pregnant", "cho con bu"])
    mentions_retinoid = _contains_any(answer, ["retinoid", "adapalene", "tretinoin", "isotretinoin"])
    if not (pregnancy_context and mentions_retinoid):
        return
    has_doctor_caution = _contains_any(answer, ["hoi bac si", "bac si", "doctor", "dermatologist", "da lieu", "san khoa", "khong nen tu dung"])
    if has_doctor_caution:
        safety_warnings.append("retinoid pregnancy doctor caution present")
    else:
        issues.append(_issue("retinoid_pregnancy_caution_missing", ERROR, "Retinoid pregnancy/breastfeeding context should advise clinician review."))
    if _contains_any(answer, ["nen dung retinoid khi mang thai", "co the dung retinoid khi mang thai", "safe during pregnancy", "an toan khi mang thai"]):
        contradictions.append("retinoid recommended as safe in pregnancy")
        issues.append(_issue("retinoid_pregnancy_unsafe", CRITICAL, "Answer recommends retinoid use in pregnancy too confidently."))


def _check_isotretinoin_safety(answer: str, issues: list[AnswerQualityIssue], contradictions: list[str], safety_warnings: list[str]) -> None:
    if "isotretinoin" not in answer:
        return
    has_doctor = _contains_any(answer, ["bac si", "doctor", "dermatologist", "da lieu", "chi dinh", "theo doi", "ke don"])
    if has_doctor:
        safety_warnings.append("isotretinoin clinician supervision present")
    else:
        issues.append(_issue("isotretinoin_supervision_missing", ERROR, "Isotretinoin mention should include clinician supervision."))
    if _contains_any(answer, ["tu dung isotretinoin", "tu uong isotretinoin", "self-use isotretinoin", "take isotretinoin yourself"]):
        contradictions.append("isotretinoin self-use recommended")
        issues.append(_issue("isotretinoin_self_use", CRITICAL, "Answer recommends self-use of isotretinoin."))


def _check_acne_type_answer(query: str, answer: str, issues: list[AnswerQualityIssue], required: list[str], detected: list[str]) -> None:
    acne_type_query = _contains_any(query, ["mun dau den", "blackhead", "mun viem", "inflamed acne", "inflammatory acne"])
    if not acne_type_query:
        return
    required.append("answer addresses acne type")
    lesion_terms = ["mun dau den", "blackhead", "comedone", "nhan mo", "mun viem", "viem", "sang thuong", "inflammatory"]
    if _contains_any(answer, lesion_terms):
        detected.append("answer addresses acne type")
    drug_mentions = sum(1 for term in ["dalacin", "epiduo", "differin", "clindamycin", "adapalene", "benzoyl peroxide"] if term in answer)
    lesion_specific_mentions = sum(1 for term in lesion_terms + ["acne type", "lesion type"] if term in answer)
    if drug_mentions >= 2 and lesion_specific_mentions == 0:
        issues.append(_issue("acne_type_drug_only_answer", ERROR, "Acne type answer appears drug-only and does not address the lesion type."))
    elif drug_mentions >= 2 and "answer addresses acne type" not in detected:
        issues.append(_issue("acne_type_overfocused_on_drugs", WARNING, "Acne type answer may be over-focused on named drugs."))


def _check_antibiotic_multi_intent(
    query: str,
    answer: str,
    issues: list[AnswerQualityIssue],
    required: list[str],
    detected: list[str],
) -> None:
    if not _is_antibiotic_monotherapy_and_bp_role_query(query):
        return

    required.extend(
        [
            "topical clindamycin monotherapy risk",
            "oral antibiotic monotherapy or prolonged-use risk",
            "benzoyl peroxide combination role",
        ]
    )

    if "clindamycin" in answer and _contains_any(
        answer,
        ["don doc", "don tri lieu", "khang khang sinh", "khong nen dung", "khong khuyen cao"],
    ):
        detected.append("topical clindamycin monotherapy risk")

    if _contains_any(answer, ["khang sinh uong", "khang sinh duong uong", "oral antibiotic", "doxycycline"]) and _contains_any(
        answer,
        ["don doc", "keo dai", "bac si", "ke don", "theo doi", "khong tu"],
    ):
        detected.append("oral antibiotic monotherapy or prolonged-use risk")

    if "benzoyl peroxide" in answer and _contains_any(
        answer,
        ["phoi hop", "ket hop", "giam nguy co khang khang sinh", "khang khang sinh", "tang hieu qua"],
    ):
        detected.append("benzoyl peroxide combination role")

    missing = [
        fact
        for fact in [
            "topical clindamycin monotherapy risk",
            "oral antibiotic monotherapy or prolonged-use risk",
            "benzoyl peroxide combination role",
        ]
        if fact not in detected
    ]
    if missing:
        issues.append(
            _issue(
                "antibiotic_multi_intent_incomplete",
                ERROR,
                "Answer collapses a multi-intent antibiotic question or misses a required sub-question.",
                evidence={"missing": missing},
            )
        )


_PREGNANCY_ENTITY_ALIASES: dict[str, list[str]] = {
    "adapalene": ["adapalene", "adapalen", "differin"],
    "tazarotene": ["tazarotene", "tazaroten", "tazorac"],
    "tretinoin": ["tretinoin"],
    "isotretinoin": ["isotretinoin"],
    "doxycycline": ["doxycycline", "doxycyclin"],
    "clindamycin": ["clindamycin", "dalacin"],
    "benzoyl_peroxide": ["benzoyl peroxide", "benzoyl peroxid", "bpo"],
}


def _check_multi_entity_pregnancy_coverage(
    query: str,
    answer: str,
    normalized_query: NormalizedQuery,
    issues: list[AnswerQualityIssue],
    required: list[str],
    detected: list[str],
) -> None:
    if not _is_pregnancy_context_query(query):
        return
    entities = [
        entity
        for entity in normalized_query.active_ingredient
        if entity in _PREGNANCY_ENTITY_ALIASES and _entity_mentioned(query, entity)
    ]
    if len(entities) < 2:
        return

    for entity in entities:
        fact = f"pregnancy safety covers {entity}"
        required.append(fact)
        if _entity_mentioned(answer, entity):
            detected.append(fact)
        else:
            issues.append(
                _issue(
                    "missing_safety_entity",
                    ERROR,
                    "Safety answer omits a medication entity mentioned by the user.",
                    evidence={"missing_entity": entity, "expected_entities": entities},
                )
            )

    if not _contains_any(answer, ["bac si", "da lieu", "san khoa", "doctor", "clinician", "khong tu", "ngung", "tranh"]):
        issues.append(
            _issue(
                "pregnancy_action_missing",
                ERROR,
                "Multi-medication pregnancy answer should advise clinician review and avoid self-directed continued use.",
            )
        )


def _check_self_harm_crisis_response(query: str, answer: str, issues: list[AnswerQualityIssue]) -> None:
    if not _is_self_harm_query(query):
        return

    has_emergency_action = _contains_any(answer, ["cap cuu", "co so y te khan cap", "y te khan cap", "emergency"])
    has_trusted_person = _contains_any(answer, ["nguoi dang tin cay", "o ben", "khong o mot minh", "khong nen o mot minh"])
    if not has_emergency_action:
        issues.append(_issue("self_harm_emergency_action_missing", ERROR, "Self-harm answer should mention emergency help if immediate danger is present."))
    if not has_trusted_person:
        issues.append(_issue("self_harm_trusted_person_missing", ERROR, "Self-harm answer should encourage contacting a trusted person and not being alone."))

    crisis_index = _first_index(answer, ["an toan", "cap cuu", "khan cap", "nguoi dang tin cay"])
    acne_index = _first_index(answer, ["tri mun", "dieu tri mun", "da lieu", "routine", "cham soc da"])
    if acne_index >= 0 and (crisis_index < 0 or acne_index < crisis_index):
        issues.append(_issue("self_harm_crisis_not_first", ERROR, "Self-harm answer should prioritize crisis escalation before dermatology advice."))
    if _contains_any(answer, ["ban bi tram cam", "chan doan tram cam", "roi loan tam than"]):
        issues.append(_issue("self_harm_diagnosis_overclaim", ERROR, "Self-harm answer should not diagnose mental health conditions via chat."))


def _check_acne_fulminans_urgency(query: str, answer: str, issues: list[AnswerQualityIssue]) -> None:
    if not _is_acne_fulminans_like_query(query):
        return
    if "acne fulminans" not in answer:
        issues.append(_issue("acne_fulminans_not_mentioned", ERROR, "Fulminans-like query should mention suspected acne fulminans."))
    if not _contains_any(answer, ["nghi", "co the", "khong the chan doan", "khong chan doan"]):
        issues.append(_issue("acne_fulminans_uncertainty_missing", ERROR, "Fulminans-like answer should avoid definitive diagnosis."))
    if not _contains_any(answer, ["trong ngay", "24 gio", "24h", "khan", "cap cuu"]):
        issues.append(_issue("acne_fulminans_urgency_missing", ERROR, "Fulminans-like answer should preserve same-day or 24-hour urgency."))
    if _contains_any(answer, ["chac chan la acne fulminans", "chinh la acne fulminans"]):
        issues.append(_issue("acne_fulminans_definitive_diagnosis", ERROR, "Answer should not diagnose acne fulminans definitively via chat."))


def _check_requested_table_schema(query: str, answer: str, issues: list[AnswerQualityIssue]) -> None:
    structure = parse_requested_structure(query)
    requested = list(structure.required_columns)
    if not structure.wants_table and not requested:
        return
    headers = _extract_table_headers(answer)
    if not headers:
        issues.append(
            _issue(
                "requested_table_missing",
                ERROR,
                "User requested a table with specific dimensions, but answer has no parseable Markdown table.",
                evidence={"requested_columns": requested, "requested_rows": list(structure.required_rows)},
            )
        )
        return

    missing = [
        column
        for column in requested
        if not any(_column_matches_header(column, header) for header in headers)
    ]
    if missing:
        issues.append(
            _issue(
                "requested_table_column_missing",
                ERROR,
                "Answer table is missing one or more user-requested columns/dimensions.",
                evidence={"missing_columns": missing, "headers": headers},
            )
        )

    if structure.exact_column_count and len(headers) != structure.exact_column_count:
        issues.append(
            _issue(
                "requested_table_column_count_mismatch",
                ERROR,
                "Answer table does not preserve the exact number of columns requested by the user.",
                evidence={"expected": structure.exact_column_count, "actual": len(headers), "headers": headers},
            )
        )

    body_text = _extract_table_body_text(answer)
    missing_rows = [
        row
        for row in structure.required_rows
        if not _row_matches_table_body(row, body_text)
    ]
    if missing_rows:
        issues.append(
            _issue(
                "requested_table_row_missing",
                ERROR,
                "Answer table is missing one or more user-requested entities/items.",
                evidence={"missing_rows": missing_rows},
            )
        )

    if missing and any(_column_matches_header(column, body_text) for column in missing):
        issues.append(
            _issue(
                "requested_table_orientation_inverted",
                ERROR,
                "Answer appears to place requested columns as row labels instead of table headers.",
                evidence={"missing_columns": missing, "headers": headers},
            )
        )


def _check_irrelevant_topical_warning(query: str, answer: str, issues: list[AnswerQualityIssue]) -> None:
    asks_oral_isotretinoin = "isotretinoin" in query and (
        _contains_any(query, ["duong uong", "uong", "oral", "lieu", "xet nghiem", "lipid", "gan", "tam than"])
        or not _contains_any(query, ["boi", "thuoc boi", "topical"])
    )
    asks_non_topical_safety = asks_oral_isotretinoin or _contains_any(
        query,
        [
            "bo qua huong dan",
            "ignore instructions",
            "ke lieu",
            "ke cho toi lieu",
            "tu uong",
            "uống",
            "can nang",
            "xet nghiem",
            "thai ky",
        ],
    )
    topical_context = _contains_any(query, ["boi", "thuoc boi"]) or any(
        _contains_entity_alias(query, alias)
        for alias in ["adapalene", "benzoyl peroxide", "clindamycin", "tretinoin", "tazarotene", "salicylic acid"]
    )
    topical_warning = (
        _contains_any(answer, ["giam tan suat", "tam ngung hoat chat", "tam ngung hoat chat de kich ung", "cham chich", "kich ung tai cho"])
        and _contains_any(answer, ["boi", "hoat chat de kich ung", "tai cho", "do rat", "kho bong"])
    )
    if asks_non_topical_safety and not topical_context and topical_warning:
        issues.append(
            _issue(
                "irrelevant_topical_warning",
                ERROR,
                "Answer injects topical-irritation frequency warnings into a non-topical safety/refusal context.",
            )
        )


def _issue(code: str, severity: str, message: str, evidence: dict[str, Any] | None = None, suggested_fix: str | None = None) -> AnswerQualityIssue:
    return AnswerQualityIssue(
        code=code,
        severity=severity,  # type: ignore[arg-type]
        message=message,
        evidence=evidence or {},
        suggested_fix=suggested_fix,
    )


def _short_warning(report: AnswerVerificationReport) -> str:
    critical_or_error = [issue for issue in report.issues if issue.severity in {CRITICAL, ERROR}]
    if critical_or_error:
        return "Câu trả lời cần được kiểm tra lại vì có thể thiếu hoặc mâu thuẫn với thông tin an toàn quan trọng."
    if report.safety_warnings or report.issues:
        return "Không tự dùng thuốc kê đơn hoặc hoạt chất nguy cơ cao; nên hỏi bác sĩ da liễu nếu cần điều trị thuốc."
    return ""


def _strict_safe_answer(query: str, report: AnswerVerificationReport) -> str:
    facts = report.required_facts[:3] or ["Cần kiểm tra lại thông tin y khoa trước khi trả lời."]
    return (
        "Tôi cần hiệu chỉnh câu trả lời vì phát hiện mâu thuẫn y khoa quan trọng.\n\n"
        "## Thông tin an toàn cần giữ\n"
        + "\n".join(f"- {fact}" for fact in facts)
        + "\n\nThông tin này chỉ mang tính tham khảo và không thay thế tư vấn của bác sĩ da liễu."
    )


def _check_structural_presentation(query: str, answer: str, issues: list[AnswerQualityIssue]) -> None:
    profile = infer_response_profile(query)
    for issue_data in assess_structural_quality(
        answer,
        user_question=query,
        response_profile=profile,
    ):
        severity = issue_data.get("severity", WARNING)
        issues.append(
            _issue(
                str(issue_data.get("code") or "presentation_contract_violation"),
                ERROR if severity == "error" else WARNING,
                str(issue_data.get("message") or "Answer violates presentation contract."),
                evidence=issue_data.get("evidence") if isinstance(issue_data.get("evidence"), dict) else {},
            )
        )

    structure = parse_requested_structure(query)
    answer_norm = _norm(answer)
    if structure.semantic_intent == "signs_symptoms" and _contains_any(
        answer_norm,
        ["do thoi quen", "thoi quen", "an do ngot", "stress", "thuc khuya", "my pham gay bit tac", "nguyen nhan"],
    ):
        issues.append(
            _issue(
                "sign_symptom_answer_contains_causes",
                ERROR,
                "User asked for signs/symptoms, but the answer drifts into causes or behaviors.",
            )
        )
    if structure.exact_item_count:
        item_count = _count_markdown_items(answer)
        if item_count != structure.exact_item_count:
            issues.append(
                _issue(
                    "requested_item_count_mismatch",
                    ERROR,
                    "Answer does not preserve the exact number of list items requested by the user.",
                    evidence={"expected": structure.exact_item_count, "actual": item_count},
                )
            )
    if "bold_headings" in structure.style_constraints and not _has_bold_or_markdown_heading(answer):
        issues.append(
            _issue(
                "requested_bold_heading_missing",
                ERROR,
                "User requested bold headings, but the answer does not contain Markdown bold headings or headings.",
            )
        )


def _required_fact_is_important(fact: str, intent: str | None) -> bool:
    if intent in {"drug_identity", "ingredient_question", "class_check"}:
        return True
    return any(term in fact for term in ["not an antibiotic", "not a retinoid", "acne type"])


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _is_antibiotic_monotherapy_and_bp_role_query(query: str) -> bool:
    topical = _contains_any(query, ["clindamycin", "erythromycin", "khang sinh boi"])
    oral = _contains_any(
        query,
        ["khang sinh uong", "khang sinh duong uong", "oral antibiotic", "doxycycline", "lymecycline", "minocycline"],
    )
    monotherapy_or_long = _contains_any(query, ["don doc", "don tri lieu", "monotherapy", "keo dai"])
    bp_role = "benzoyl peroxide" in query and _contains_any(query, ["phoi hop", "ket hop", "vai tro"])
    return bp_role and monotherapy_or_long and (topical or oral)


def _is_pregnancy_context_query(query: str) -> bool:
    return _contains_any(
        query,
        [
            "mang thai",
            "co thai",
            "dang co thai",
            "co bau",
            "dang bau",
            "thai ky",
            "pregnancy",
            "pregnant",
            "cho con bu",
        ],
    )


def _entity_mentioned(text: str, entity: str) -> bool:
    aliases = _PREGNANCY_ENTITY_ALIASES.get(entity, [entity])
    return any(_contains_entity_alias(text, alias) for alias in aliases)


def _contains_entity_alias(text: str, alias: str) -> bool:
    alias_norm = _norm(alias)
    if not alias_norm:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])", text) is not None


def _is_self_harm_query(query: str) -> bool:
    return _contains_any(
        query,
        ["tu lam hai", "tu hai", "lam hai ban than", "hai ban than", "tu sat", "self harm", "suicide"],
    )


def _is_acne_fulminans_like_query(query: str) -> bool:
    severe_lesions = _contains_any(query, ["cuc", "nang viem", "mun nang", "mun cuc"])
    erosive = _contains_any(query, ["trot loet", "loet", "vay xuat huyet", "dong vay xuat huyet"])
    systemic = _contains_any(query, ["sot", "dau khop", "dot ngot"])
    return severe_lesions and erosive and systemic


def _first_index(text: str, needles: list[str]) -> int:
    positions = [text.find(needle) for needle in needles if needle in text]
    return min(positions) if positions else -1


def _extract_requested_table_columns(query: str) -> list[str]:
    return list(parse_requested_structure(query).required_columns)


def _extract_table_headers(answer: str) -> list[str]:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    for index in range(len(lines) - 1):
        if _is_table_row(lines[index]) and _is_table_separator(lines[index + 1]):
            return [_norm(cell) for cell in _split_table_row(lines[index]) if cell.strip()]
    return []


def _is_table_row(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|") and line.count("|") >= 2


def _is_table_separator(line: str) -> bool:
    if not _is_table_row(line):
        return False
    cells = _split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _column_matches_header(column: str, header: str) -> bool:
    column = canonical_column_name(column.strip())
    header = header.strip()
    if column in header or header in column:
        return True
    synonyms = {
        "thuoc": ["thuoc", "lua chon", "hoat chat"],
        "duong dung": ["duong dung", "cach dung", "dang dung"],
        "uu diem": ["uu diem", "loi ich", "diem manh"],
        "luu y an toan": ["luu y an toan", "an toan", "luu y"],
    }
    for canonical, values in synonyms.items():
        if column == canonical or column in values:
            return any(value in header for value in values)
    return False


def _extract_table_body_text(answer: str) -> str:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    body_rows: list[str] = []
    in_table = False
    for index in range(len(lines)):
        if index + 1 < len(lines) and _is_table_row(lines[index]) and _is_table_separator(lines[index + 1]):
            in_table = True
            continue
        if in_table and _is_table_separator(lines[index]):
            continue
        if in_table and _is_table_row(lines[index]):
            body_rows.extend(_split_table_row(lines[index]))
        elif in_table:
            break
    return _norm(" ".join(body_rows))


def _row_matches_table_body(row: str, body_text: str) -> bool:
    row_norm = _norm(row)
    aliases = {
        "benzoyl peroxide": ["benzoyl peroxide", "bpo", "bp"],
        "salicylic acid": ["salicylic acid", "acid salicylic", "bha"],
        "Epiduo": ["epiduo"],
        "Differin": ["differin"],
        "Tazorac": ["tazorac", "tazarotene"],
    }.get(row, [row_norm])
    return any(_norm(alias) in body_text for alias in aliases)


def _count_markdown_items(answer: str) -> int:
    return sum(1 for line in answer.splitlines() if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+\S", line))


def _has_bold_or_markdown_heading(answer: str) -> bool:
    return bool(re.search(r"^\s*(?:#{1,4}\s+\S|\*\*[^*\n]{2,80}\*\*)", answer, flags=re.MULTILINE))


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _norm(text: str) -> str:
    _, accentless = build_matching_views(text)
    return accentless


def _has_prop(propositions: list[DomainProposition], subject: str, relation: str, object_: str) -> bool:
    return proposition_exists(propositions, subject=subject, relation=relation, object_=object_)


def _evidence_for(propositions: list[DomainProposition], subject: str, relation: str, object_: str) -> dict[str, Any]:
    for proposition in propositions:
        if proposition_exists([proposition], subject=subject, relation=relation, object_=object_):
            return {
                "matched_subject": proposition.subject,
                "matched_predicate": proposition.object,
                "matched_clause": proposition.matched_text[:200],
                "source_rule": proposition.source_rule,
                "normalized_clause": proposition.normalized_text[:200],
            }
    return {}


def _proposition_rule_counts(propositions: list[DomainProposition]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for proposition in propositions:
        counts[proposition.source_rule] = counts.get(proposition.source_rule, 0) + 1
    return counts


__all__ = ["apply_answer_guard", "verify_answer_quality"]
