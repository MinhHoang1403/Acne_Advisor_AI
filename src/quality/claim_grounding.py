"""Deterministic P4 claim-level grounding contracts and shadow verifier."""

from __future__ import annotations

import hashlib
import os
import re
import time
from enum import Enum
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.quality.proposition_detector import extract_domain_propositions
from src.quality.severity_guard import classify_medical_severity
from src.quality.vietnamese_text import build_matching_views
from src.retrieval.contracts import PackedContext
from src.retrieval.query_normalization import normalize_query


P4_CLAIM_GROUNDING_VERSION = "claim_level_grounding_v1"
P4_CLAIM_SCHEMA_VERSION = "answer_claim_v1"
P4_EVIDENCE_MAPPING_VERSION = "claim_evidence_mapping_v1"
P4_ENTAILMENT_VERSION = "deterministic_entailment_v1"
P4_CRITICAL_POLICY_VERSION = "critical_claim_policy_v1"
P4_MAX_CLAIMS = 16
P4_MAX_EVIDENCE_PER_CLAIM = 3
P4_TRACE_EVENT_LIMIT = 96


class P4Mode(str, Enum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    ENFORCE_CRITICAL = "enforce_critical"
    ENFORCE_ALL = "enforce_all"


class ClaimType(str, Enum):
    DEFINITION = "DEFINITION"
    MECHANISM = "MECHANISM"
    CAUSE_OR_ASSOCIATION = "CAUSE_OR_ASSOCIATION"
    TREATMENT = "TREATMENT"
    SAFETY = "SAFETY"
    CONTRAINDICATION = "CONTRAINDICATION"
    DOSING_OR_USE = "DOSING_OR_USE"
    COMPARISON = "COMPARISON"
    PROGNOSIS = "PROGNOSIS"
    SOURCE_ATTRIBUTION = "SOURCE_ATTRIBUTION"
    OTHER_MEDICAL_FACT = "OTHER_MEDICAL_FACT"


class ClaimCriticality(str, Enum):
    NON_CRITICAL = "NON_CRITICAL"
    CRITICAL = "CRITICAL"


class EvidenceScope(str, Enum):
    GENERATION_CONTEXT_EVIDENCE = "GENERATION_CONTEXT_EVIDENCE"
    RETRIEVED_BUT_NOT_PROMPTED_EVIDENCE = "RETRIEVED_BUT_NOT_PROMPTED_EVIDENCE"


class EntailmentStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NO_EVIDENCE = "NO_EVIDENCE"
    VERIFIER_ERROR = "VERIFIER_ERROR"


class ShadowPolicyAction(str, Enum):
    WOULD_ALLOW = "WOULD_ALLOW"
    WOULD_REWRITE_PARTIAL = "WOULD_REWRITE_PARTIAL"
    WOULD_DROP_NONCRITICAL = "WOULD_DROP_NONCRITICAL"
    WOULD_BLOCK_CRITICAL = "WOULD_BLOCK_CRITICAL"
    WOULD_ABSTAIN = "WOULD_ABSTAIN"
    VERIFIER_UNAVAILABLE = "VERIFIER_UNAVAILABLE"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AnswerClaim(_FrozenModel):
    schema_version: str = P4_CLAIM_SCHEMA_VERSION
    claim_id: str
    text: str
    normalized_text: str
    sentence_index: int
    claim_index: int
    claim_type: ClaimType
    criticality: ClaimCriticality
    source_requirement: str = "SOURCE_BACKED_CHUNK"
    candidate_evidence_ids: tuple[str, ...] = ()
    mapped_evidence_ids: tuple[str, ...] = ()
    mapped_source_ids: tuple[str, ...] = ()


class ClaimEvidenceLink(_FrozenModel):
    schema_version: str = P4_EVIDENCE_MAPPING_VERSION
    claim_id: str
    evidence_id: str
    source_id: str
    scope: EvidenceScope = EvidenceScope.GENERATION_CONTEXT_EVIDENCE
    mapping_reason: str
    lexical_overlap: float = 0.0
    entity_overlap: tuple[str, ...] = ()
    semantic_score: float | None = None
    provenance_valid: bool
    rank: int

    @field_validator("lexical_overlap")
    @classmethod
    def _bounded_overlap(cls, value: float) -> float:
        return min(1.0, max(0.0, float(value)))


class ClaimEntailmentVerdict(_FrozenModel):
    schema_version: str = P4_ENTAILMENT_VERSION
    claim_id: str
    verdict: EntailmentStatus
    verifier_confidence: float | None = None
    evidence_ids_used: tuple[str, ...] = ()
    contradiction_evidence_ids: tuple[str, ...] = ()
    reason_code: str
    verifier: str = "deterministic"
    verifier_model: str | None = None


class VerifiedClaimSet(_FrozenModel):
    supported_claims: tuple[str, ...] = ()
    partially_supported_claims: tuple[str, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    contradicted_claims: tuple[str, ...] = ()
    no_evidence_claims: tuple[str, ...] = ()
    verifier_error_claims: tuple[str, ...] = ()
    critical_failures: tuple[str, ...] = ()
    source_mapping: dict[str, tuple[str, ...]] = Field(default_factory=dict)


class P4TraceEvent(_FrozenModel):
    event: str
    claim_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    verdict: EntailmentStatus | None = None
    elapsed_ms: float | None = None
    reason_code: str | None = None
    provider: str = "deterministic"
    model: str | None = None
    schema_version: str = P4_ENTAILMENT_VERSION


class ClaimGroundingResult(_FrozenModel):
    version: str = P4_CLAIM_GROUNDING_VERSION
    mode: P4Mode
    status: str
    claims: tuple[AnswerClaim, ...] = ()
    evidence_links: tuple[ClaimEvidenceLink, ...] = ()
    verdicts: tuple[ClaimEntailmentVerdict, ...] = ()
    verified_claims: VerifiedClaimSet = Field(default_factory=VerifiedClaimSet)
    shadow_action: ShadowPolicyAction = ShadowPolicyAction.WOULD_ALLOW
    shadow_verified_answer: str = ""
    production_answer_modified: bool = False
    degraded: bool = False
    degraded_reason: str | None = None
    extraction_truncated: bool = False
    critical_claim_overflow: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)
    trace: tuple[P4TraceEvent, ...] = ()


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "this", "to", "with", "va", "la", "co", "cua",
    "cho", "trong", "mot", "nhung", "duoc", "khi", "voi", "nay", "do", "cac", "giup", "the",
}
_BOILERPLATE_MARKERS = (
    "thong tin nay chi mang tinh tham khao",
    "khong thay the tu van",
    "toi co the giup gi",
    "duoi day la",
)
_MEDICAL_MARKERS = (
    "acne", "mun", "da", "thuoc", "retinoid", "antibiotic", "khang sinh", "adapalene",
    "tazarotene", "tazorac", "benzoyl", "clindamycin", "erythromycin", "isotretinoin",
    "thai", "pregnan", "kich ung", "viem", "nang long", "lo chan long", "ba nhon", "bacteria",
    "c. acnes", "c acnes", "dieu tri", "chong chi dinh", "cap cuu", "safety", "an toan",
)
_CRITICAL_MARKERS = (
    "mang thai", "thai ky", "co thai", "co bau", "cho con bu", "pregnan", "contraindicat",
    "chong chi dinh", "khang sinh", "antibiotic", "isotretinoin", "tu tu", "tu sat", "cap cuu",
    "kho tho", "sung luoi", "sung hong", "phan ve", "hoai tu", "phong rop", "loet niem mac",
    "khong nen dung", "khong duoc dung", "nguy hiem", "dangerous",
)
_PARTIAL_MARKERS = (
    "luon luon", "hoan toan", "chac chan", "100%", "trong mot tuan", "trong 1 tuan",
    "ngay lap tuc", "never", "always", "completely", "within one week", "guarantee",
)
_NEGATION_MARKERS = (" khong ", " chang ", " chua ", " not ", " never ", " does not ", " do not ")


def p4_mode_from_env(value: str | None = None) -> P4Mode:
    configured = str(value if value is not None else os.getenv("P4_MODE", "shadow")).strip().lower()
    try:
        return P4Mode(configured)
    except ValueError:
        return P4Mode.SHADOW


def p4_effective_mode(
    requested: P4Mode | str,
    *,
    critical_enforcement_ready: bool = False,
    all_claim_enforcement_ready: bool = False,
) -> P4Mode:
    """Apply calibration interlocks before an enforcement mode can serve."""

    mode = P4Mode(requested)
    if mode == P4Mode.ENFORCE_CRITICAL and not critical_enforcement_ready:
        return P4Mode.SHADOW
    if mode == P4Mode.ENFORCE_ALL and not all_claim_enforcement_ready:
        return P4Mode.SHADOW
    return mode


def p4_max_claims_from_env() -> int:
    return _bounded_env_int("P4_MAX_CLAIMS", P4_MAX_CLAIMS, minimum=1, maximum=32)


def p4_max_evidence_from_env() -> int:
    return _bounded_env_int(
        "P4_MAX_EVIDENCE_PER_CLAIM",
        P4_MAX_EVIDENCE_PER_CLAIM,
        minimum=1,
        maximum=5,
    )


def extract_answer_claims(
    answer: str,
    *,
    query: str = "",
    max_claims: int = P4_MAX_CLAIMS,
) -> tuple[tuple[AnswerClaim, ...], dict[str, bool]]:
    """Extract bounded inspectable propositions while retaining critical claims."""

    candidates = _claim_candidates(answer)
    extracted: list[AnswerClaim] = []
    for sentence_index, text in enumerate(candidates):
        for proposition in _split_conjoined_medical_claims(text):
            clean = _clean_claim_text(proposition)
            if not _is_claim_worthy(clean):
                continue
            normalized = _normalize(clean)
            criticality = _claim_criticality(clean, query=query)
            claim_index = len(extracted)
            extracted.append(
                AnswerClaim(
                    claim_id=_claim_id(clean, sentence_index, claim_index),
                    text=clean,
                    normalized_text=normalized,
                    sentence_index=sentence_index,
                    claim_index=claim_index,
                    claim_type=_claim_type(clean),
                    criticality=criticality,
                )
            )

    critical = [claim for claim in extracted if claim.criticality == ClaimCriticality.CRITICAL]
    noncritical = [claim for claim in extracted if claim.criticality == ClaimCriticality.NON_CRITICAL]
    truncated = len(extracted) > max_claims
    critical_overflow = len(critical) > max_claims
    if truncated:
        kept = critical + noncritical[: max(0, max_claims - len(critical))]
        kept.sort(key=lambda item: item.claim_index)
    else:
        kept = extracted
    return tuple(kept), {
        "extraction_truncated": truncated,
        "critical_claim_overflow": critical_overflow,
    }


def map_claims_to_evidence(
    claims: Iterable[AnswerClaim],
    packed_context: PackedContext | dict[str, Any] | None,
    *,
    max_evidence_per_claim: int = P4_MAX_EVIDENCE_PER_CLAIM,
) -> tuple[tuple[AnswerClaim, ...], tuple[ClaimEvidenceLink, ...]]:
    """Map claims only to packed source-backed chunks with valid provenance."""

    context = _packed_context(packed_context)
    evidence_records = _valid_generation_evidence(context)
    mapped_claims: list[AnswerClaim] = []
    all_links: list[ClaimEvidenceLink] = []
    for claim in claims:
        ranked: list[tuple[float, ClaimEvidenceLink]] = []
        claim_tokens = _content_tokens(claim.text)
        claim_entities = _normalized_entities(claim.text)
        for evidence_rank, record in enumerate(evidence_records, start=1):
            text = record["text"]
            evidence_tokens = _content_tokens(text)
            lexical = _coverage(claim_tokens, evidence_tokens)
            entities = tuple(sorted(claim_entities & _normalized_entities(text)))
            citation = _citation_match(claim.text, record["evidence_id"], record["source_id"])
            if citation:
                reason = "DIRECT_CITATION"
                mapping_score = 2.0 + lexical
            elif entities:
                reason = "ENTITY_AND_LEXICAL_OVERLAP" if lexical > 0 else "ENTITY_ALIAS_OVERLAP"
                mapping_score = 1.0 + lexical + min(0.3, len(entities) * 0.1)
            elif lexical >= 0.2:
                reason = "LEXICAL_OVERLAP"
                mapping_score = lexical
            else:
                continue
            link = ClaimEvidenceLink(
                claim_id=claim.claim_id,
                evidence_id=record["evidence_id"],
                source_id=record["source_id"],
                mapping_reason=reason,
                lexical_overlap=round(lexical, 6),
                entity_overlap=entities,
                semantic_score=None,
                provenance_valid=True,
                rank=evidence_rank,
            )
            ranked.append((mapping_score, link))
        selected = [item[1] for item in sorted(ranked, key=lambda item: (-item[0], item[1].rank))]
        selected = selected[:max_evidence_per_claim]
        all_links.extend(selected)
        mapped_claims.append(
            claim.model_copy(
                update={
                    "candidate_evidence_ids": tuple(link.evidence_id for _, link in ranked),
                    "mapped_evidence_ids": tuple(link.evidence_id for link in selected),
                    "mapped_source_ids": tuple(dict.fromkeys(link.source_id for link in selected)),
                }
            )
        )
    return tuple(mapped_claims), tuple(all_links)


def verify_claim_entailment(
    claim: AnswerClaim,
    links: Iterable[ClaimEvidenceLink],
    packed_context: PackedContext | dict[str, Any] | None,
) -> ClaimEntailmentVerdict:
    """Judge only whether mapped generation evidence entails one claim."""

    claim_links = tuple(link for link in links if link.claim_id == claim.claim_id and link.provenance_valid)
    if not claim_links:
        return ClaimEntailmentVerdict(
            claim_id=claim.claim_id,
            verdict=EntailmentStatus.NO_EVIDENCE,
            reason_code="NO_RELEVANT_EVIDENCE",
        )
    try:
        context = _packed_context(packed_context)
        evidence_by_id = {record["evidence_id"]: record["text"] for record in _valid_generation_evidence(context)}
        evidence_ids = tuple(link.evidence_id for link in claim_links if link.evidence_id in evidence_by_id)
        if not evidence_ids:
            return ClaimEntailmentVerdict(
                claim_id=claim.claim_id,
                verdict=EntailmentStatus.NO_EVIDENCE,
                reason_code="NO_RELEVANT_EVIDENCE",
            )
        evidence_text = "\n".join(evidence_by_id[evidence_id] for evidence_id in evidence_ids)
        verdict, confidence, reason, contradiction = _deterministic_entailment(claim.text, evidence_text)
        return ClaimEntailmentVerdict(
            claim_id=claim.claim_id,
            verdict=verdict,
            verifier_confidence=round(confidence, 6),
            evidence_ids_used=evidence_ids,
            contradiction_evidence_ids=evidence_ids if contradiction else (),
            reason_code=reason,
        )
    except Exception:
        return ClaimEntailmentVerdict(
            claim_id=claim.claim_id,
            verdict=EntailmentStatus.VERIFIER_ERROR,
            evidence_ids_used=tuple(link.evidence_id for link in claim_links),
            reason_code="INVALID_VERIFIER_OUTPUT",
        )


def evaluate_claim_grounding(
    *,
    answer: str,
    query: str,
    packed_context: PackedContext | dict[str, Any] | None,
    mode: P4Mode | str = P4Mode.SHADOW,
    p3_status: str | None = None,
    max_claims: int | None = None,
    max_evidence_per_claim: int | None = None,
) -> ClaimGroundingResult:
    """Run deterministic P4 diagnostics without using outside medical knowledge."""

    started = time.perf_counter()
    effective_mode = P4Mode(mode)
    if effective_mode == P4Mode.DISABLED:
        return ClaimGroundingResult(mode=effective_mode, status="disabled")
    if p3_status and p3_status != "SUFFICIENT":
        return ClaimGroundingResult(
            mode=effective_mode,
            status="skipped_p3_precedence",
            degraded=True,
            degraded_reason=f"P3_{p3_status}",
            shadow_action=ShadowPolicyAction.WOULD_ABSTAIN,
        )

    trace: list[P4TraceEvent] = [P4TraceEvent(event="CLAIM_EXTRACTION_STARTED")]
    extraction_started = time.perf_counter()
    claims, extraction_flags = extract_answer_claims(
        answer,
        query=query,
        max_claims=max_claims or p4_max_claims_from_env(),
    )
    extraction_ms = _elapsed_ms(extraction_started)
    trace.append(P4TraceEvent(event="CLAIM_EXTRACTION_COMPLETED", elapsed_ms=extraction_ms))

    mapping_started = time.perf_counter()
    mapped_claims, links = map_claims_to_evidence(
        claims,
        packed_context,
        max_evidence_per_claim=max_evidence_per_claim or p4_max_evidence_from_env(),
    )
    mapping_ms = _elapsed_ms(mapping_started)
    for claim in mapped_claims:
        trace.append(
            P4TraceEvent(
                event="CLAIM_MAPPED",
                claim_id=claim.claim_id,
                evidence_ids=claim.mapped_evidence_ids,
                elapsed_ms=mapping_ms,
            )
        )

    verifier_started = time.perf_counter()
    verdicts: list[ClaimEntailmentVerdict] = []
    for claim in mapped_claims:
        trace.append(P4TraceEvent(event="CLAIM_VERIFICATION_STARTED", claim_id=claim.claim_id))
        verdict = verify_claim_entailment(claim, links, packed_context)
        verdicts.append(verdict)
        trace.append(
            P4TraceEvent(
                event="CLAIM_VERIFIED" if verdict.verdict != EntailmentStatus.VERIFIER_ERROR else "CLAIM_VERIFIER_FAILED",
                claim_id=claim.claim_id,
                evidence_ids=verdict.evidence_ids_used,
                verdict=verdict.verdict,
                reason_code=verdict.reason_code,
            )
        )
        if claim.criticality == ClaimCriticality.CRITICAL and verdict.verdict != EntailmentStatus.SUPPORTED:
            trace.append(
                P4TraceEvent(
                    event="CRITICAL_CLAIM_FLAGGED",
                    claim_id=claim.claim_id,
                    verdict=verdict.verdict,
                    reason_code=verdict.reason_code,
                )
            )
    verifier_ms = _elapsed_ms(verifier_started)
    verified = build_verified_claim_set(mapped_claims, verdicts)
    shadow_action = evaluate_shadow_policy(mapped_claims, verdicts)
    trace.append(P4TraceEvent(event="SHADOW_POLICY_EVALUATED", reason_code=shadow_action.value))
    projection = build_shadow_verified_answer(mapped_claims, verdicts)
    degraded = any(verdict.verdict == EntailmentStatus.VERIFIER_ERROR for verdict in verdicts)
    status = "degraded" if degraded else "completed"
    timings = {
        "claim_extraction": extraction_ms,
        "evidence_mapping": mapping_ms,
        "entailment_verifier": verifier_ms,
        "total_p4": _elapsed_ms(started),
    }
    return ClaimGroundingResult(
        mode=effective_mode,
        status=status,
        claims=mapped_claims,
        evidence_links=links,
        verdicts=tuple(verdicts),
        verified_claims=verified,
        shadow_action=shadow_action,
        shadow_verified_answer=projection,
        production_answer_modified=False,
        degraded=degraded,
        degraded_reason="VERIFIER_ERROR" if degraded else None,
        extraction_truncated=extraction_flags["extraction_truncated"],
        critical_claim_overflow=extraction_flags["critical_claim_overflow"],
        timings_ms=timings,
        trace=tuple(trace[:P4_TRACE_EVENT_LIMIT]),
    )


def build_verified_claim_set(
    claims: Iterable[AnswerClaim],
    verdicts: Iterable[ClaimEntailmentVerdict],
) -> VerifiedClaimSet:
    claim_by_id = {claim.claim_id: claim for claim in claims}
    groups: dict[EntailmentStatus, list[str]] = {status: [] for status in EntailmentStatus}
    source_mapping: dict[str, tuple[str, ...]] = {}
    critical_failures: list[str] = []
    for verdict in verdicts:
        groups[verdict.verdict].append(verdict.claim_id)
        source_mapping[verdict.claim_id] = verdict.evidence_ids_used
        claim = claim_by_id.get(verdict.claim_id)
        if claim and claim.criticality == ClaimCriticality.CRITICAL and verdict.verdict != EntailmentStatus.SUPPORTED:
            critical_failures.append(verdict.claim_id)
    return VerifiedClaimSet(
        supported_claims=tuple(groups[EntailmentStatus.SUPPORTED]),
        partially_supported_claims=tuple(groups[EntailmentStatus.PARTIALLY_SUPPORTED]),
        unsupported_claims=tuple(groups[EntailmentStatus.UNSUPPORTED]),
        contradicted_claims=tuple(groups[EntailmentStatus.CONTRADICTED]),
        no_evidence_claims=tuple(groups[EntailmentStatus.NO_EVIDENCE]),
        verifier_error_claims=tuple(groups[EntailmentStatus.VERIFIER_ERROR]),
        critical_failures=tuple(critical_failures),
        source_mapping=source_mapping,
    )


def evaluate_shadow_policy(
    claims: Iterable[AnswerClaim],
    verdicts: Iterable[ClaimEntailmentVerdict],
) -> ShadowPolicyAction:
    claim_by_id = {claim.claim_id: claim for claim in claims}
    verdict_list = tuple(verdicts)
    if any(
        claim_by_id.get(item.claim_id)
        and claim_by_id[item.claim_id].criticality == ClaimCriticality.CRITICAL
        and item.verdict != EntailmentStatus.SUPPORTED
        for item in verdict_list
    ):
        return ShadowPolicyAction.WOULD_BLOCK_CRITICAL
    if any(item.verdict == EntailmentStatus.VERIFIER_ERROR for item in verdict_list):
        return ShadowPolicyAction.VERIFIER_UNAVAILABLE
    if verdict_list and all(item.verdict in {EntailmentStatus.UNSUPPORTED, EntailmentStatus.NO_EVIDENCE} for item in verdict_list):
        return ShadowPolicyAction.WOULD_ABSTAIN
    if any(item.verdict == EntailmentStatus.PARTIALLY_SUPPORTED for item in verdict_list):
        return ShadowPolicyAction.WOULD_REWRITE_PARTIAL
    if any(item.verdict in {EntailmentStatus.UNSUPPORTED, EntailmentStatus.CONTRADICTED, EntailmentStatus.NO_EVIDENCE} for item in verdict_list):
        return ShadowPolicyAction.WOULD_DROP_NONCRITICAL
    return ShadowPolicyAction.WOULD_ALLOW


def build_shadow_verified_answer(
    claims: Iterable[AnswerClaim],
    verdicts: Iterable[ClaimEntailmentVerdict],
) -> str:
    verdict_by_id = {item.claim_id: item for item in verdicts}
    return "\n\n".join(
        claim.text
        for claim in claims
        if verdict_by_id.get(claim.claim_id)
        and verdict_by_id[claim.claim_id].verdict == EntailmentStatus.SUPPORTED
    )


def compact_claim_grounding(result: ClaimGroundingResult | dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    if isinstance(result, dict) and "claim_count" in result and "claims" not in result:
        allowed = {
            "version",
            "mode",
            "status",
            "claim_count",
            "supported_count",
            "partial_count",
            "unsupported_count",
            "contradicted_count",
            "no_evidence_count",
            "verifier_error_count",
            "critical_failures",
            "shadow_action",
            "verifier_degraded",
            "production_answer_modified",
            "timings_ms",
        }
        return {key: value for key, value in result.items() if key in allowed}
    parsed = result if isinstance(result, ClaimGroundingResult) else ClaimGroundingResult.model_validate(result)
    counts = {status.value: 0 for status in EntailmentStatus}
    for verdict in parsed.verdicts:
        counts[verdict.verdict.value] += 1
    return {
        "version": parsed.version,
        "mode": parsed.mode.value,
        "status": parsed.status,
        "claim_count": len(parsed.claims),
        "supported_count": counts[EntailmentStatus.SUPPORTED.value],
        "partial_count": counts[EntailmentStatus.PARTIALLY_SUPPORTED.value],
        "unsupported_count": counts[EntailmentStatus.UNSUPPORTED.value],
        "contradicted_count": counts[EntailmentStatus.CONTRADICTED.value],
        "no_evidence_count": counts[EntailmentStatus.NO_EVIDENCE.value],
        "verifier_error_count": counts[EntailmentStatus.VERIFIER_ERROR.value],
        "critical_failures": len(parsed.verified_claims.critical_failures),
        "shadow_action": parsed.shadow_action.value,
        "verifier_degraded": parsed.degraded,
        "production_answer_modified": parsed.production_answer_modified,
        "timings_ms": dict(parsed.timings_ms),
    }


def _claim_candidates(answer: str) -> list[str]:
    lines = str(answer or "").replace("\r\n", "\n").split("\n")
    table_headers: list[str] | None = None
    candidates: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if re.fullmatch(r"\|?[\s:|-]+\|?", line):
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if table_headers is None:
                table_headers = cells
                continue
            candidates.append("; ".join(f"{header}: {cell}" for header, cell in zip(table_headers, cells) if cell))
            continue
        table_headers = None
        if re.match(r"^#{1,6}\s+", line) or re.fullmatch(r"\*\*[^*]+\*\*:?", line):
            continue
        line = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", line)
        protected = re.sub(r"\b([A-Za-z])\.\s+(?=[a-z])", r"\1<abbr> ", line)
        candidates.extend(
            part.replace("<abbr>", ".").strip()
            for part in re.split(r"(?<=[.!?])\s+", protected)
            if part.strip()
        )
    return candidates


def _clean_claim_text(text: str) -> str:
    value = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    value = re.sub(r"[`*_]", "", value)
    value = re.sub(r"\s+", " ", value).strip(" -|;:")
    return value


def _split_conjoined_medical_claims(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\s*\b(?:và|nhưng|and|but)\b\s*", text) if part.strip()]
    if len(parts) < 2:
        return [text]
    medical_parts = [
        part
        for part in parts
        if len(_content_tokens(part)) >= 2
        and _normalized_entities(part)
        and any(marker in _normalize(part) for marker in _MEDICAL_MARKERS)
    ]
    return parts if len(medical_parts) == len(parts) else [text]


def _is_claim_worthy(text: str) -> bool:
    normalized = _normalize(text)
    token_count = len(_content_tokens(text))
    if token_count < 2:
        return False
    if token_count < 3 and not extract_domain_propositions(text):
        return False
    if any(marker in normalized for marker in _BOILERPLATE_MARKERS):
        return False
    return any(marker in normalized for marker in _MEDICAL_MARKERS)


def _claim_criticality(text: str, *, query: str) -> ClaimCriticality:
    combined = _normalize(f"{query} {text}")
    classification = classify_medical_severity(f"{query} {text}")
    stewardship_identity = any(
        drug in combined for drug in ("clindamycin", "erythromycin", "doxycycline", "minocycline")
    ) and any(
        marker in combined
        for marker in ("khang sinh", "antibiotic", "retinoid", "don doc", "keo dai", "an toan")
    )
    if (
        classification.severity in {"urgent", "emergency"}
        or stewardship_identity
        or any(marker in combined for marker in _CRITICAL_MARKERS)
    ):
        return ClaimCriticality.CRITICAL
    return ClaimCriticality.NON_CRITICAL


def _claim_type(text: str) -> ClaimType:
    normalized = _normalize(text)
    if any(term in normalized for term in ("nguon", "tai lieu", "source", "theo ")):
        return ClaimType.SOURCE_ATTRIBUTION
    if any(term in normalized for term in ("khong duoc", "chong chi dinh", "contraindicat")):
        return ClaimType.CONTRAINDICATION
    if any(term in normalized for term in ("mang thai", "an toan", "nguy co", "tac dung phu", "cap cuu", "safety")):
        return ClaimType.SAFETY
    if any(term in normalized for term in ("la mot", "thuoc nhom", "thuoc loai", "is a", "khong phai")):
        return ClaimType.DEFINITION
    if any(term in normalized for term in ("lieu", "tan suat", "moi ngay", "boi", "uong", "use")):
        return ClaimType.DOSING_OR_USE
    if any(term in normalized for term in ("khac", "so voi", "giong", "comparison")):
        return ClaimType.COMPARISON
    if any(term in normalized for term in ("co che", "hoat dong", "oxy hoa", "tac dong", "mechanism")):
        return ClaimType.MECHANISM
    if any(term in normalized for term in ("gay", "lien quan", "do ", "association")):
        return ClaimType.CAUSE_OR_ASSOCIATION
    if any(term in normalized for term in ("dieu tri", "giam mun", "hieu qua", "treatment")):
        return ClaimType.TREATMENT
    if any(term in normalized for term in ("tien luong", "de lai seo", "prognosis")):
        return ClaimType.PROGNOSIS
    return ClaimType.OTHER_MEDICAL_FACT


def _packed_context(value: PackedContext | dict[str, Any] | None) -> PackedContext | None:
    if value is None:
        return None
    if isinstance(value, PackedContext):
        return value
    return PackedContext.model_validate(value)


def _valid_generation_evidence(context: PackedContext | None) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if context is None:
        return records
    for item in context.items:
        if item.source != "chunk":
            continue
        payload = item.payload
        evidence_id = str(payload.get("chunk_id") or item.item_id or "").strip()
        source_id = str(payload.get("source_path") or payload.get("document_id") or "").strip()
        if not evidence_id or not source_id:
            continue
        records.append({
            "evidence_id": evidence_id,
            "source_id": source_id,
            "text": _evidence_text(item.text),
        })
    return records


def _evidence_text(value: str) -> str:
    marker = "\nText:\n"
    if marker in value:
        return value.split(marker, 1)[1].replace("\n...[truncated]", "").strip()
    return value.strip()


def _normalized_entities(text: str) -> set[str]:
    try:
        normalized = normalize_query(text)
    except Exception:
        return set()
    values: set[str] = set()
    for field in ("drug_product", "active_ingredient", "drug_class", "condition", "aliases"):
        values.update(_normalize(item) for item in getattr(normalized, field, []) if item)
    return values


def _deterministic_entailment(claim: str, evidence: str) -> tuple[EntailmentStatus, float, str, bool]:
    claim_norm = f" {_normalize(claim)} "
    evidence_norm = f" {_normalize(evidence)} "
    claim_props = extract_domain_propositions(claim)
    evidence_props = extract_domain_propositions(evidence)
    for claim_prop in claim_props:
        for evidence_prop in evidence_props:
            if claim_prop.subject != evidence_prop.subject or claim_prop.object != evidence_prop.object:
                continue
            opposite = {
                ("is_a", "is_not_a"),
                ("is_not_a", "is_a"),
                ("contains", "does_not_contain"),
                ("does_not_contain", "contains"),
            }
            if (claim_prop.relation, evidence_prop.relation) in opposite:
                return EntailmentStatus.CONTRADICTED, 0.99, "CONTRADICTORY_STATEMENT", True

    claim_tokens = _content_tokens(claim)
    evidence_tokens = _content_tokens(evidence)
    coverage = _coverage(claim_tokens, evidence_tokens)
    shared = claim_tokens & evidence_tokens
    shared_entities = _normalized_entities(claim) & _normalized_entities(evidence)
    claim_asserts_safe = " an toan " in claim_norm and " khong an toan " not in claim_norm
    evidence_prohibits = any(
        marker in evidence_norm
        for marker in (" khong duoc dung ", " khong nen dung ", " chong chi dinh ", " tranh dung ")
    )
    claim_prohibits = any(
        marker in claim_norm
        for marker in (" khong duoc dung ", " khong nen dung ", " chong chi dinh ", " tranh dung ")
    )
    evidence_asserts_safe = " an toan " in evidence_norm and " khong an toan " not in evidence_norm
    if shared_entities and ((claim_asserts_safe and evidence_prohibits) or (claim_prohibits and evidence_asserts_safe)):
        return EntailmentStatus.CONTRADICTED, 0.95, "CONTRADICTORY_STATEMENT", True
    claim_negated = any(marker in claim_norm for marker in _NEGATION_MARKERS)
    evidence_negated = any(marker in evidence_norm for marker in _NEGATION_MARKERS)
    if claim_negated != evidence_negated and coverage >= 0.55 and len(shared) >= 3:
        return EntailmentStatus.CONTRADICTED, max(0.8, coverage), "CONTRADICTORY_STATEMENT", True
    if claim_norm.strip(" ") in evidence_norm:
        return EntailmentStatus.SUPPORTED, 0.99, "DIRECT_SUPPORT", False
    has_unsupported_specificity = any(marker in claim_norm for marker in _PARTIAL_MARKERS) and not any(
        marker in evidence_norm for marker in _PARTIAL_MARKERS
    )
    if has_unsupported_specificity and (coverage >= 0.35 or len(shared) >= 2):
        return EntailmentStatus.PARTIALLY_SUPPORTED, coverage, "SUPPORTS_WEAKER_CLAIM", False
    if coverage >= 0.72 and len(shared) >= 3:
        return EntailmentStatus.SUPPORTED, coverage, "DIRECT_SUPPORT", False
    if coverage >= 0.38 and len(shared) >= 2:
        return EntailmentStatus.PARTIALLY_SUPPORTED, coverage, "SUPPORTS_WEAKER_CLAIM", False
    return EntailmentStatus.UNSUPPORTED, 1.0 - coverage, "MISSING_KEY_RELATION", False


def _content_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", _normalize(text)))
    return {token for token in tokens if len(token) > 1 and token not in _STOPWORDS}


def _coverage(required: set[str], available: set[str]) -> float:
    if not required:
        return 0.0
    return len(required & available) / len(required)


def _citation_match(claim: str, evidence_id: str, source_id: str) -> bool:
    normalized_claim = claim.casefold()
    return bool(
        (evidence_id and evidence_id.casefold() in normalized_claim)
        or (source_id and source_id.casefold() in normalized_claim)
    )


def _normalize(text: str) -> str:
    _, accentless = build_matching_views(str(text or ""))
    return re.sub(r"\s+", " ", accentless).strip()


def _claim_id(text: str, sentence_index: int, claim_index: int) -> str:
    digest = hashlib.sha256(f"{sentence_index}:{claim_index}:{_normalize(text)}".encode("utf-8")).hexdigest()[:16]
    return f"claim:{digest}"


def _bounded_env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        configured = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        configured = default
    return min(maximum, max(minimum, configured))


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


__all__ = [
    "AnswerClaim",
    "ClaimCriticality",
    "ClaimEntailmentVerdict",
    "ClaimEvidenceLink",
    "ClaimGroundingResult",
    "ClaimType",
    "EntailmentStatus",
    "EvidenceScope",
    "P4Mode",
    "P4_CLAIM_GROUNDING_VERSION",
    "P4_CLAIM_SCHEMA_VERSION",
    "P4_CRITICAL_POLICY_VERSION",
    "P4_ENTAILMENT_VERSION",
    "P4_EVIDENCE_MAPPING_VERSION",
    "P4_MAX_CLAIMS",
    "P4_MAX_EVIDENCE_PER_CLAIM",
    "ShadowPolicyAction",
    "VerifiedClaimSet",
    "build_shadow_verified_answer",
    "build_verified_claim_set",
    "compact_claim_grounding",
    "evaluate_claim_grounding",
    "evaluate_shadow_policy",
    "extract_answer_claims",
    "map_claims_to_evidence",
    "p4_max_claims_from_env",
    "p4_max_evidence_from_env",
    "p4_effective_mode",
    "p4_mode_from_env",
    "verify_claim_entailment",
]
