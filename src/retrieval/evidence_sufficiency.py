"""Deterministic P3 evidence sufficiency, bounded retry, and abstention contracts."""

from __future__ import annotations

import hashlib
import os
import time
from enum import Enum
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, field_validator


P3_EVIDENCE_SUFFICIENCY_VERSION = "evidence_sufficiency_v1"
P3_MAX_RETRIEVAL_ATTEMPTS = 2
P3_TRACE_EVENT_LIMIT = 16
P3_EVIDENCE_ID_LIMIT = 64


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceSufficiencyStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    CRITICAL_EVIDENCE_MISSING = "CRITICAL_EVIDENCE_MISSING"


class EvidenceSufficiencyStage(str, Enum):
    PRE_PACK = "PRE_PACK"
    POST_PACK = "POST_PACK"


class RetryEligibility(str, Enum):
    RETRYABLE = "RETRYABLE"
    NON_RETRYABLE = "NON_RETRYABLE"


class EvidenceAbstentionType(str, Enum):
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CRITICAL_EVIDENCE_MISSING = "CRITICAL_EVIDENCE_MISSING"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    SOURCE_PROVENANCE_FAILURE = "SOURCE_PROVENANCE_FAILURE"
    RETRIEVAL_PROVIDER_FAILURE = "RETRIEVAL_PROVIDER_FAILURE"


class P3TraceEventType(str, Enum):
    SUFFICIENCY_ASSESSED = "SUFFICIENCY_ASSESSED"
    RETRY_PLANNED = "RETRY_PLANNED"
    RETRY_STARTED = "RETRY_STARTED"
    RETRY_COMPLETED = "RETRY_COMPLETED"
    RETRY_SKIPPED = "RETRY_SKIPPED"
    ABSTENTION_TRIGGERED = "ABSTENTION_TRIGGERED"


class EvidenceSufficiencyAssessment(_FrozenModel):
    version: str = P3_EVIDENCE_SUFFICIENCY_VERSION
    stage: EvidenceSufficiencyStage
    status: EvidenceSufficiencyStatus
    required_roles: tuple[str, ...] = ()
    satisfied_roles: tuple[str, ...] = ()
    missing_roles: tuple[str, ...] = ()
    critical_missing_roles: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    retry_eligibility: RetryEligibility = RetryEligibility.NON_RETRYABLE
    retry_reason: str | None = None
    provenance_valid: bool = False
    attempt_index: int = 0
    trace_id: str
    elapsed_ms: float = 0.0

    @field_validator("attempt_index")
    @classmethod
    def _attempt_is_bounded(cls, value: int) -> int:
        if value not in {0, 1}:
            raise ValueError("P3 attempt_index must be 0 or 1")
        return value

    @field_validator("evidence_ids", "source_ids")
    @classmethod
    def _identifiers_are_bounded(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > P3_EVIDENCE_ID_LIMIT:
            raise ValueError("P3 evidence identifiers exceed the trace limit")
        return value


class EvidenceRetryPlan(_FrozenModel):
    version: str = P3_EVIDENCE_SUFFICIENCY_VERSION
    attempt_index: int = 1
    reason: str
    query_variant: str
    original_query_hash: str
    retry_query_hash: str
    missing_roles: tuple[str, ...] = ()
    missing_concepts: tuple[str, ...] = ()
    entity_expansion: tuple[str, ...] = ()
    graph_hints: tuple[str, ...] = ()
    retrieval_strategy: str = "deterministic_query_representation_refinement"

    @field_validator("attempt_index")
    @classmethod
    def _retry_is_second_attempt(cls, value: int) -> int:
        if value != 1:
            raise ValueError("P3 permits only attempt_index=1 for retry")
        return value


class EvidenceAbstention(_FrozenModel):
    version: str = P3_EVIDENCE_SUFFICIENCY_VERSION
    abstention_type: EvidenceAbstentionType
    reason: str
    missing_roles: tuple[str, ...] = ()
    critical_missing_roles: tuple[str, ...] = ()
    attempts: int
    evidence_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    safe_user_message_category: str
    trace_id: str

    @field_validator("attempts")
    @classmethod
    def _attempt_count_is_bounded(cls, value: int) -> int:
        if value < 1 or value > P3_MAX_RETRIEVAL_ATTEMPTS:
            raise ValueError("P3 abstention attempts must be between 1 and 2")
        return value


class P3TraceEvent(_FrozenModel):
    event_type: P3TraceEventType
    attempt_index: int
    status: str | None = None
    reason_code: str | None = None
    missing_roles: tuple[str, ...] = ()
    query_hash: str | None = None
    elapsed_ms: float | None = None


class P3ExecutionTrace(_FrozenModel):
    version: str = P3_EVIDENCE_SUFFICIENCY_VERSION
    trace_id: str
    max_attempts: int = P3_MAX_RETRIEVAL_ATTEMPTS
    events: tuple[P3TraceEvent, ...] = ()

    @field_validator("max_attempts")
    @classmethod
    def _max_attempts_is_bounded(cls, value: int) -> int:
        if value < 1 or value > P3_MAX_RETRIEVAL_ATTEMPTS:
            raise ValueError("P3 max_attempts must be between 1 and 2")
        return value

    @field_validator("events")
    @classmethod
    def _events_are_bounded(cls, value: tuple[P3TraceEvent, ...]) -> tuple[P3TraceEvent, ...]:
        if len(value) > P3_TRACE_EVENT_LIMIT:
            raise ValueError("P3 trace event limit exceeded")
        return value

    def append(self, event: P3TraceEvent) -> "P3ExecutionTrace":
        return self.model_copy(update={"events": (*self.events, event)})


class EvidenceSufficiencyResult(_FrozenModel):
    pre_pack: EvidenceSufficiencyAssessment
    post_pack: EvidenceSufficiencyAssessment
    final: EvidenceSufficiencyAssessment


def p3_enabled_from_env() -> bool:
    return _env_bool(os.getenv("P3_EVIDENCE_SUFFICIENCY_ENABLED", "true"), default=True)


def p3_max_attempts_from_env() -> int:
    try:
        configured = int(os.getenv("P3_MAX_RETRIEVAL_ATTEMPTS", "2"))
    except ValueError:
        configured = P3_MAX_RETRIEVAL_ATTEMPTS
    return min(P3_MAX_RETRIEVAL_ATTEMPTS, max(1, configured))


def assess_evidence_sufficiency(
    *,
    evidence_selector: dict[str, Any] | None,
    evidence_packer: dict[str, Any] | None,
    retrieval_status: str | None,
    is_in_domain: bool | None,
    attempt_index: int,
    trace_id: str,
) -> EvidenceSufficiencyResult:
    """Assess source-backed role coverage before and after V5 packing."""

    started = time.perf_counter()
    selector = evidence_selector if isinstance(evidence_selector, dict) else {}
    packer = evidence_packer if isinstance(evidence_packer, dict) else {}
    requirements = (
        selector.get("requirements")
        if isinstance(selector.get("requirements"), dict)
        else {}
    )
    required_roles = _unique_strings(
        requirements.get("required_roles") or ("primary", "source_traceability")
    )
    critical_flags = _unique_strings(requirements.get("critical_safety_flags") or ())
    selected = tuple(
        item for item in selector.get("selected_evidence", ()) if isinstance(item, dict)
    )
    valid_items = tuple(item for item in selected if _valid_provenance(item))
    all_ids = _unique_strings(_candidate_id(item) for item in valid_items)
    source_ids = _unique_strings(_source_id(item) for item in valid_items)
    pre_roles = _roles_for(valid_items)
    invalid_provenance = bool(selected) and len(valid_items) != len(selected)

    pre = _assessment(
        stage=EvidenceSufficiencyStage.PRE_PACK,
        required_roles=required_roles,
        covered_roles=pre_roles,
        critical_flags=critical_flags,
        evidence_ids=all_ids,
        source_ids=source_ids,
        retrieval_status=retrieval_status,
        is_in_domain=is_in_domain,
        invalid_provenance=invalid_provenance,
        packer_status=None,
        attempt_index=attempt_index,
        trace_id=trace_id,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )

    packed_ids = set(_unique_strings(packer.get("selected_evidence_ids") or ()))
    packed_items = tuple(item for item in valid_items if _candidate_id(item) in packed_ids)
    post_roles = _roles_for(packed_items)
    post = _assessment(
        stage=EvidenceSufficiencyStage.POST_PACK,
        required_roles=required_roles,
        covered_roles=post_roles,
        critical_flags=critical_flags,
        evidence_ids=_unique_strings(_candidate_id(item) for item in packed_items),
        source_ids=_unique_strings(_source_id(item) for item in packed_items),
        retrieval_status=retrieval_status,
        is_in_domain=is_in_domain,
        invalid_provenance=invalid_provenance,
        packer_status=str(packer.get("status") or ""),
        attempt_index=attempt_index,
        trace_id=trace_id,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )
    final = post if pre.status == EvidenceSufficiencyStatus.SUFFICIENT else pre
    return EvidenceSufficiencyResult(pre_pack=pre, post_pack=post, final=final)


def build_retry_plan(
    *,
    original_query: str,
    assessment: EvidenceSufficiencyAssessment,
    retrieval_trace_v5: dict[str, Any] | None,
) -> EvidenceRetryPlan:
    """Build one deterministic query representation that cannot equal attempt 0."""

    if (
        assessment.retry_eligibility != RetryEligibility.RETRYABLE
        or assessment.attempt_index != 0
    ):
        raise ValueError("A retry plan requires a retryable attempt-0 assessment")
    trace = retrieval_trace_v5 if isinstance(retrieval_trace_v5, dict) else {}
    observation = (
        trace.get("query_observation")
        if isinstance(trace.get("query_observation"), dict)
        else {}
    )
    entity_expansion = _entity_hints(trace)
    graph_hints = _graph_hints(trace)
    unresolved_roles = _unique_strings(
        [*assessment.missing_roles, *assessment.critical_missing_roles]
    )
    missing_concepts = _unique_strings(
        [*_role_hints(unresolved_roles), *(observation.get("normalized_entity_ids") or ())]
    )
    additions = _unique_strings([*missing_concepts, *entity_expansion, *graph_hints])
    if not additions:
        additions = ("source-backed acne clinical evidence",)
    base = str(original_query or "").strip()
    query_variant = f"{base} | evidence focus: {', '.join(additions)}"[:1000].strip()
    if query_variant.casefold() == base.casefold():
        query_variant = f"{base} | source-backed evidence"[:1000]
    return EvidenceRetryPlan(
        reason=assessment.retry_reason or "Missing required source-backed evidence roles.",
        query_variant=query_variant,
        original_query_hash=_query_hash(base),
        retry_query_hash=_query_hash(query_variant),
        missing_roles=unresolved_roles,
        missing_concepts=missing_concepts,
        entity_expansion=entity_expansion,
        graph_hints=graph_hints,
    )


def build_evidence_abstention(
    assessment: EvidenceSufficiencyAssessment,
) -> EvidenceAbstention:
    abstention_type = _abstention_type(assessment)
    category = {
        EvidenceAbstentionType.CRITICAL_EVIDENCE_MISSING: "critical_medical_evidence_limited",
        EvidenceAbstentionType.RETRIEVAL_PROVIDER_FAILURE: "temporary_retrieval_failure",
        EvidenceAbstentionType.OUT_OF_SCOPE: "out_of_scope",
        EvidenceAbstentionType.SOURCE_PROVENANCE_FAILURE: "untraceable_source_evidence",
        EvidenceAbstentionType.INSUFFICIENT_EVIDENCE: "evidence_limited",
    }[abstention_type]
    return EvidenceAbstention(
        abstention_type=abstention_type,
        reason="; ".join(assessment.reasons) or "Required source-backed evidence is insufficient.",
        missing_roles=assessment.missing_roles,
        critical_missing_roles=assessment.critical_missing_roles,
        attempts=assessment.attempt_index + 1,
        evidence_ids=assessment.evidence_ids,
        source_ids=assessment.source_ids,
        safe_user_message_category=category,
        trace_id=assessment.trace_id,
    )


def append_p3_event(
    trace: dict[str, Any] | None,
    event: P3TraceEvent,
    *,
    trace_id: str,
) -> dict[str, Any]:
    current = P3ExecutionTrace.model_validate(trace) if trace else P3ExecutionTrace(
        trace_id=trace_id,
        max_attempts=p3_max_attempts_from_env(),
    )
    return current.append(event).model_dump(mode="json")


def _assessment(
    *,
    stage: EvidenceSufficiencyStage,
    required_roles: tuple[str, ...],
    covered_roles: tuple[str, ...],
    critical_flags: tuple[str, ...],
    evidence_ids: tuple[str, ...],
    source_ids: tuple[str, ...],
    retrieval_status: str | None,
    is_in_domain: bool | None,
    invalid_provenance: bool,
    packer_status: str | None,
    attempt_index: int,
    trace_id: str,
    elapsed_ms: float,
) -> EvidenceSufficiencyAssessment:
    missing = tuple(role for role in required_roles if role not in covered_roles)
    critical_roles = {"safety", "critical"} if critical_flags else set()
    critical_missing = tuple(role for role in missing if role in critical_roles)
    if critical_flags and "critical" not in covered_roles:
        required_critical_role = "safety" if "safety" not in covered_roles else "critical"
        critical_missing = tuple(dict.fromkeys((*critical_missing, required_critical_role)))

    reasons: list[str] = []
    status = EvidenceSufficiencyStatus.SUFFICIENT
    eligibility = RetryEligibility.NON_RETRYABLE
    retry_reason = None
    retrieval_status = str(retrieval_status or "not_started")
    if is_in_domain is False:
        status = EvidenceSufficiencyStatus.INSUFFICIENT
        reasons.append("OUT_OF_SCOPE")
    elif retrieval_status == "recoverable_error":
        status = EvidenceSufficiencyStatus.INSUFFICIENT
        reasons.append("RETRIEVAL_PROVIDER_FAILURE")
    elif packer_status == "CRITICAL_EVIDENCE_OVERFLOW":
        status = EvidenceSufficiencyStatus.CRITICAL_EVIDENCE_MISSING
        critical_missing = tuple(dict.fromkeys((*critical_missing, "critical")))
        reasons.append("PACKER_CRITICAL_EVIDENCE_OVERFLOW")
    elif invalid_provenance and not evidence_ids:
        status = EvidenceSufficiencyStatus.INSUFFICIENT
        reasons.append("SOURCE_PROVENANCE_FAILURE")
    elif critical_missing:
        status = EvidenceSufficiencyStatus.CRITICAL_EVIDENCE_MISSING
        reasons.append("CRITICAL_SOURCE_BACKED_ROLE_MISSING")
        if attempt_index == 0 and stage == EvidenceSufficiencyStage.PRE_PACK:
            eligibility = RetryEligibility.RETRYABLE
            retry_reason = "A critical source-backed role may be recovered by query refinement."
    elif missing or not evidence_ids or not source_ids:
        status = EvidenceSufficiencyStatus.INSUFFICIENT
        reasons.append("REQUIRED_SOURCE_BACKED_ROLE_MISSING")
        if attempt_index == 0 and stage == EvidenceSufficiencyStage.PRE_PACK:
            eligibility = RetryEligibility.RETRYABLE
            retry_reason = "Missing source-backed roles may be recovered by query refinement."
    else:
        reasons.append("ALL_REQUIRED_SOURCE_BACKED_ROLES_SATISFIED")

    if (
        stage == EvidenceSufficiencyStage.POST_PACK
        and status != EvidenceSufficiencyStatus.SUFFICIENT
    ):
        eligibility = RetryEligibility.NON_RETRYABLE
        retry_reason = "Packing limitations are not corrected by another retrieval call."
    if attempt_index >= 1:
        eligibility = RetryEligibility.NON_RETRYABLE
        if status != EvidenceSufficiencyStatus.SUFFICIENT:
            retry_reason = "P3 maximum retrieval attempts reached."

    return EvidenceSufficiencyAssessment(
        stage=stage,
        status=status,
        required_roles=required_roles,
        satisfied_roles=covered_roles,
        missing_roles=missing,
        critical_missing_roles=critical_missing,
        evidence_ids=evidence_ids,
        source_ids=source_ids,
        reasons=tuple(reasons),
        retry_eligibility=eligibility,
        retry_reason=retry_reason,
        provenance_valid=bool(evidence_ids and source_ids),
        attempt_index=attempt_index,
        trace_id=trace_id,
        elapsed_ms=round(elapsed_ms, 3),
    )


def _candidate_payload(item: dict[str, Any]) -> dict[str, Any]:
    evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
    fused = evidence.get("candidate") if isinstance(evidence.get("candidate"), dict) else {}
    candidate = fused.get("candidate") if isinstance(fused.get("candidate"), dict) else {}
    return candidate


def _candidate_id(item: dict[str, Any]) -> str:
    return str(_candidate_payload(item).get("candidate_id") or "").strip()


def _provenance(item: dict[str, Any]) -> dict[str, Any]:
    value = _candidate_payload(item).get("provenance")
    return value if isinstance(value, dict) else {}


def _valid_provenance(item: dict[str, Any]) -> bool:
    provenance = _provenance(item)
    return bool(
        _candidate_id(item)
        and str(provenance.get("chunk_id") or "").strip()
        and str(provenance.get("source_path") or provenance.get("document_id") or "").strip()
    )


def _source_id(item: dict[str, Any]) -> str:
    provenance = _provenance(item)
    return str(provenance.get("source_path") or provenance.get("document_id") or "").strip()


def _roles_for(items: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    return _unique_strings(
        role
        for item in items
        for role in item.get("roles", ())
        if isinstance(role, str)
    )


def _entity_hints(trace: dict[str, Any]) -> tuple[str, ...]:
    hints: list[str] = []
    for signal in trace.get("entity_signals", ()):
        if not isinstance(signal, dict):
            continue
        hints.extend((str(signal.get("canonical_name") or ""), *signal.get("matched_terms", ())))
    return _unique_strings(hints)


def _graph_hints(trace: dict[str, Any]) -> tuple[str, ...]:
    hints: list[str] = []
    for signal in trace.get("graph_signals", ()):
        if not isinstance(signal, dict):
            continue
        hints.extend(signal.get("relation_path", ()))
        hints.append(str(signal.get("target_entity_id") or ""))
    return _unique_strings(hints)


def _role_hints(roles: Iterable[str]) -> tuple[str, ...]:
    mapping = {
        "primary": "primary clinical evidence",
        "source_traceability": "guideline source",
        "safety": "safety contraindication",
        "critical": "critical safety evidence",
        "drug_class": "drug class classification",
        "ingredient": "active ingredient",
        "contradiction": "class distinction",
        "treatment": "treatment guidance",
    }
    return _unique_strings(mapping.get(role, role) for role in roles)


def _abstention_type(assessment: EvidenceSufficiencyAssessment) -> EvidenceAbstentionType:
    reasons = set(assessment.reasons)
    if assessment.status == EvidenceSufficiencyStatus.CRITICAL_EVIDENCE_MISSING:
        return EvidenceAbstentionType.CRITICAL_EVIDENCE_MISSING
    if "OUT_OF_SCOPE" in reasons:
        return EvidenceAbstentionType.OUT_OF_SCOPE
    if "RETRIEVAL_PROVIDER_FAILURE" in reasons:
        return EvidenceAbstentionType.RETRIEVAL_PROVIDER_FAILURE
    if "SOURCE_PROVENANCE_FAILURE" in reasons:
        return EvidenceAbstentionType.SOURCE_PROVENANCE_FAILURE
    return EvidenceAbstentionType.INSUFFICIENT_EVIDENCE


def _unique_strings(values: Iterable[Any]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(text)
    return tuple(output)


def _query_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _env_bool(value: Any, *, default: bool) -> bool:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


__all__ = [
    "EvidenceAbstention",
    "EvidenceAbstentionType",
    "EvidenceRetryPlan",
    "EvidenceSufficiencyAssessment",
    "EvidenceSufficiencyResult",
    "EvidenceSufficiencyStage",
    "EvidenceSufficiencyStatus",
    "P3ExecutionTrace",
    "P3TraceEvent",
    "P3TraceEventType",
    "P3_EVIDENCE_SUFFICIENCY_VERSION",
    "P3_MAX_RETRIEVAL_ATTEMPTS",
    "RetryEligibility",
    "append_p3_event",
    "assess_evidence_sufficiency",
    "build_evidence_abstention",
    "build_retry_plan",
    "p3_enabled_from_env",
    "p3_max_attempts_from_env",
]
