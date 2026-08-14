"""Deterministic answer quality checks for Acne Advisor AI."""

from src.quality.answer_verifier import apply_answer_guard, verify_answer_quality
from src.quality.contracts import (
    AnswerGuardResult,
    AnswerQualityIssue,
    AnswerVerificationReport,
    DomainProposition,
)
from src.quality.severity_guard import (
    SeverityClassification,
    SeverityGuardResult,
    apply_severity_aware_answer_guard,
    classify_medical_severity,
)
from src.quality.safe_fallback import (
    SAFE_FALLBACK_FLOW_VERSION,
    SafeFallbackDecision,
    build_safe_fallback_answer,
    decide_generation_fallback,
    decide_retrieval_fallback,
)

__all__ = [
    "AnswerGuardResult",
    "AnswerQualityIssue",
    "AnswerVerificationReport",
    "DomainProposition",
    "SeverityClassification",
    "SeverityGuardResult",
    "SAFE_FALLBACK_FLOW_VERSION",
    "SafeFallbackDecision",
    "apply_answer_guard",
    "apply_severity_aware_answer_guard",
    "build_safe_fallback_answer",
    "classify_medical_severity",
    "decide_generation_fallback",
    "decide_retrieval_fallback",
    "verify_answer_quality",
]
