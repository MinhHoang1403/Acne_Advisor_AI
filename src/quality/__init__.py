"""Deterministic answer quality checks for Acne Advisor AI."""

from src.quality.answer_verifier import verify_answer_quality
from src.quality.contracts import (
    AnswerQualityIssue,
    AnswerVerificationReport,
)
from src.quality.safe_fallback import (
    SAFE_FALLBACK_FLOW_VERSION,
    SafeFallbackDecision,
    build_safe_fallback_answer,
    decide_generation_fallback,
)

__all__ = [
    "AnswerQualityIssue",
    "AnswerVerificationReport",
    "SAFE_FALLBACK_FLOW_VERSION",
    "SafeFallbackDecision",
    "build_safe_fallback_answer",
    "decide_generation_fallback",
    "verify_answer_quality",
]
