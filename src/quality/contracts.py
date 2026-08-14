"""Pydantic contracts for deterministic answer quality verification."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AnswerQualityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    severity: Literal["info", "warning", "error", "critical"]
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    suggested_fix: str | None = None


class AnswerVerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    original_query: str
    intent: str | None = None
    checked_answer: str
    issues: list[AnswerQualityIssue] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnswerGuardResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    original_answer: str
    report: AnswerVerificationReport
    modified: bool
    modification_reason: str | None = None


__all__ = [
    "AnswerGuardResult",
    "AnswerQualityIssue",
    "AnswerVerificationReport",
]
