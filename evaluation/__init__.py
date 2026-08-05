"""Canonical, CLI-first evaluation framework for Acne Advisor AI."""

from .models import (
    DATASET_SCHEMA_VERSION,
    JUDGE_RUBRIC_VERSION,
    METRICS_VERSION,
    EvaluationConfig,
)

__all__ = [
    "DATASET_SCHEMA_VERSION",
    "JUDGE_RUBRIC_VERSION",
    "METRICS_VERSION",
    "EvaluationConfig",
]
