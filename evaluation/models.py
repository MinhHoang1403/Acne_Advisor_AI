"""Shared constants and immutable configuration for Evaluation V3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DATASET_SCHEMA_VERSION = "evaluation_case_v3"
METRICS_VERSION = "evaluation_metrics_v3"
JUDGE_RUBRIC_VERSION = "route_aware_gemini_v3"
JUDGE_SCORE_DELTA_MAX = 25.0
FINAL_REPORT_NAME = "BAO_CAO_DANH_GIA_HE_THONG_V3.md"

CATEGORIES = (
    "core_knowledge",
    "active_ingredients",
    "product_entity_alias",
    "comparison",
    "skincare_routine",
    "treatment_plan_reference",
    "retrieval_source_traceability",
    "entity_graph_relation",
    "multi_turn_context",
    "exact_format_instruction",
    "antibiotic_stewardship",
    "pregnancy_lactation",
    "mild_adverse_false_escalation",
    "urgent_emergency",
    "out_of_domain_insufficient_evidence",
)

EXPECTED_BEHAVIORS = {
    "answer",
    "cautious_answer",
    "emergency_action",
    "refuse_or_redirect",
    "safe_insufficient_evidence",
}
ACCEPTABLE_ORIGINS = {
    "llm_generated",
    "safe_fallback",
    "guardrail",
    "emergency_response",
}
SAFETY_LEVELS = {"normal", "caution", "urgent", "emergency", "out_of_domain"}


@dataclass(frozen=True)
class EvaluationConfig:
    """Frozen configuration persisted with each live or judge run."""

    dataset_path: Path
    report_root: Path
    live_provider: str = "ollama"
    live_model: str = "qwen3:8b"
    judge_provider: str = "gemini"
    judge_model: str = "gemini-3.1-flash-lite"
    question_limit: int = 300
    judge_limit: int = 300
    case_ids: tuple[str, ...] = ()
    bypass_cache: bool = True
    no_persistence: bool = True
    checkpoint: bool = True
    request_timeout_seconds: int = 210
    judge_attempts: int = 5
    judge_retry_base_seconds: float = 2.0

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_path"] = str(self.dataset_path)
        payload["report_root"] = str(self.report_root)
        return payload
