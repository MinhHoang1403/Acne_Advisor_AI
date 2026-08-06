"""Shared constants and immutable configuration for Evaluation V3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DATASET_SCHEMA_VERSION = "evaluation_case_v3"
METRICS_VERSION = "evaluation_metrics_v4"
JUDGE_RUBRIC_VERSION = "route_aware_gemini_v3"
CHECKPOINT_SCHEMA_VERSION = "evaluation_checkpoint_v2"
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
    run_dir: Path | None = None

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_path"] = str(self.dataset_path)
        payload["report_root"] = str(self.report_root)
        payload["run_dir"] = str(self.run_dir) if self.run_dir is not None else None
        return payload

    def resume_identity_json(self) -> dict[str, Any]:
        """Return the stable semantic identity for a checkpointed run.

        Invocation controls such as ``judge_limit`` and retry tuning are kept in
        ``as_json`` for auditability, but must not prevent a 3-case smoke from
        resuming the same 300-case judge run.
        """

        return {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "metrics_version": METRICS_VERSION,
            "judge_rubric_version": JUDGE_RUBRIC_VERSION,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "live_provider": self.live_provider,
            "live_model": self.live_model,
            "judge_provider": self.judge_provider,
            "judge_model": self.judge_model,
            "question_limit": self.question_limit,
            "case_ids": list(self.case_ids),
            "bypass_cache": self.bypass_cache,
            "no_persistence": self.no_persistence,
        }
