"""LangGraph node for deterministic answer quality verification."""

from __future__ import annotations

import logging
import os
from typing import Any

from src.agent.answer_formatting import finalize_answer_presentation, infer_response_profile
from src.agent.state import ClinicalState
from src.quality.answer_verifier import apply_answer_guard
from src.quality.safe_fallback import sanitize_fallback_reason
from src.quality.severity_guard import apply_severity_aware_answer_guard
from src.retrieval.contracts import PackedContext, RetrievalTrace
from src.retrieval.query_normalization import normalize_query

logger = logging.getLogger(__name__)


async def answer_quality_node(state: ClinicalState) -> dict[str, Any]:
    """Verify the finalized answer without calling external services."""

    enabled = os.getenv("ANSWER_VERIFIER_ENABLED", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if not enabled:
        return {
            "answer_quality_report": None,
            "answer_guard_modified": False,
        }

    query = state.get("standalone_question") or state.get("user_question", "")
    answer = state.get("final_answer", "")
    if not query or not answer:
        return {}

    try:
        normalized_query = normalize_query(query)
        packed_context = _parse_model(PackedContext, state.get("packed_context"))
        retrieval_trace = _parse_model(RetrievalTrace, state.get("retrieval_trace"))
        guard_mode = os.getenv("ANSWER_GUARD_MODE", "metadata_only")
        strict_enabled = os.getenv("ANSWER_VERIFIER_STRICT", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if strict_enabled and guard_mode.strip().lower() == "metadata_only":
            guard_mode = "strict_safe"

        guard = apply_answer_guard(
            query=query,
            answer=answer,
            normalized_query=normalized_query,
            packed_context=packed_context,
            retrieval_trace=retrieval_trace,
            mode=guard_mode,
        )
        severity_guard = apply_severity_aware_answer_guard(query=query, answer=guard.answer)
        response_profile = state.get("response_profile") or infer_response_profile(
            query,
            severity=severity_guard.classification.severity,
            guardrail=state.get("guardrail"),
            fallback_type=state.get("fallback_type") if state.get("fallback_applied") else None,
        )
        presented_answer = finalize_answer_presentation(
            severity_guard.answer,
            user_question=query,
            response_profile=response_profile,
            severity=severity_guard.classification.severity,
            guardrail=state.get("guardrail"),
            fallback_type=state.get("fallback_type") if state.get("fallback_applied") else None,
            add_disclaimer=None,
        )
        report_data = guard.report.model_dump(mode="json")
        severity_data = severity_guard.classification.model_dump(mode="json")
        report_data.setdefault("metadata", {})
        report_data["metadata"]["source_validation"] = dict(state.get("source_validation") or {})
        report_data["metadata"]["severity_guard"] = {
            **severity_data,
            "version": "severity_aware_answer_guard_v1",
            "modified": severity_guard.modified,
            "modification_reason": severity_guard.modification_reason,
            "cache_eligible": severity_guard.cache_eligible,
        }
        if severity_guard.modified:
            report_data.setdefault("issues", []).append(
                {
                    "code": severity_guard.modification_reason or "severity_guard_modified_answer",
                    "severity": "warning",
                    "message": "Severity-aware answer guard adjusted the response for medical safety.",
                    "evidence": severity_data,
                    "suggested_fix": None,
                }
            )
        logger.info(
            "Answer quality checked: passed=%s issues=%d modified=%s severity=%s severity_modified=%s",
            guard.report.passed,
            len(guard.report.issues),
            guard.modified,
            severity_guard.classification.severity,
            severity_guard.modified,
        )
        result: dict[str, Any] = {
            "final_answer": presented_answer,
            "answer_quality_report": report_data,
            "answer_guard_modified": guard.modified or severity_guard.modified,
            "answer_guard_mode": guard_mode,
            "medical_severity": severity_guard.classification.severity,
            "severity_guard": severity_data,
            "severity_guard_modified": severity_guard.modified,
            "severity_guard_cache_eligible": severity_guard.cache_eligible,
            "response_profile": response_profile,
        }
        if (
            severity_guard.modified
            and severity_guard.classification.severity == "emergency"
        ):
            # The emergency template replaces the generated text completely, so
            # provenance must describe the final deterministic answer, not the
            # discarded provider draft.
            result.update(
                {
                    "actual_provider": "system",
                    "actual_model": None,
                    "llm_fallback_used": False,
                    "fallback_provider": None,
                    "fallback_model": None,
                    "fallback_applied": True,
                    "fallback_type": "severity_emergency_safety_fallback",
                    "fallback_reason": severity_guard.modification_reason,
                    "fallback_answer": presented_answer,
                    "fallback_cache_eligible": False,
                }
            )
        return result
    except Exception as exc:
        safe_error = sanitize_fallback_reason(exc)
        logger.warning("Answer quality verifier failed safely: %s", safe_error)
        return {
            "answer_quality_report": {
                "passed": False,
                "original_query": query,
                "intent": None,
                "checked_answer": answer,
                "issues": [
                    {
                        "code": "answer_verifier_runtime_error",
                        "severity": "warning",
                        "message": safe_error,
                        "evidence": {},
                        "suggested_fix": None,
                    }
                ],
                "required_facts": [],
                "detected_facts": [],
                "missing_facts": [],
                "contradictions": [],
                "safety_warnings": [],
                "metadata": {},
            },
            "answer_guard_modified": False,
        }


def _parse_model(model_cls: Any, value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, model_cls):
        return value
    if isinstance(value, dict):
        return model_cls.model_validate(value)
    return None


__all__ = ["answer_quality_node"]
