"""Node chạy verifier kỹ thuật sau khi answer đã qua presentation.

Safety có owner riêng; verifier ở đây chỉ kiểm tra cấu trúc và provenance
identity. Khi verifier lỗi runtime, report fail closed nhưng answer không được
gắn nhãn là đã xác minh đúng về mặt lâm sàng.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agent.answer_formatting import finalize_answer_presentation, infer_response_profile
from src.agent.state import ClinicalState
from src.quality.answer_verifier import verify_answer_quality
from src.quality.safe_fallback import sanitize_fallback_reason
from src.retrieval.contracts import PackedContext

logger = logging.getLogger(__name__)


async def answer_quality_node(state: ClinicalState) -> dict[str, Any]:
    """Chỉ kiểm structure/provenance; medical safety thuộc owner riêng."""

    query = state.get("normalized_question") or state.get("user_question", "")
    answer = state.get("final_answer", "")
    if not query or not answer:
        return {}

    if state.get("safety_override"):
        return {
            "answer_quality_report": {
                "passed": True,
                "original_query": query,
                "issues": [],
                "metadata": {
                    "verification_scope": ["deterministic_safety_origin"],
                    "safety_decision": state.get("safety_decision"),
                },
            },
            "source_validation": {
                "version": "source_validation_v1",
                "allowlist_source_ids": [],
                "removed_invalid_source_mentions": [],
                "invalid_source_name_count": 0,
                "origin": "deterministic_safety_policy",
            },
        }

    try:
        packed_context = _parse_model(PackedContext, state.get("packed_context"))
        report_model = verify_answer_quality(
            query=query,
            answer=answer,
            packed_context=packed_context,
            retrieval_trace=state.get("retrieval_trace"),
        )
        profile = state.get("response_profile") or infer_response_profile(
            query,
            severity=state.get("safety_severity"),
            fallback_type=state.get("fallback_type") if state.get("fallback_applied") else None,
        )
        presented = finalize_answer_presentation(
            answer,
            user_question=query,
            response_profile=profile,
            severity=state.get("safety_severity"),
            fallback_type=state.get("fallback_type") if state.get("fallback_applied") else None,
        )
        report = report_model.model_dump(mode="json")
        report.setdefault("metadata", {})["source_validation"] = dict(
            state.get("source_validation") or {}
        )
        return {
            "final_answer": presented,
            "answer_quality_report": report,
            "response_profile": profile,
        }
    except Exception as exc:
        safe_error = sanitize_fallback_reason(exc)
        logger.warning("Answer quality verifier failed safely: %s", safe_error)
        return {
            "answer_quality_report": {
                "passed": False,
                "original_query": query,
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
                "metadata": {
                    "verification_scope": ["presentation", "structural_contract", "provenance_identity"],
                    "medical_semantic_verification": False,
                },
            },
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
