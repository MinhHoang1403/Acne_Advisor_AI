"""LangGraph integration for P4 claim-level grounding diagnostics."""

from __future__ import annotations

import logging
from typing import Any

from src.agent.state import ClinicalState
from src.quality.claim_grounding import (
    ClaimGroundingResult,
    P4TraceEvent,
    P4Mode,
    ShadowPolicyAction,
    compact_claim_grounding,
    evaluate_claim_grounding,
    p4_mode_from_env,
)
from src.quality.safe_fallback import sanitize_fallback_reason


logger = logging.getLogger(__name__)

_CRITICAL_ABSTENTION = (
    "Tôi chưa thể xác nhận an toàn phần thông tin y khoa quan trọng này từ các nguồn đã được truy xuất. "
    "Bạn nên hỏi bác sĩ hoặc dược sĩ trước khi đưa ra quyết định điều trị."
)
_GENERAL_ABSTENTION = (
    "Tôi chưa có đủ bằng chứng nguồn để xác nhận câu trả lời này. "
    "Vui lòng hỏi lại với thông tin cụ thể hơn hoặc tham khảo bác sĩ da liễu."
)


async def claim_grounding_node(state: ClinicalState) -> dict[str, Any]:
    """Evaluate the draft in P4 shadow mode without changing normal output."""

    manifest = state.get("pipeline_manifest") or {}
    mode = p4_mode_from_env(str(manifest.get("p4_mode") or "shadow"))
    draft = state.get("draft_answer") or ""
    p3 = state.get("evidence_sufficiency") or {}
    p3_status = str(p3.get("status") or "") or None
    try:
        result = evaluate_claim_grounding(
            answer=draft,
            query=state.get("standalone_question") or state.get("user_question", ""),
            packed_context=state.get("packed_context"),
            mode=mode,
            p3_status=p3_status,
        )
        output_draft = draft
        if mode == P4Mode.ENFORCE_CRITICAL and result.verified_claims.critical_failures:
            output_draft = _enforced_answer(result, critical=True)
            result = result.model_copy(update={"production_answer_modified": output_draft != draft})
        elif mode == P4Mode.ENFORCE_ALL:
            output_draft = _enforced_answer(result, critical=bool(result.verified_claims.critical_failures))
            result = result.model_copy(update={"production_answer_modified": output_draft != draft})
        compact = compact_claim_grounding(result)
        logger.info(
            "P4 claim grounding: mode=%s status=%s claims=%d critical_failures=%d action=%s",
            mode.value,
            result.status,
            len(result.claims),
            len(result.verified_claims.critical_failures),
            result.shadow_action.value,
        )
        return {
            "draft_answer": output_draft,
            "p4_mode": mode.value,
            "claim_grounding": result.model_dump(mode="json"),
            "p4_trace": [event.model_dump(mode="json") for event in result.trace],
            "p4_degraded": result.degraded,
            "p4_shadow_policy": result.shadow_action.value,
            "shadow_verified_answer": result.shadow_verified_answer,
            "p4_answer_modified": result.production_answer_modified,
            "performance_timings": {
                **(state.get("performance_timings") or {}),
                **{f"p4_{key}": value for key, value in result.timings_ms.items()},
            },
            "retrieval_diagnostics": {
                **(state.get("retrieval_diagnostics") or {}),
                "claim_grounding": compact,
            },
        }
    except Exception as exc:
        safe_error = sanitize_fallback_reason(exc)
        logger.warning("P4 claim grounding degraded safely: %s", safe_error)
        degraded = ClaimGroundingResult(
            mode=mode,
            status="degraded",
            degraded=True,
            degraded_reason=safe_error,
            shadow_action=ShadowPolicyAction.VERIFIER_UNAVAILABLE,
            trace=(
                P4TraceEvent(
                    event="CLAIM_VERIFIER_FAILED",
                    reason_code="P4_RUNTIME_ERROR",
                ),
            ),
        )
        return {
            "draft_answer": draft,
            "p4_mode": mode.value,
            "claim_grounding": degraded.model_dump(mode="json"),
            "p4_trace": [event.model_dump(mode="json") for event in degraded.trace],
            "p4_degraded": True,
            "p4_shadow_policy": degraded.shadow_action.value,
            "shadow_verified_answer": "",
            "p4_answer_modified": False,
        }


def _enforced_answer(result: ClaimGroundingResult, *, critical: bool) -> str:
    projection = result.shadow_verified_answer.strip()
    caution = _CRITICAL_ABSTENTION if critical else _GENERAL_ABSTENTION
    if projection:
        return f"{projection}\n\n**Giới hạn bằng chứng**\n{caution}"
    return caution


__all__ = ["claim_grounding_node"]
