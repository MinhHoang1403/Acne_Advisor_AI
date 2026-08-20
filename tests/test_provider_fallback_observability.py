from __future__ import annotations

import pytest

from src.agent import action_decision as decision_module
from src.resilience.exceptions import ProviderUnavailableError


@pytest.mark.asyncio
async def test_agent_decision_preserves_failed_provider_fallback_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider failure trace remains visible after the Agent fails closed."""

    error = ProviderUnavailableError("fallback provider unavailable")
    error.requested_provider = "gemini"  # type: ignore[attr-defined]
    error.requested_model = "gemini-3.5-flash-lite"  # type: ignore[attr-defined]
    error.fallback_chain = [  # type: ignore[attr-defined]
        {
            "provider": "gemini",
            "model": "gemini-3.5-flash-lite",
            "role": "primary",
            "status": "failed",
            "reason": "quota_exhausted",
        },
        {
            "provider": "gemini",
            "model": "gemini-3.1-flash-lite",
            "role": "fallback",
            "status": "failed",
            "reason": "provider_unavailable",
        },
    ]

    async def fail_with_trace(**_: object) -> dict:
        raise error

    monkeypatch.setattr(decision_module, "generate_llm_response", fail_with_trace)

    result = await decision_module.select_agent_action(
        {
            "normalized_question": "Mụn đầu đen là gì?",
            "retrieval_attempt": 0,
            "allow_model_fallback": True,
        }
    )

    assert result["next_action"] == "abstain"
    assert result["fallback_reason_code"] == "provider_unavailable"
    assert result["requested_provider"] == "gemini"
    assert result["requested_model"] == "gemini-3.5-flash-lite"
    assert [entry["status"] for entry in result["fallback_chain"]] == ["failed", "failed"]
    assert result["agent_decision"]["fallback_chain"] == result["fallback_chain"]
