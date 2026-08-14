from __future__ import annotations

import time

import pytest

from src.agent.nodes.guardrails import domain_guard_node
from src.agent.nodes.preparation import rewrite_question_node
from src.agent.nodes.respond import finalize_response_node
from src.agent.source_presentation import (
    build_source_allowlist,
    build_source_metadata,
    validate_answer_source_mentions,
)


def _allowlist() -> list[dict]:
    return build_source_allowlist(
        ["C:\\knowledge\\QD_4416_CUT.PDF", "web_raw_dataset.json"],
        contexts=[{"source_file": "qd_4416_cut.pdf", "page": 3, "chunk_id": "chunk-1"}],
    )


def test_source_alias_keeps_canonical_identity_and_page() -> None:
    metadata = build_source_metadata(
        ["C:\\knowledge\\QD_4416_CUT.PDF"],
        contexts=[{"source_file": "qd_4416_cut.pdf", "page": 3}],
    )
    assert metadata[0]["source_id"] == "qd_4416_cut.pdf"
    assert metadata[0]["page"] == 3


def test_source_validation_removes_only_unretrieved_names() -> None:
    result = validate_answer_source_mentions(
        "Theo invented-guide.pdf, hãy xem qd_4416_cut.pdf ở trang 3.",
        _allowlist(),
    )
    assert "invented-guide.pdf" not in result.answer
    assert "qd_4416_cut.pdf" in result.answer
    assert result.removed_mentions == ("invented-guide.pdf",)


@pytest.mark.asyncio
async def test_source_request_uses_only_retrieved_canonical_sources() -> None:
    result = await finalize_response_node(
        {
            "user_question": "Theo kho dữ liệu, nguồn nào đã được truy hồi?",
            "is_in_domain": True,
            "guardrail": "in_domain_rule",
            "fallback_applied": False,
            "fallback_type": "none",
            "source_allowlist": _allowlist(),
            "sources": ["qd_4416_cut.pdf", "web_raw_dataset.json"],
            "vector_contexts": [],
            "draft_answer": "Tài liệu 1",
            "performance_timings": {},
        }
    )
    assert "Tài liệu 1" not in result["final_answer"]
    assert result["source_validation"]["invalid_source_name_count"] == 0


@pytest.mark.asyncio
async def test_coreference_rewrite_uses_bounded_history(monkeypatch) -> None:
    captured: dict[str, str] = {}

    async def fake_generate_llm_response(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return {"text": "Hoạt chất chính của thuốc đang nói tới là gì?"}

    monkeypatch.setattr("src.agent.nodes.preparation.generate_llm_response", fake_generate_llm_response)
    result = await rewrite_question_node(
        {
            "normalized_question": "Hoạt chất chính của thuốc đó là gì?",
            "conversation_history": [
                *({"role": "user", "content": f"old-{index}"} for index in range(8)),
                {"role": "user", "content": "Tôi đang nói về một thuốc trị mụn."},
            ],
            "llm_provider": "mock",
            "allow_model_fallback": False,
        }
    )
    assert result["use_history_context"] is True
    assert "old-0" not in captured["prompt"]


@pytest.mark.asyncio
async def test_ood_emergency_redirection_is_preserved() -> None:
    result = await domain_guard_node(
        {"standalone_question": "Tôi bị đau bụng dữ dội, chẩn đoán giúp tôi.", "conversation_history": []}
    )
    assert result["is_in_domain"] is False
    assert result["guardrail"] == "medical_emergency_out_of_scope"


def test_source_validation_has_bounded_overhead() -> None:
    started = time.perf_counter()
    result = validate_answer_source_mentions(
        "Theo invented.pdf và qd_4416_cut.pdf, hãy xem Tài liệu 1.",
        _allowlist(),
    )
    assert "invented.pdf" not in result.answer
    assert (time.perf_counter() - started) < 0.05
