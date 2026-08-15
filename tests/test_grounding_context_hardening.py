from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.nodes.preparation import prepare_request_node
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


def test_source_metadata_deduplicates_filename_and_stable_source_id() -> None:
    metadata = build_source_metadata(
        [
            "nice_ng198_2026-08-03.md",
            "nice_ng198_2026_08",
            "qd_4416_cut.pdf",
            "vn_moh_dermatology_4416_2023_acne",
        ],
        contexts=[
            {
                "source_file": "nice_ng198_2026-08-03.md",
                "document_title": "Acne vulgaris: management (NG198)",
                "chunk_id": "nice-chunk",
            },
            {
                "source_file": "qd_4416_cut.pdf",
                "document_title": "Huong dan chan doan va dieu tri cac benh da lieu",
                "chunk_id": "moh-chunk",
            },
        ],
    )

    assert {entry["source_id"] for entry in metadata} == {
        "nice_ng198_2026-08-03.md",
        "qd_4416_cut.pdf",
    }
    assert {entry["display_name"] for entry in metadata} == {
        "Acne vulgaris: management (NG198)",
        "Tài liệu tiếng Việt về mụn trứng cá",
    }
    assert all(entry["chunk_id"] for entry in metadata)


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
            "user_question": "Nguồn nào đã được truy hồi?",
            "fallback_applied": False,
            "fallback_type": "none",
            "source_allowlist": _allowlist(),
            "sources": ["qd_4416_cut.pdf", "web_raw_dataset.json"],
            "vector_contexts": [],
            "draft_answer": "Tài liệu 1",
        }
    )
    assert "Tài liệu tiếng Việt về mụn trứng cá" in result["final_answer"]
    assert "invented-guide.pdf" not in result["final_answer"]
    assert {item["source_id"] for item in _allowlist()} == {
        "qd_4416_cut.pdf",
        "web_raw_dataset.json",
    }


@pytest.mark.asyncio
async def test_preparation_has_one_bounded_history_representation(monkeypatch) -> None:
    monkeypatch.setenv("MAX_CONVERSATION_HISTORY_MESSAGES", "2")
    monkeypatch.setenv("MAX_HISTORY_MESSAGE_CHARS", "8")
    result = await prepare_request_node(
        {
            "user_question": "  Còn thuốc đó? ",
            "conversation_history": [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "benzoyl peroxide"},
                {"role": "user", "content": "dùng thế nào"},
            ],
        }
    )
    assert result["normalized_question"] == "Còn thuốc đó?"
    assert result["conversation_context"] == {
        "messages": [
            {"role": "assistant", "content": "benzoyl "},
            {"role": "user", "content": "dùng thế"},
        ],
        "message_count": 2,
    }


def test_duplicate_semantic_router_modules_are_removed() -> None:
    assert not Path("src/agent/nodes/guardrails.py").exists()
    assert not Path("src/agent/nodes/severity.py").exists()
