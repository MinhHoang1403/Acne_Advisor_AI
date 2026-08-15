from __future__ import annotations

from src.quality.answer_verifier import verify_answer_quality
from src.retrieval.contracts import ContextItem, PackedContext


def _packed(*, source_id: str = "guideline") -> PackedContext:
    return PackedContext(
        original_query="Câu hỏi",
        items=[
            ContextItem(
                item_id="chunk-1",
                role="medical_evidence",
                text="Bằng chứng đã truy hồi.",
                payload={"source_id": source_id, "chunk_id": "chunk-1"},
                reason="rrf_rank",
            )
        ],
        context_text="[Evidence 1 | source=guideline | chunk=chunk-1]\nBằng chứng đã truy hồi.",
    )


def test_verifier_checks_structure_and_provenance_without_medical_truth_table() -> None:
    report = verify_answer_quality(
        query="Thuốc X thuộc nhóm nào?",
        answer="Câu trả lời do model tổng hợp từ bằng chứng.",
        packed_context=_packed(),
    )

    assert report.metadata["medical_semantic_verification"] is False
    assert report.metadata["verification_scope"] == [
        "presentation",
        "structural_contract",
        "provenance_identity",
    ]


def test_verifier_reports_missing_packed_source_identity() -> None:
    report = verify_answer_quality(
        query="Câu hỏi",
        answer="Câu trả lời.",
        packed_context=_packed(source_id=""),
    )

    assert "packed_evidence_missing_identity" in [issue.code for issue in report.issues]
