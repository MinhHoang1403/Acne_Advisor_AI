from __future__ import annotations

from src.quality.answer_verifier import verify_answer_quality


def test_validator_does_not_correct_controlled_medical_semantics() -> None:
    answer = "Controlled medical draft that must remain model-owned."
    report = verify_answer_quality(query="Question", answer=answer)
    assert report.metadata["medical_semantic_verification"] is False
    assert report.checked_answer == answer


def test_validator_detects_empty_or_internal_prompt_markers() -> None:
    empty = verify_answer_quality(query="Question", answer="")
    leaked = verify_answer_quality(query="Question", answer="SYSTEM PROMPT: hidden")
    assert empty.passed is False
    assert leaked.passed is False


def test_validator_has_one_structural_provenance_scope() -> None:
    report = verify_answer_quality(query="Question", answer="Draft.")
    assert report.metadata["verification_scope"] == [
        "presentation",
        "structural_contract",
        "provenance_identity",
    ]
