from __future__ import annotations

from src.quality.answer_verifier import apply_answer_guard, verify_answer_quality


def test_verifier_does_not_correct_controlled_medical_semantics() -> None:
    answer = "Controlled medical draft that must remain model-owned."
    report = verify_answer_quality(query="Question", answer=answer)
    guard = apply_answer_guard(query="Question", answer=answer, mode="strict_safe")
    assert report.metadata["medical_semantic_verification"] is False
    assert guard.answer == answer
    assert guard.modified is False


def test_verifier_detects_empty_or_malformed_presentation() -> None:
    empty = verify_answer_quality(query="Question", answer="")
    leaked = verify_answer_quality(query="Question", answer="SYSTEM PROMPT: hidden")
    assert empty.passed is False
    assert leaked.passed is False


def test_unknown_guard_mode_is_metadata_only_and_preserves_answer() -> None:
    guard = apply_answer_guard(query="Question", answer="Draft", mode="unknown")
    assert guard.answer == "Draft"
    assert guard.modified is False
    assert guard.modification_reason == "unsupported_guard_mode_preserved_answer"
