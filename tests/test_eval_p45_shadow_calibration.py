import json
from datetime import datetime, timezone

import pytest

from scripts.eval_p45_shadow_calibration import (
    HumanReviewRecord,
    assert_shadow_runtime,
    build_review_package,
    capture_generation_record,
    load_question_set,
    merge_human_labels_with_predictions,
    score_human_labels,
    validate_human_labels,
    validate_question_set,
    verify_frozen_package,
)


def test_frozen_question_set_has_required_size_diversity_and_critical_ratio():
    report = validate_question_set(load_question_set())

    assert report["passed"] is True
    assert report["question_count"] == 75
    assert report["critical_question_count"] == 18
    assert report["critical_question_ratio"] == 0.24
    assert set(report["languages"]) == {"VI", "EN", "MIXED"}
    assert len(report["categories"]) >= 15
    assert set(report["origins"]) == {
        "HUMAN_WRITTEN",
        "HISTORICAL_USER_STYLE",
        "CURATED_PRODUCTION_LIKE",
        "EDGE_CASE",
    }


def test_shadow_runtime_rejects_enforcement(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_PIPELINE_VERSION", "v5")
    monkeypatch.setenv("P4_MODE", "enforce_critical")
    monkeypatch.setenv("P4_CRITICAL_ENFORCEMENT_READY", "true")

    with pytest.raises(RuntimeError, match="P4_5_MODE_BLOCKER"):
        assert_shadow_runtime()


def test_capture_generation_record_preserves_natural_output_and_diagnostics():
    question = load_question_set().questions[0]
    result = {
        "answer": "Mụn hình thành do nhiều cơ chế.",
        "packed_context": {
            "items": [
                {
                    "item_id": "chunk-1",
                    "source": "chunk",
                    "role": "primary",
                    "text": "Mụn liên quan đến bít tắc nang lông.",
                    "payload": {
                        "chunk_id": "chunk-1",
                        "document_id": "doc-1",
                        "source_path": "sample_data/acne.pdf",
                    },
                }
            ]
        },
        "evidence_sufficiency": {"status": "SUFFICIENT"},
        "retrieval_attempt": 0,
        "sources": ["sample_data/acne.pdf"],
        "source_allowlist": ["sample_data/acne.pdf"],
        "p4_mode": "shadow",
        "p4_shadow_policy": "WOULD_ALLOW",
        "p4_degraded": False,
        "p4_answer_modified": False,
        "claim_grounding": {"claims": [], "evidence_links": [], "verdicts": []},
        "cache_checked": True,
        "cache_hit": False,
        "actual_provider": "gemini",
        "actual_model": "gemini-test",
        "pipeline_fingerprint": "fingerprint",
    }

    record = capture_generation_record(
        question=question,
        result=result,
        started=datetime.now(timezone.utc),
        ordinal=1,
    )

    assert record["final_answer"] == result["answer"]
    assert record["fresh_generation"] is True
    assert record["packed_evidence_ids"] == ["chunk-1"]
    assert record["p4_mode"] == "shadow"
    assert record["p4_answer_modified"] is False


def test_review_package_is_blind_frozen_and_does_not_invent_human_labels(tmp_path):
    question_set = load_question_set()
    question = question_set.questions[0]
    generated = {
        "schema_version": "p45_calibration_package_v1",
        "dataset_version": question_set.dataset_version,
        "pipeline_fingerprint": "fingerprint",
        "records": [
            {
                "case_id": question.id,
                "status": "SUCCESS",
                "final_answer": "Mụn liên quan đến bít tắc nang lông.",
                "p3_status": "SUFFICIENT",
                "p4_shadow_action": "WOULD_ALLOW",
                "actual_provider": "gemini",
                "actual_model": "gemini-test",
                "packed_evidence": [
                    {
                        "evidence_id": "chunk-1",
                        "text": "Mụn liên quan đến bít tắc nang lông.",
                    }
                ],
                "claim_grounding": {
                    "claims": [
                        {
                            "claim_id": "claim-1",
                            "text": "Mụn liên quan đến bít tắc nang lông.",
                            "claim_type": "MECHANISM",
                            "criticality": "NON_CRITICAL",
                        }
                    ],
                    "evidence_links": [
                        {
                            "claim_id": "claim-1",
                            "evidence_id": "chunk-1",
                            "source_id": "sample_data/acne.pdf",
                            "provenance_valid": True,
                        }
                    ],
                    "verdicts": [
                        {
                            "claim_id": "claim-1",
                            "verdict": "SUPPORTED",
                            "reason_code": "DIRECT_SUPPORT",
                        }
                    ],
                    "shadow_action": "WOULD_ALLOW",
                },
            }
        ],
    }
    (tmp_path / "p45_generated_answers.json").write_text(
        json.dumps(generated, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = build_review_package(question_set=question_set, output_dir=tmp_path)
    blind = [json.loads(line) for line in (tmp_path / "p45_review_blind.jsonl").read_text(encoding="utf-8").splitlines()]
    predicted = [json.loads(line) for line in (tmp_path / "p45_review_with_predictions.jsonl").read_text(encoding="utf-8").splitlines()]

    assert manifest["frozen"] is True
    assert manifest["reviewer_count"] == 0
    assert manifest["adjudication_status"] == "PENDING_HUMAN_REVIEW"
    assert all(row["human_verdict"] is None for row in blind)
    assert all(row["p4_verdict"] is None for row in blind)
    claim_prediction = next(row for row in predicted if row["record_type"] == "CLAIM_REVIEW")
    assert claim_prediction["p4_verdict"] == "SUPPORTED"
    assert json.loads((tmp_path / "p45_metrics.json").read_text(encoding="utf-8"))["status"] == "P4_5_WAITING_FOR_HUMAN_REVIEW"


def test_label_validation_preserves_frozen_row_identities(tmp_path):
    expected = [
        HumanReviewRecord(
            record_type="CLAIM_REVIEW",
            case_id="p45-001",
            claim_id="claim-1",
            question="Q",
            answer="A",
            claim_text="C",
        ).model_dump(mode="json")
    ]
    label_path = tmp_path / "labels.jsonl"
    label_path.write_text(json.dumps(expected[0], ensure_ascii=False) + "\n", encoding="utf-8")

    result = validate_human_labels(label_path, expected)

    assert result["passed"] is True
    assert result["complete"] is False
    assert result["completed_claim_labels"] == 0


def test_human_scoring_uses_human_labels_and_reports_false_allow_denominator():
    rows = [
        HumanReviewRecord(
            record_type="CASE_REVIEW",
            case_id="p45-001",
            question="Q",
            answer="A",
            missing_claims=[],
            missing_critical_claims=[],
        ),
        HumanReviewRecord(
            record_type="CLAIM_REVIEW",
            case_id="p45-001",
            claim_id="claim-1",
            question="Q",
            answer="A",
            claim_text="C",
            criticality="CRITICAL",
            p4_verdict="SUPPORTED",
            p4_shadow_action="WOULD_ALLOW",
            human_verdict="UNSUPPORTED",
            human_criticality="CRITICAL",
            mapping_correct=True,
            claim_extraction_correct=True,
            provenance_valid=True,
        ),
    ]

    report = score_human_labels(rows)

    assert report["adjudicated_claims"] == 1
    assert report["metrics"]["critical_false_allow_numerator"] == 1
    assert report["metrics"]["critical_false_allow_denominator"] == 1
    assert report["metrics"]["critical_false_allow_rate"] == 1.0
    assert report["readiness"] == "P4_CRITICAL_ENFORCEMENT_NEEDS_MORE_DATA"


def test_blind_human_labels_are_merged_with_frozen_predictions_before_scoring():
    human = HumanReviewRecord(
        record_type="CLAIM_REVIEW",
        case_id="p45-001",
        claim_id="claim-1",
        question="Q",
        answer="A",
        claim_text="C",
        p4_verdict=None,
        p4_shadow_action=None,
        human_verdict="SUPPORTED",
        human_criticality="NON_CRITICAL",
        mapping_correct=True,
        claim_extraction_correct=True,
    )
    prediction = human.model_dump(mode="json") | {
        "p4_verdict": "SUPPORTED",
        "p4_shadow_action": "WOULD_ALLOW",
    }

    merged = merge_human_labels_with_predictions([human], [prediction])

    assert merged[0].human_verdict == "SUPPORTED"
    assert merged[0].p4_verdict == "SUPPORTED"
    assert merged[0].p4_shadow_action == "WOULD_ALLOW"


def test_frozen_package_verifier_detects_immutable_answer_tampering(tmp_path):
    question_set = load_question_set()
    generated = {
        "schema_version": "p45_calibration_package_v1",
        "dataset_version": question_set.dataset_version,
        "pipeline_fingerprint": "fingerprint",
        "records": [],
    }
    (tmp_path / "p45_generated_answers.json").write_text(
        json.dumps(generated, ensure_ascii=False),
        encoding="utf-8",
    )
    build_review_package(question_set=question_set, output_dir=tmp_path)
    assert verify_frozen_package(tmp_path)["passed"] is True

    (tmp_path / "p45_generated_answers.json").write_text("{}\n", encoding="utf-8")

    report = verify_frozen_package(tmp_path)
    assert report["passed"] is False
    assert "frozen hash mismatch: p45_generated_answers.json" in report["errors"]
