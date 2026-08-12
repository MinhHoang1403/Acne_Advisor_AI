from scripts.eval_p4_claim_grounding import evaluate_fixture


def test_frozen_p4_calibration_fixture_is_large_diverse_and_shadow_safe():
    report = evaluate_fixture()
    assert report["fixture_frozen"] is True
    assert report["case_count"] >= 30
    assert set(report["gold_distribution"]) == {
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "UNSUPPORTED",
        "CONTRADICTED",
        "NO_EVIDENCE",
    }
    assert report["critical_case_count"] >= 10
    assert report["metrics"]["critical_false_allow_rate"] == 0.0
    assert report["metrics"]["critical_extraction_recall"] == 1.0
    assert report["metrics"]["verifier_error_rate"] == 0.0
    assert report["metrics"]["shadow_answer_change_rate"] == 0.0
    assert report["shadow_ready"] is True


def test_metric_registry_has_explicit_denominator_contracts():
    definitions = evaluate_fixture()["metrics"]["definitions"]
    assert definitions
    assert all("/" in definition for definition in definitions.values())
