from scripts.check_answer_contracts import run_answer_contract_checks


def test_answer_contract_check_is_explicitly_not_clinical_quality() -> None:
    report = run_answer_contract_checks()
    assert report["passed"] is True
    assert report["scope"] == "implementation_contracts_not_clinical_quality"
