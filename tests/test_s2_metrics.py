import pytest

from src.evaluation.ablation_metrics import (
    arithmetic_mean,
    duplicate_slot_rate,
    evidence_retention_rate,
    first_relevant_rank,
    mean_reciprocal_rank,
    recall_at_k,
)


def test_first_relevant_rank_is_one_based_and_missing_is_none():
    assert first_relevant_rank(["a", "b", "c"], {"b"}) == 2
    assert first_relevant_rank(["a", "b", "c"], {"z"}) is None


def test_recall_at_k_uses_query_level_numerator_and_denominator():
    result = recall_at_k(
        [["a", "b", "c"], ["x", "y", "z"], ["m", "n"]],
        [{"b"}, {"x"}, {"missing"}],
        k=2,
    )

    assert result == {"numerator": 2, "denominator": 3, "value": pytest.approx(2 / 3)}


def test_mrr_hand_calculation_counts_missing_query_as_zero():
    result = mean_reciprocal_rank(
        [["a", "b", "c"], ["x", "y"], ["m"]],
        [{"b"}, {"x"}, {"missing"}],
    )

    assert result["reciprocal_rank_sum"] == 1.5
    assert result["denominator"] == 3
    assert result["value"] == 0.5


def test_engineering_rates_have_obvious_hand_calculations():
    assert evidence_retention_rate(["a", "b", "c"], ["a", "c"]) == {
        "numerator": 2,
        "denominator": 3,
        "value": pytest.approx(2 / 3),
    }
    assert duplicate_slot_rate(["a", "a", "b", "b"]) == {
        "numerator": 2,
        "denominator": 4,
        "value": 0.5,
    }
    assert arithmetic_mean([1.0, 2.0, 6.0]) == {
        "sum": 9.0,
        "denominator": 3,
        "value": 3.0,
    }


def test_metrics_reject_invalid_denominators_or_parallel_inputs():
    with pytest.raises(ValueError, match="equal length"):
        recall_at_k([["a"]], [], k=1)
    with pytest.raises(ValueError, match="positive"):
        recall_at_k([["a"]], [{"a"}], k=0)


def test_empty_inputs_are_defined_without_division_by_zero():
    assert recall_at_k([], [], k=1) == {"numerator": 0, "denominator": 0, "value": 0.0}
    assert mean_reciprocal_rank([], []) == {
        "reciprocal_rank_sum": 0.0,
        "denominator": 0,
        "value": 0.0,
    }
    assert evidence_retention_rate([], []) == {"numerator": 0, "denominator": 0, "value": 1.0}
