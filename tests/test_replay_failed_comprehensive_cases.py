from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "replay_failed_comprehensive_cases.py"
    spec = importlib.util.spec_from_file_location("replay_failed_comprehensive_cases", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_selection_filters_previous_runtime_outcomes_and_prioritizes_failures():
    module = _load_module()
    cases = {
        "comparison_ok": {"id": "comparison_ok", "category": "comparison"},
        "comparison_failed": {"id": "comparison_failed", "category": "comparison"},
        "routine_failed": {"id": "routine_failed", "category": "routine"},
    }
    previous = {
        "comparison_ok": {"failure_reasons": "[]", "actual_origin": "llm_generated", "route_match": "True"},
        "comparison_failed": {"failure_reasons": "['format']", "actual_origin": "system_safe_fallback", "route_match": "False"},
        "routine_failed": {"failure_reasons": "['route_mismatch']", "actual_origin": "guardrail", "route_match": "False"},
    }

    fallback_mismatches = module._select_cases(
        cases,
        previous_results=previous,
        failures_only=True,
        categories=set(),
        case_ids=set(),
        previous_origins={"system_safe_fallback"},
        previous_route_mismatch=True,
        stratified_per_category=None,
    )
    stratified = module._select_cases(
        cases,
        previous_results=previous,
        failures_only=False,
        categories=set(),
        case_ids=set(),
        previous_origins=set(),
        previous_route_mismatch=False,
        stratified_per_category=1,
    )

    assert [case["id"] for case in fallback_mismatches] == ["comparison_failed"]
    assert [case["id"] for case in stratified] == ["comparison_failed", "routine_failed"]
