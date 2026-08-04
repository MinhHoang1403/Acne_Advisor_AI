from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

from scripts.validate_comprehensive_eval_set import read_jsonl, validate_rows


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "notebooks" / "eval_data" / "acne_rag_eval_comprehensive_v1.jsonl"
BUILDER_PATH = ROOT / "notebooks" / "eval_data" / "build_comprehensive_eval_set.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_comprehensive_eval_set", BUILDER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comprehensive_builder_produces_balanced_valid_dataset() -> None:
    builder = _load_builder()
    rows = builder.build_cases()
    builder.validate_cases(rows)
    counts = validate_rows(rows)
    assert len(rows) == 300
    assert set(counts) == set(builder.CATEGORIES)
    assert all(value == 20 for value in counts.values())


def test_canonical_dataset_has_route_and_safety_coverage() -> None:
    rows = read_jsonl(DATASET_PATH)
    counts = validate_rows(rows)
    routes = Counter(row["expected_route"] for row in rows)
    assert len(rows) == 300
    assert all(counts[category] == 20 for category in counts)
    assert routes["llm_generated"] >= 220
    assert routes["system_safe_fallback"] >= 40
    assert routes["guardrail"] >= 20
    assert sum(1 for row in rows if row["critical_case"]) >= 40
