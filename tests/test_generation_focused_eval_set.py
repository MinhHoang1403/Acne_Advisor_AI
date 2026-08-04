from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "notebooks" / "eval_data" / "acne_rag_eval_generation_focused.jsonl"
BUILDER_PATH = PROJECT_ROOT / "notebooks" / "eval_data" / "build_generation_focused_eval_set.py"
VALIDATOR_PATH = PROJECT_ROOT / "scripts" / "validate_generation_focused_eval_set.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generation_focused_dataset_matches_builder_contract() -> None:
    builder = _load_module(BUILDER_PATH, "build_generation_focused_eval_set")
    rows = builder.build_cases()
    builder.validate_cases(rows)

    assert len(rows) == 300
    assert Counter(row["category"] for row in rows) == builder.CATEGORY_TARGETS
    assert len({row["id"] for row in rows}) == 300
    assert len({row["question"] for row in rows}) == 300


def test_generation_focused_dataset_passes_validator() -> None:
    validator = _load_module(VALIDATOR_PATH, "validate_generation_focused_eval_set")
    rows = validator.load_rows(DATASET_PATH)
    counts = validator.validate_rows(rows)

    assert len(rows) == 300
    assert counts == validator.CATEGORY_TARGETS
