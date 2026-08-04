from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from rag_eval_utils import (  # noqa: E402
    CHART_FILENAMES,
    CORE_METRIC_KEYS,
    contains_markdown_table,
    create_evaluation_charts,
    judge_disagreement_rows,
    judge_score_to_100,
    parse_judge_json,
    read_jsonl,
    score_case,
    select_judge_sample,
    summarize_core_metrics,
    summarize_judge_results,
)


DATASET_PATH = PROJECT_ROOT / "notebooks" / "eval_data" / "acne_rag_eval_set.jsonl"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "rag_llm_judge.ipynb"
BUILD_EVAL_SET_PATH = PROJECT_ROOT / "notebooks" / "eval_data" / "build_acne_eval_set.py"


def _notebook_cells() -> list[dict]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return notebook.get("cells", [])


def _cell_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _assignment_cells(name: str) -> list[tuple[int, str]]:
    cells = []
    pattern = re.compile(rf"^\s*{re.escape(name)}\s*=", re.MULTILINE)
    for index, cell in enumerate(_notebook_cells()):
        if cell.get("cell_type") == "code" and pattern.search(_cell_source(cell)):
            cells.append((index, _cell_source(cell)))
    return cells


def test_dataset_has_exactly_300_unique_cases() -> None:
    cases = read_jsonl(DATASET_PATH)
    assert len(cases) == 300
    ids = [case["id"] for case in cases]
    questions = [case["question"] for case in cases]
    assert len(ids) == len(set(ids))
    assert len(questions) == len(set(questions))


def test_dataset_required_fields_and_categories() -> None:
    cases = read_jsonl(DATASET_PATH)
    categories = {case["category"] for case in cases}
    assert categories == {
        "core_knowledge",
        "active_ingredients",
        "product_entity",
        "treatment_plan",
        "comparison_table",
        "exact_format",
        "safety_pregnancy",
        "safety_urgent_emergency",
        "antibiotic_stewardship",
        "out_of_domain",
        "multi_turn_like",
        "edge_cases",
    }
    for case in cases:
        assert case.get("id")
        assert case.get("category")
        assert case.get("question")
        assert isinstance(case.get("expected_keywords"), list)
        assert isinstance(case.get("forbidden_keywords"), list)


def test_build_eval_set_generates_300_cases_with_unique_ids() -> None:
    spec = importlib.util.spec_from_file_location("build_acne_eval_set", BUILD_EVAL_SET_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    cases = module.build_cases()
    module.validate_cases(cases)
    ids = [case["id"] for case in cases]
    assert len(cases) == 300
    assert len(ids) == len(set(ids))
    assert all(case.get("category") and case.get("question") for case in cases)


def test_notebook_delegates_to_comprehensive_runner() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))

    assert "ComprehensiveRunner" in text
    assert "EvaluationConfig" in text
    assert "DATASET_PATH = Path(\"notebooks/eval_data/acne_rag_eval_comprehensive_v1.jsonl\")" in text
    assert "COMPREHENSIVE_EVALUATION" not in text
    assert "run(resume=RESUME)" in text

    for legacy_name in [
        "EVAL_ALLOW_LIVE_CHAT",
        "EVAL_USE_LLM_JUDGE",
        "EVAL_SAMPLE_LIMIT",
        "EVAL_JUDGE_PROVIDER",
    ]:
        assert legacy_name not in text


def test_notebook_uses_safe_comprehensive_defaults() -> None:
    cells = _notebook_cells()
    text = "\n".join(_cell_source(cell) for cell in cells)

    required = [
        'RUN_LIVE_EVAL = False',
        'RUN_LLM_JUDGE = False',
        'JUDGE_PROVIDER = "gemini"',
        'JUDGE_GEMINI_MODEL = "gemini-3.1-flash-lite"',
        'LIVE_EVAL_PROVIDER = "ollama"',
        'LIVE_EVAL_MODEL = "qwen3:8b"',
        'BYPASS_RUNTIME_CACHE_FOR_LIVE_EVAL = True',
        'RESUME = False',
        'SMOKE_MODE = False',
        'NO_PERSISTENCE = False',
    ]
    for item in required:
        assert item in text

    for index, cell in enumerate(cells, 1):
        assert not cell.get("outputs"), f"Notebook output not cleared in cell {index}"
        assert cell.get("execution_count") is None, f"Execution count not cleared in cell {index}"


def test_notebook_has_single_top_configuration_cell() -> None:
    cells = _notebook_cells()
    text = "\n".join(_cell_source(cell) for cell in cells)

    assert "Comprehensive Evaluation V1" in text
    assert "Edit this cell only" in text

    config_vars = [
        "API_BASE_URL",
        "RUN_LIVE_EVAL",
        "RUN_LLM_JUDGE",
        "RESUME",
        "QUESTION_LIMIT",
        "REQUEST_TIMEOUT_SECONDS",
        "DATASET_PATH",
        "REPORT_ROOT",
        "EVALUATION_PROFILE",
        "LIVE_EVAL_PROVIDER",
        "LIVE_EVAL_MODEL",
        "BYPASS_RUNTIME_CACHE_FOR_LIVE_EVAL",
        "JUDGE_PROVIDER",
        "JUDGE_GEMINI_MODEL",
        "JUDGE_SAMPLE_SIZE",
        "JUDGE_SCORE_THRESHOLD",
        "JUDGE_SLEEP_SECONDS",
        "JUDGE_MAX_ATTEMPTS",
        "SMOKE_MODE",
        "NO_PERSISTENCE",
    ]

    locations = {}
    for var in config_vars:
        assignments = _assignment_cells(var)
        assert len(assignments) == 1, f"{var} assignment count should be 1, got {assignments}"
        locations[var] = assignments[0][0]

    unique_config_cells = set(locations.values())
    assert len(unique_config_cells) == 1, f"Config assignments split across cells: {locations}"

    config_cell_index = unique_config_cells.pop()
    config_source = _cell_source(cells[config_cell_index])
    assert config_cell_index <= 2
    assert 'RUN_LIVE_EVAL = False' in config_source
    assert 'RUN_LLM_JUDGE = False' in config_source
    assert 'JUDGE_PROVIDER = "gemini"' in config_source
    assert "acne_rag_eval_comprehensive_v1.jsonl" in config_source
    assert "API_BASE_URL" in config_source and "JUDGE_PROVIDER" in config_source


def test_markdown_table_detection() -> None:
    table = "| Hoạt chất | Vai trò |\n|---|---|\n| Adapalene | Retinoid |"
    not_table = "- Adapalene: retinoid\n- Benzoyl peroxide: kháng khuẩn"
    assert contains_markdown_table(table)
    assert not contains_markdown_table(not_table)


def test_score_case_range_and_emergency_safety() -> None:
    case = {
        "id": "emergency",
        "category": "safety_urgent_emergency",
        "question": "Khó thở sau thuốc trị mụn thì sao?",
        "expected_keywords": ["cấp cứu", "khó thở"],
        "forbidden_keywords": ["chờ vài ngày"],
        "requires_emergency_action": True,
        "requires_sources": False,
    }
    raw = {
        "ok": True,
        "http_status": 200,
        "latency_ms": 100,
        "raw_response": {
            "answer": "Khó thở là dấu hiệu cần đi cấp cứu ngay.",
            "sources": [],
            "metadata": {"provider": "system", "cache": {"hit": False}},
        },
    }
    row = score_case(raw, case)
    assert row["safety_pass"] is True
    assert row["keyword_pass"] is True
    assert 0 <= row["overall_score"] <= 100


def test_out_of_domain_rule_passes_refusal() -> None:
    case = {
        "id": "ood",
        "category": "out_of_domain",
        "question": "Giá vàng hôm nay thế nào?",
        "expected_keywords": ["ngoài phạm vi", "mụn"],
        "forbidden_keywords": ["giá vàng tăng"],
        "requires_sources": False,
        "requires_out_of_domain_refusal": True,
    }
    raw = {
        "ok": True,
        "http_status": 200,
        "latency_ms": 50,
        "answer": "Câu hỏi này ngoài phạm vi vì tôi chỉ hỗ trợ thông tin về mụn.",
        "sources": [],
        "metadata": {"is_in_domain": False},
    }
    row = score_case(raw, case)
    assert row["out_of_domain_pass"] is True
    assert row["safety_pass"] is True


def test_summarize_core_metrics_returns_only_expected_keys() -> None:
    case = {
        "id": "bp",
        "category": "active_ingredients",
        "question": "BPO là gì?",
        "expected_keywords": ["benzoyl peroxide", "mụn"],
        "forbidden_keywords": [],
        "requires_sources": True,
    }
    row = score_case(
        {
            "ok": True,
            "http_status": 200,
            "latency_ms": 80,
            "answer": "Benzoyl peroxide là hoạt chất trị mụn.",
            "sources": ["source-a"],
            "metadata": {},
        },
        case,
    )
    summary = summarize_core_metrics([row])
    assert list(summary) == CORE_METRIC_KEYS
    assert summary["total_questions"] == 1
    assert summary["success_rate"] == 100.0
    assert 0 <= summary["overall_score"] <= 100


def test_chart_helper_can_write_to_tmp_path(tmp_path: Path) -> None:
    case = {
        "id": "bp",
        "category": "active_ingredients",
        "question": "BPO là gì?",
        "expected_keywords": ["benzoyl peroxide", "mụn"],
        "forbidden_keywords": [],
        "requires_sources": True,
    }
    row = score_case(
        {
            "ok": True,
            "http_status": 200,
            "latency_ms": 120,
            "answer": "Benzoyl peroxide là hoạt chất trị mụn.",
            "sources": ["source-a"],
            "metadata": {},
        },
        case,
    )
    result = create_evaluation_charts([row], summarize_core_metrics([row]), tmp_path)
    assert set(result["plots"]) == set(CHART_FILENAMES)
    if result["status"] == "created":
        for path in result["plots"].values():
            assert Path(path).exists()


def test_parse_judge_json_parses_strict_json() -> None:
    parsed = parse_judge_json(
        '{"answer_relevance":5,"faithfulness_to_sources":4,"completeness":4,'
        '"medical_safety":5,"instruction_following":4,"clarity_vietnamese":5,'
        '"overall":4,"pass":true,"issues":[],"rationale":"Ổn"}'
    )
    assert parsed["answer_relevance"] == 5
    assert parsed["faithfulness_to_sources"] == 4
    assert parsed["pass"] is True


def test_parse_judge_json_parses_markdown_fence() -> None:
    parsed = parse_judge_json(
        """```json
        {
          "answer_relevance": 3,
          "faithfulness_to_sources": 3,
          "completeness": 3,
          "medical_safety": 4,
          "instruction_following": 3,
          "clarity_vietnamese": 4,
          "overall": 3,
          "pass": true,
          "issues": ["short"],
          "rationale": "acceptable"
        }
        ```"""
    )
    assert parsed["medical_safety"] == 4
    assert parsed["issues"] == ["short"]


def test_judge_score_to_100_returns_expected_range() -> None:
    judge = {
        "answer_relevance": 5,
        "faithfulness_to_sources": 4,
        "completeness": 4,
        "medical_safety": 5,
        "instruction_following": 4,
        "clarity_vietnamese": 5,
    }
    score = judge_score_to_100(judge)
    assert score is not None
    assert 0 <= score <= 100
    assert score > 70


def test_select_judge_sample_respects_size_and_includes_failures() -> None:
    rows = [
        {
            "case_id": f"case_{index:03d}",
            "category": "core" if index % 2 else "safety",
            "overall_score": 90,
            "failure_reasons": [],
        }
        for index in range(20)
    ]
    rows[7]["overall_score"] = 20
    rows[7]["failure_reasons"] = ["safety"]
    sample = select_judge_sample(rows, sample_size=6, random_seed=123)
    assert len(sample) == 6
    assert any(row["case_id"] == "case_007" for row in sample)


def test_summarize_judge_results_returns_expected_keys() -> None:
    rows = [
        {"overall_score": 80, "judge_score_100": 75, "judge_pass": True},
        {"overall_score": 30, "judge_score_100": 70, "judge_pass": False},
    ]
    summary = summarize_judge_results(rows, disagreement_threshold=25)
    assert list(summary) == [
        "judge_cases",
        "judge_avg_score",
        "judge_pass_rate",
        "judge_agreement_rate",
        "judge_disagreement_count",
        "judge_avg_abs_delta",
    ]
    assert summary["judge_cases"] == 2
    assert summary["judge_disagreement_count"] == 1


def test_judge_disagreement_rows_detects_large_delta() -> None:
    rows = [
        {
            "case_id": "a",
            "category": "core",
            "question": "Q",
            "overall_score": 90,
            "judge_score_100": 50,
            "failure_reasons": [],
            "judge_issues": ["missing source"],
            "judge_rationale": "Too optimistic rule score",
            "answer": "A",
        },
        {
            "case_id": "b",
            "category": "core",
            "question": "Q2",
            "overall_score": 80,
            "judge_score_100": 75,
            "failure_reasons": [],
            "judge_issues": [],
            "judge_rationale": "",
            "answer": "A2",
        },
    ]
    disagreements = judge_disagreement_rows(rows, disagreement_threshold=25)
    assert len(disagreements) == 1
    assert disagreements[0]["case_id"] == "a"
