from __future__ import annotations

import ast
import asyncio
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import evaluation.build_formal_package as package_builder
import evaluation.formal_evaluation_support as evaluation_support
from evaluation.create_formal_notebook import build_notebook
from evaluation.formal_evaluation_support import (
    BASELINE_RESULTS_DIR,
    BENCHMARK_PATH,
    CALIBRATION_BLOCKED,
    CALIBRATION_PATH,
    CALIBRATION_READY,
    CALIBRATION_RESULTS_PATH,
    CALIBRATION_REVIEW_REQUIRED,
    CASE_METRICS_PATH,
    CHECKING_REASONING_EFFORT,
    EVALUATOR_MODEL,
    EXPECTED_KB_BUILD_ID,
    EXPECTED_PIPELINE_FINGERPRINT,
    EvaluationBlocked,
    EXTRACTION_REASONING_EFFORT,
    MANIFEST_PATH,
    METRICS_SUMMARY_PATH,
    POST_IMPROVEMENT_PATHS,
    POST_IMPROVEMENT_RUN_ID,
    RAW_RESULTS_PATH,
    SYSTEM_UNDER_TEST_SHA,
    atomic_write_json,
    canonical_json_file_sha256,
    canonical_json_sha256,
    _checker_claim_triplets,
    _checker_label,
    _extraction_assessment,
    _ratio_to_percent,
    _validate_aggregate_percent,
    build_openai_batch_adapter,
    build_baseline_comparison,
    build_evaluation_run_paths,
    evaluate_calibration_runs,
    export_metrics,
    load_evaluation_artifacts,
    load_json,
    negative_rejection_rate,
    require_complete_formal_run,
    run_calibration_once,
    run_formal_cases,
    save_calibration_results,
    score_ragchecker,
    validate_benchmark,
    validate_system_under_test,
    write_pretty_json,
)
EXPECTED_CATEGORIES = {
    "adverse_effects_precautions": 9,
    "comparison_integration": 10,
    "definition_classification": 10,
    "explicit_topic_switch": 3,
    "follow_up_continuity": 4,
    "mechanism_role": 8,
    "pronoun_coreference": 5,
    "referral_care_seeking": 6,
    "repeated_question_history_isolation": 3,
    "treatment_use_combination": 12,
    "unsupported_absolute_certainty_guarantee": 10,
    "unsupported_comparison_relationship_specificity": 10,
    "unsupported_exact_quantity_time": 10,
}


def test_formal_benchmark_contract_and_hash() -> None:
    benchmark, manifest, calibration = load_evaluation_artifacts()

    report = validate_benchmark(benchmark, manifest, calibration)

    assert report["total"] == 100
    assert report["answerable"] == 70
    assert report["evidence_gap"] == 30
    assert report["family_counts"] == {
        "answerable_multi_turn": 15,
        "answerable_single_turn": 55,
        "evidence_gap": 30,
    }
    assert report["category_counts"] == EXPECTED_CATEGORIES
    assert manifest["benchmark_sha256"] == canonical_json_file_sha256(BENCHMARK_PATH)
    assert manifest["benchmark_sha256"] == "f61d6807c0ce39f902936844d810562c486f1bcadaa57a8a2da0e460ad7e534b"
    assert manifest["evaluation_base_sha"] == "6a1809c4ddedbccab986ec76eb730321686ff3ff"
    assert manifest["active_kb_build_id"] == "94d613bc9b33628de3ef"
    assert manifest["researcher_review_status"] == "pending"
    assert manifest["ragchecker"]["per_case_metric_scale"] == "ratio_0_1"
    assert manifest["ragchecker"]["aggregate_metric_scale"] == "percent_0_100"
    assert manifest["ragchecker"]["headline_reporting_scale"] == "percent_0_100"
    assert benchmark["researcher_review_status"] == "pending"
    assert benchmark["anti_contamination"]["internal_query_near_duplicates_at_or_above_0_80"] == 0
    assert benchmark["anti_contamination"]["manual_pattern_matches_at_or_above_0_86"] == 0
    assert benchmark["anti_contamination"]["registry_source_paths"] == ["tests", "docs", "scripts"]
    assert benchmark["anti_contamination"]["repeated_gold_claim_sets"] >= 0


def test_answerable_gold_provenance_is_self_contained() -> None:
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))

    for case in benchmark["cases"]:
        if not case["family"].startswith("answerable"):
            continue
        provenance = case["provenance"]
        provenance_by_id = {item["chunk_id"]: item for item in provenance}
        provenance_ids = set(provenance_by_id)
        assert provenance_ids
        assert len(provenance_by_id) == len(provenance)
        for item in provenance:
            assert item["source_id"]
            assert item["source_title"]
            assert item["source_authority"]
            assert item["source_url"]
            assert item["excerpt"]
        for claim in case["gold_claims"]:
            assert set(claim["source_chunk_ids"]) <= provenance_ids
            assert claim["annotation_status"] == "source_annotated_pending_researcher_review"
            assert claim["evidence_snippets"]
            for snippet in claim["evidence_snippets"]:
                assert snippet["chunk_id"] in claim["source_chunk_ids"]
                assert evaluation_support._normalize(snippet["text"])
        assert case["gold_answer"] == " ".join(claim["text"] for claim in case["gold_claims"])


def test_package_builder_validates_evidence_snippets_against_source_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim_key = "synthetic_claim"
    claim = {"chunks": ["chunk-1"]}
    records = {"chunk-1": {"text": "The source supports this evidence claim."}}
    monkeypatch.setitem(
        package_builder.CLAIM_EVIDENCE,
        claim_key,
        [("chunk-1", "supports this evidence")],
    )

    assert package_builder._claim_evidence_snippets(claim_key, claim, records) == [
        {"chunk_id": "chunk-1", "text": "supports this evidence"}
    ]

    monkeypatch.setitem(
        package_builder.CLAIM_EVIDENCE,
        claim_key,
        [("chunk-1", "unsupported evidence")],
    )
    with pytest.raises(RuntimeError, match="Evidence snippet is not present in frozen chunk"):
        package_builder._claim_evidence_snippets(claim_key, claim, records)


def test_evidence_gap_audits_are_corpus_wide_and_read_only() -> None:
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    gaps = [case for case in benchmark["cases"] if case["family"] == "evidence_gap"]

    assert len(gaps) == 30
    for case in gaps:
        audit = case["absence_verification"]
        assert case["expected"] == {"action": "abstain", "reason": "evidence_gap"}
        assert audit["scope"] == "entire_active_frozen_corpus"
        assert audit["lexical_corpus_scan"]["records_scanned"] == 512
        assert audit["candidate_search_completed"] is True
        assert audit["absence_review_status"] == "pending_researcher_review"
        assert audit["unsupported_factual_requirement"]
        assert audit["bm25_candidate_search"]["engine"] == "qdrant_native_bm25_read_only"
        assert len(audit["bm25_candidate_search"]["candidate_chunk_ids"]) == 20
        assert audit["bm25_candidate_search"]["review_candidates"]
        assert audit["related_topic_dense_probe"]["engine"] == "qdrant_cosine_read_only"
        assert audit["related_topic_dense_probe"]["probe_method"] == "cached_related_source_chunk_vector_not_query_embedding"
        assert len(audit["related_topic_dense_probe"]["candidate_chunk_ids"]) == 20
        assert audit["related_topic_dense_probe"]["review_candidates"]
        assert (
            audit["related_topic_dense_probe"]["cached_vectors_available"]
            + audit["related_topic_dense_probe"]["cached_vectors_missing"]
            == 512
        )
        assert audit["provider_calls"] == 0
        assert audit["datastore_writes"] == 0


def test_calibration_matrix_is_predeclared_and_balanced() -> None:
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    extraction = calibration["claim_extraction"]
    checking = calibration["claim_checking"]

    assert Counter(item["category"] for item in extraction) == {
        "simple_factual": 2,
        "qualifier_sensitive": 2,
        "multi_clause_treatment": 2,
        "comparison_mixed_terminology": 2,
    }
    assert Counter(item["expected_label"] for item in checking) == {
        "SUPPORTED": 6,
        "NOT_SUPPORTED": 6,
    }
    assert Counter(item["language_direction"] for item in checking) == {
        "vi_to_vi": 6,
        "en_to_vi": 6,
    }
    vietnamese_marks = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
    assert all(
        any(char.casefold() in vietnamese_marks for char in item["evidence"])
        for item in checking
        if item["language_direction"] == "vi_to_vi"
    )
    dimensions = {dimension for item in checking for dimension in item["critical_dimensions"]}
    assert {
        "negation",
        "qualifier_modality",
        "exact_number",
        "exact_time",
        "unsupported_comparison",
        "entity_distinction",
        "cross_language_entailment",
    } <= dimensions
    assert all(item["reference_claims"] for item in extraction)
    assert all(
        item["claim_triplets"]
        and all(
            isinstance(triplet, list)
            and len(triplet) == 3
            and all(isinstance(part, str) and part.strip() for part in triplet)
            for triplet in item["claim_triplets"]
        )
        for item in checking
    )
    assert all(
        len(item["reference_claims"]) > 1
        for item in extraction
        if item["item_id"] in {"CAL-EXT-05", "CAL-EXT-06", "CAL-EXT-07", "CAL-EXT-08"}
    )


def test_notebook_is_unexecuted_and_keeps_manual_gate_closed() -> None:
    notebook_path = Path("evaluation/formal_evaluation.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    expected = build_notebook()

    assert notebook == expected
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 5
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code_cells)
    all_source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "RUN_AUTHORIZED = False" in all_source
    assert "chỉ cho phép thực hiện lần đánh giá" in all_source
    assert "không có nghĩa toàn bộ benchmark hoặc calibration" in all_source
    assert "CALIBRATION_BLOCKED" in all_source
    assert "CALIBRATION_REVIEW_REQUIRED" in all_source
    assert CALIBRATION_BLOCKED == "BLOCKED_BY_EVALUATOR_CALIBRATION"
    assert "## Thuật ngữ" in all_source
    assert "### Bảng đối chiếu 30 trường hợp thiếu bằng chứng" not in all_source
    assert "evidence_gap_review_rows" not in all_source
    assert "gpt-5.4-mini-2026-03-17" not in all_source  # Imported from the fixed helper contract.
    assert "Phiên bản hệ thống: {manifest['evaluation_base_sha']}" not in all_source
    assert "Mốc tham chiếu của bộ đánh giá" in all_source
    assert "Hệ thống được đánh giá" in all_source
    assert "Pipeline fingerprint kỳ vọng" in all_source
    assert "Post-improvement score (%)" in all_source
    assert "So sánh với Formal Run baseline" in all_source
    assert "post_improvement_47b10954" not in all_source  # Imported from the fixed helper contract.
    for cell in code_cells:
        compile("".join(cell["source"]), "<notebook-cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def test_formal_outputs_are_not_precreated_or_tracked() -> None:
    assert MANIFEST_PATH.is_file()
    assert BENCHMARK_PATH.is_file()
    assert CALIBRATION_PATH.is_file()
    for path in (RAW_RESULTS_PATH, CASE_METRICS_PATH, METRICS_SUMMARY_PATH):
        assert not path.exists()
    tracked = subprocess.run(
        ["git", "ls-files", "--", str(RAW_RESULTS_PATH), str(CASE_METRICS_PATH), str(METRICS_SUMMARY_PATH), str(CALIBRATION_RESULTS_PATH)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert tracked == ""


def test_post_improvement_output_paths_are_isolated_from_baseline() -> None:
    paths = build_evaluation_run_paths(POST_IMPROVEMENT_RUN_ID)
    writable = {
        paths.raw_results,
        paths.case_metrics,
        paths.metrics_summary,
        paths.calibration_results,
        paths.ragchecker_checkpoint,
    }

    assert paths == POST_IMPROVEMENT_PATHS
    assert all(path.parent == paths.directory for path in writable)
    assert paths.directory.parent == BASELINE_RESULTS_DIR.parent
    assert paths.directory != BASELINE_RESULTS_DIR
    assert all(BASELINE_RESULTS_DIR not in path.parents for path in writable)


@pytest.mark.parametrize(
    ("production_diff", "should_pass"),
    [
        ("", True),
        ("src/agent/graph.py", False),
    ],
)
def test_system_under_test_validation_blocks_only_production_sensitive_diff(
    monkeypatch: pytest.MonkeyPatch,
    production_diff: str,
    should_pass: bool,
) -> None:
    manifest = load_json(MANIFEST_PATH)

    def fake_git(*args: str) -> str:
        if args[:2] == ("rev-parse", "HEAD"):
            return "evaluation-only-head"
        if args[:2] == ("diff", "--name-only"):
            assert "evaluation" not in args
            assert "tests" not in args
            return production_diff
        raise AssertionError(args)

    monkeypatch.setattr(evaluation_support, "_git", fake_git)
    monkeypatch.setattr(
        evaluation_support.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )

    if should_pass:
        report = validate_system_under_test(manifest)
        assert report["system_under_test_is_ancestor"] is True
        assert report["production_diff_after_system_under_test"] == []
    else:
        with pytest.raises(EvaluationBlocked, match="System under test"):
            validate_system_under_test(manifest)


def test_system_under_test_validation_fails_when_checkpoint_is_not_ancestor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluation_support,
        "_git",
        lambda *args: "head" if args[:2] == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        evaluation_support.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )

    with pytest.raises(EvaluationBlocked, match="System under test"):
        validate_system_under_test(load_json(MANIFEST_PATH))


def test_baseline_comparison_reads_csv_and_calculates_descriptive_delta(tmp_path: Path) -> None:
    baseline = tmp_path / "metrics_summary.csv"
    with baseline.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Metric", "N cases", "Score"])
        writer.writeheader()
        writer.writerow({"Metric": "Claim Recall", "N cases": 70, "Score": 60.0})
    post = [{"Metric": "Claim Recall", "N cases": 70, "Score": 62.5}]

    assert build_baseline_comparison(post, baseline) == [{
        "Metric": "Claim Recall",
        "N": 70,
        "Formal Run baseline (%)": 60.0,
        "Post-improvement (%)": 62.5,
        "Chênh lệch điểm %": 2.5,
    }]
    assert build_baseline_comparison(post, tmp_path / "missing.csv") is None


def test_research_json_is_pretty_and_identity_is_format_independent(tmp_path: Path) -> None:
    for path in (BENCHMARK_PATH, MANIFEST_PATH, CALIBRATION_PATH):
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert "\n  \"" in text
        assert load_json(path)

    benchmark = load_json(BENCHMARK_PATH)
    pretty_path = tmp_path / "pretty.json"
    compact_path = tmp_path / "compact.json"
    write_pretty_json(pretty_path, benchmark)
    compact_path.write_text(
        json.dumps(benchmark, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    expected = "f61d6807c0ce39f902936844d810562c486f1bcadaa57a8a2da0e460ad7e534b"
    assert canonical_json_sha256(benchmark) == expected
    assert canonical_json_file_sha256(pretty_path) == expected
    assert canonical_json_file_sha256(compact_path) == expected


def test_nrr_uses_structured_action_and_reason_only() -> None:
    records = [
        {
            "case_id": f"GAP-{index:03d}",
            "case_family": "evidence_gap",
            "actual_action": "abstain" if index <= 27 else "generate",
            "actual_reason": "evidence_gap" if index != 27 else "other_reason",
        }
        for index in range(1, 31)
    ]

    score, correct = negative_rejection_rate({"records": records})

    assert correct == 26
    assert score == pytest.approx((26 / 30) * 100)


def test_atomic_extraction_accepts_paraphrase_but_rejects_merged_claims() -> None:
    calibration = load_json(CALIBRATION_PATH)
    item = next(row for row in calibration["claim_extraction"] if row["item_id"] == "CAL-EXT-05")

    merged_status, merged_reasons = _extraction_assessment(item, [item["input_text"]])
    paraphrase_status, paraphrase_reasons = _extraction_assessment(
        item,
        [
            "Kháng sinh bôi không nên dùng một mình.",
            "Benzoyl peroxide giúp giảm nguy cơ kháng kháng sinh.",
        ],
    )

    assert merged_status == "rejected"
    assert "atomicity:independent_references_merged" in merged_reasons
    assert paraphrase_status == "accepted"
    assert paraphrase_reasons == []


@pytest.mark.parametrize(
    ("item_id", "claims"),
    [
        (
            "CAL-EXT-01",
            [
                "Mụn trứng cá là bệnh viêm mạn tính.",
                "Mụn trứng cá thuộc đơn vị nang lông tuyến bã.",
            ],
        ),
        (
            "CAL-EXT-02",
            [
                "Benzoyl peroxide is an antibacterial intended for topical application.",
                "Benzoyl peroxide has a mild comedolytic action.",
            ],
        ),
        (
            "CAL-EXT-03",
            [
                "Evidence concerning a low glycemic load is conflicting.",
                "A low glycemic load suggests a possible reduction of acne.",
            ],
        ),
        (
            "CAL-EXT-04",
            [
                "Maintenance therapy is not invariably required.",
                "Maintenance treatment may be considered for recurrent episodes.",
            ],
        ),
    ],
)
def test_extraction_accepts_distributed_and_multilingual_semantic_equivalence(
    item_id: str,
    claims: list[str],
) -> None:
    calibration = load_json(CALIBRATION_PATH)
    item = next(row for row in calibration["claim_extraction"] if row["item_id"] == item_id)

    assert _extraction_assessment(item, claims) == ("accepted", [])


def test_extraction_preserves_negation_and_entity_ownership() -> None:
    calibration = load_json(CALIBRATION_PATH)
    items = {row["item_id"]: row for row in calibration["claim_extraction"]}

    accepted = _extraction_assessment(
        items["CAL-EXT-08"],
        [
            "Benzoyl peroxide is antimicrobial.",
            "Benzoyl peroxide is not a topical antibiotic.",
            "Clindamycin is a topical antibiotic.",
        ],
    )
    wrong_polarity = _extraction_assessment(
        items["CAL-EXT-08"],
        [
            "Benzoyl peroxide is antimicrobial.",
            "Benzoyl peroxide is a topical antibiotic.",
            "Clindamycin is a topical antibiotic.",
        ],
    )
    wrong_entity = _extraction_assessment(
        items["CAL-EXT-02"],
        [
            "Clindamycin is an antibacterial intended for topical application.",
            "Benzoyl peroxide has a mild comedolytic action.",
        ],
    )

    assert accepted == ("accepted", [])
    assert wrong_polarity[0] == "rejected"
    assert "missing_qualifier:R02" in wrong_polarity[1]
    assert wrong_entity[0] == "review_required"
    assert "missing_reference:R01" in wrong_entity[1]


def test_extraction_detects_actual_missing_reference() -> None:
    calibration = load_json(CALIBRATION_PATH)
    item = next(row for row in calibration["claim_extraction"] if row["item_id"] == "CAL-EXT-05")

    status, reasons = _extraction_assessment(item, ["Kháng sinh bôi không nên dùng một mình."])

    assert status == "review_required"
    assert "missing_reference:R02" in reasons


@pytest.mark.parametrize("item_id", ["CAL-EXT-03", "CAL-EXT-05"])
def test_extraction_bilingual_matching_does_not_hide_genuine_omissions(item_id: str) -> None:
    calibration = load_json(CALIBRATION_PATH)
    item = next(row for row in calibration["claim_extraction"] if row["item_id"] == item_id)

    status, reasons = _extraction_assessment(item, [item["reference_claims"][1]["text"]])

    assert status == "review_required"
    assert "missing_reference:R01" in reasons


def test_extraction_bilingual_matching_rejects_missing_maintenance_claim() -> None:
    calibration = load_json(CALIBRATION_PATH)
    item = next(row for row in calibration["claim_extraction"] if row["item_id"] == "CAL-EXT-04")

    status, reasons = _extraction_assessment(
        item,
        ["Maintenance treatment may be considered for frequent recurrence."],
    )

    assert status == "review_required"
    assert "missing_reference:R01" in reasons


def test_atomic_extraction_detects_missing_qualifier_and_invention() -> None:
    calibration = load_json(CALIBRATION_PATH)
    item = next(row for row in calibration["claim_extraction"] if row["item_id"] == "CAL-EXT-03")
    first_reference = item["reference_claims"][0]["text"]

    missing_status, missing_reasons = _extraction_assessment(
        item,
        [first_reference, "Chế độ ăn tải đường huyết thấp giảm mụn."],
    )
    invention_status, invention_reasons = _extraction_assessment(
        item,
        [
            first_reference,
            item["reference_claims"][1]["text"] + " Chế độ này chắc chắn chữa khỏi mụn.",
        ],
    )

    assert missing_status == "rejected"
    assert "missing_qualifier:R02" in missing_reasons
    assert invention_status == "rejected"
    assert any(reason.startswith("forbidden_invention:") for reason in invention_reasons)


def test_unresolved_extraction_paraphrase_requires_researcher_review() -> None:
    calibration = load_json(CALIBRATION_PATH)
    item = next(row for row in calibration["claim_extraction"] if row["item_id"] == "CAL-EXT-01")

    status, reasons = _extraction_assessment(
        item,
        ["Tình trạng này là một tiến trình lâu dài có viêm ở cấu trúc sinh lông và tiết dầu."],
    )

    assert status == "review_required"
    assert reasons == ["missing_reference:R01"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Entailment", "SUPPORTED"),
        ("Contradiction", "NOT_SUPPORTED"),
        ("Neutral", "NOT_SUPPORTED"),
        ("Supported", "SUPPORTED"),
        ("Not supported", "NOT_SUPPORTED"),
        (["Entailment", "Entailment"], "SUPPORTED"),
        (["Entailment", "Neutral"], "NOT_SUPPORTED"),
        (["Entailment", "Contradiction"], "NOT_SUPPORTED"),
        (["Entailment", "unknown"], "UNPARSEABLE"),
        ("unexpected text", "UNPARSEABLE"),
    ],
)
def test_checker_label_parser_uses_exact_normalized_contract(raw: str, expected: str) -> None:
    assert _checker_label(raw) == expected


def test_calibration_checker_uses_atomic_triplets_without_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = load_json(CALIBRATION_PATH)
    extraction_by_id = {item["item_id"]: item for item in calibration["claim_extraction"]}
    checking_by_id = {item["item_id"]: item for item in calibration["claim_checking"]}
    captured_triplets: dict[str, list[list[str]]] = {}

    class FakeEvaluator:
        def extract_claims(self, results, extract_type: str) -> None:
            assert extract_type == "gt_answer"
            for result in results:
                item = extraction_by_id[result.query_id]
                result.gt_answer_claims = [reference["text"] for reference in item["reference_claims"]]

        def check_claims(self, results, check_type: str) -> None:
            assert check_type == "answer2response"
            for result in results.results:
                item = checking_by_id[result.query_id]
                captured_triplets[result.query_id] = result.response_claims
                label = "Entailment" if item["expected_label"] == "SUPPORTED" else "Neutral"
                result.answer2response = [label] * len(result.response_claims)

    class FakeRAGResult:
        def __init__(self, **values) -> None:
            self.__dict__.update(values)

    class FakeRAGResults:
        def __init__(self, *, results) -> None:
            self.results = results

    def provider_must_not_run(_prompts: list[str]) -> list[str]:
        pytest.fail("offline calibration regression attempted to call the provider adapter")

    monkeypatch.setitem(
        sys.modules,
        "ragchecker",
        SimpleNamespace(RAGResult=FakeRAGResult, RAGResults=FakeRAGResults),
    )
    monkeypatch.setattr(evaluation_support, "_make_ragchecker", lambda _adapter: FakeEvaluator())

    result = run_calibration_once(calibration, provider_must_not_run)

    assert len(result["extraction"]) == 8
    assert all(row["agreement"] for row in result["checking"])
    assert captured_triplets == {
        item["item_id"]: _checker_claim_triplets(item)
        for item in calibration["claim_checking"]
    }


def test_calibration_checker_rejects_malformed_triplets() -> None:
    item = {"item_id": "CAL-BAD", "claim_triplets": [["subject", "predicate"]]}

    with pytest.raises(ValueError, match="requires atomic claim_triplets"):
        _checker_claim_triplets(item)


@pytest.mark.parametrize(
    ("item_id", "dimension"),
    [
        ("CAL-CHK-04", "exact_number"),
        ("CAL-CHK-10", "unsupported_comparison"),
        ("CAL-CHK-12", "unsupported_comparison"),
    ],
)
def test_negative_checker_cases_preserve_critical_dimensions(item_id: str, dimension: str) -> None:
    calibration = load_json(CALIBRATION_PATH)
    item = next(row for row in calibration["claim_checking"] if row["item_id"] == item_id)

    assert item["expected_label"] == "NOT_SUPPORTED"
    assert dimension in item["critical_dimensions"]


def _clean_calibration_runs() -> tuple[dict, dict]:
    extraction = [
        {"item_id": f"EXT-{index}", "status": "accepted", "acceptable": True, "reasons": [], "claims": ["claim"]}
        for index in range(8)
    ]
    checking = [
        {
            "item_id": f"CHK-{index}",
            "expected": "SUPPORTED",
            "actual": "SUPPORTED",
            "agreement": True,
            "critical_dimensions": ["negation"],
        }
        for index in range(12)
    ]
    payload = {"extraction": extraction, "checking": checking}
    return json.loads(json.dumps(payload)), json.loads(json.dumps(payload))


def test_calibration_decision_distinguishes_ready_review_and_blocked() -> None:
    first, second = _clean_calibration_runs()
    assert evaluate_calibration_runs(first, second)["decision"] == CALIBRATION_READY

    first, second = _clean_calibration_runs()
    second["checking"][0].update(actual="UNPARSEABLE", agreement=False)
    assert evaluate_calibration_runs(first, second)["decision"] == CALIBRATION_REVIEW_REQUIRED

    first, second = _clean_calibration_runs()
    first["checking"][0].update(actual="NOT_SUPPORTED", agreement=False)
    second["checking"][0].update(actual="NOT_SUPPORTED", agreement=False)
    assert evaluate_calibration_runs(first, second)["decision"] == CALIBRATION_BLOCKED


def test_metrics_export_normalizes_all_headlines_to_percent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = load_json(BENCHMARK_PATH)
    records = []
    rag_items = []
    for case in benchmark["cases"]:
        is_gap = case["family"] == "evidence_gap"
        is_single_nrr_miss = case["case_id"] == "GAP-ABS-010"
        records.append({
            "case_id": case["case_id"],
            "case_family": case["family"],
            "actual_action": "generate" if not is_gap or is_single_nrr_miss else "abstain",
            "actual_reason": None if not is_gap or is_single_nrr_miss else "evidence_gap",
            "infrastructure_error": None,
        })
        if not is_gap:
            rag_items.append(SimpleNamespace(
                query_id=case["case_id"],
                metrics={"claim_recall": 0.5, "context_precision": 0.6, "faithfulness": 0.7, "f1": 0.8},
            ))
    rag_results = SimpleNamespace(
        results=rag_items,
        metrics={
            "retriever_metrics": {"claim_recall": 50.0, "context_precision": 60.0},
            "generator_metrics": {"faithfulness": 70.0},
            "overall_metrics": {"f1": 80.0},
        },
    )
    monkeypatch.setattr(evaluation_support, "CASE_METRICS_PATH", tmp_path / "case_metrics.csv")
    monkeypatch.setattr(evaluation_support, "METRICS_SUMMARY_PATH", tmp_path / "metrics_summary.csv")

    case_rows, summary = export_metrics(benchmark, {"records": records}, rag_results)

    assert {row["Unit"] for row in summary} == {"percent_0_100"}
    assert [row["Score"] for row in summary[:4]] == [50.0, 60.0, 70.0, 80.0]
    assert summary[4]["Score"] == pytest.approx((29 / 30) * 100)
    assert next(row for row in case_rows if row["case_family"] != "evidence_gap")["Claim Recall (%)"] == 50.0


def test_ragchecker_metric_scales_are_explicit_and_validated() -> None:
    assert _ratio_to_percent(0.614) == pytest.approx(61.4)
    assert _validate_aggregate_percent(61.4) == pytest.approx(61.4)
    with pytest.raises(EvaluationBlocked, match="per-case metric outside ratio 0-1"):
        _ratio_to_percent(1.1)
    with pytest.raises(EvaluationBlocked, match="aggregate metric outside percent 0-100"):
        _validate_aggregate_percent(101)


def test_targeted_gold_mappings_are_minimal_and_query_aligned() -> None:
    benchmark = load_json(BENCHMARK_PATH)
    cases = {case["case_id"]: case for case in benchmark["cases"]}

    assert "bít hoàn toàn" not in cases["ANS-DEF-003"]["gold_answer"]
    assert "mặt hoặc thân mình" not in cases["ANS-DEF-002"]["gold_answer"]
    assert "kháng sinh uống" not in cases["ANS-TRT-003"]["gold_answer"]
    assert "không kê đơn" not in cases["ANS-TRT-009"]["gold_answer"]
    assert "không kê đơn" not in cases["ANS-MEC-001"]["gold_answer"]
    assert "rối loạn sắc tố" not in cases["ANS-MEC-002"]["gold_answer"]
    assert "duy trì" not in cases["ANS-MEC-002"]["gold_answer"]
    assert "không kê đơn" not in cases["ANS-MEC-006"]["gold_answer"]
    assert "bã nhờn" not in cases["ANS-MUL-002"]["gold_answer"]

    assert len(cases["ANS-DEF-010"]["gold_claims"]) == 1
    assert "khác" not in cases["ANS-DEF-010"]["query"].casefold()
    assert "kháng sinh" not in cases["ANS-CMP-003"]["gold_answer"].split(".")[0].casefold()
    assert "khác gì về thời gian" not in cases["ANS-CMP-005"]["query"].casefold()
    assert "lymecycline" in cases["ANS-CMP-007"]["gold_answer"].casefold()
    assert "doxycycline" in cases["ANS-CMP-007"]["gold_answer"].casefold()
    assert "cơ chế và hình thái" not in cases["ANS-CMP-010"]["query"].casefold()
    assert "không kê đơn" not in cases["ANS-CMP-004"]["gold_answer"]
    assert "retinoid" not in cases["ANS-MUL-012"]["gold_answer"].casefold()


def test_pronoun_history_identifies_clascoterone_without_leaking_target_answer() -> None:
    benchmark = load_json(BENCHMARK_PATH)
    case = next(case for case in benchmark["cases"] if case["case_id"] == "ANS-MUL-001")
    history = " ".join(message["content"] for message in case["history"]).casefold()

    assert "clascoterone" in history
    assert "chẹn androgen" not in history
    assert "kháng androgen" not in history
    assert "tác động lên androgen" not in history


def test_calibration_result_and_run_metadata_are_auditable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = load_json(CALIBRATION_PATH)
    first, second = _clean_calibration_runs()
    decision = evaluate_calibration_runs(first, second)
    output = tmp_path / "evaluator_calibration_results.json"
    monkeypatch.setattr(evaluation_support, "CALIBRATION_RESULTS_PATH", output)

    payload = save_calibration_results(calibration, first, second, decision)
    run_metadata = evaluation_support._formal_run_metadata(
        allow_model_fallback=True,
        run_authorized=True,
    )

    assert load_json(output) == payload
    assert payload["final_calibration_decision"] == CALIBRATION_READY
    assert payload["calibration_dataset_sha256"] == canonical_json_file_sha256(CALIBRATION_PATH)
    expected_request_configuration = {
        "extraction": {"model": EVALUATOR_MODEL, "reasoning_effort": "medium"},
        "checking": {"model": EVALUATOR_MODEL, "reasoning_effort": "low"},
    }
    assert payload["evaluator_request_configuration"] == expected_request_configuration
    assert run_metadata["evaluator_request_configuration"] == expected_request_configuration
    assert {
        "benchmark_sha256",
        "benchmark_reference_base_sha",
        "system_under_test_sha",
        "repository_head_at_run",
        "expected_pipeline_fingerprint",
        "run_id",
        "evaluation_base_sha",
        "current_git_head",
        "active_kb_build_id",
        "run_timestamp",
        "python_version",
        "ragchecker_version",
        "openai_package_version",
        "spacy_version",
        "spacy_model_identifier",
        "evaluator_model",
        "observed_production_requested_model_configuration",
        "observed_fallback_model_configuration",
        "allow_model_fallback",
        "execution_authorization_state",
        "run_authorized",
    } <= set(run_metadata)


def test_atomic_result_serialization_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    payload = {
        "benchmark_sha256": "abc",
        "records": [{"case_id": "CASE-001", "answer": "Tiếng Việt chuẩn"}],
    }

    atomic_write_json(path, payload)

    assert load_json(path) == payload


def test_evaluator_adapter_fails_closed_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(EvaluationBlocked, match="OPENAI_API_KEY"):
        build_openai_batch_adapter()


def _ragchecker_role_prompts() -> tuple[list[str], str]:
    # Stable task markers from the audited RefChecker prompts bundled with RAGChecker 0.1.9.
    extraction_prompts = [
        (
            "Given a question and a candidate answer to the question, please extract a KG "
            "from the candidate answer. Please note that this is an EXTRACTION task."
        ),
        (
            "Given a question and a response to the question, please extract a KG from the "
            "response. Please note that this is an EXTRACTION task."
        ),
    ]
    checking_prompt = (
        "I have a list of claims that made by a language model to a question, please help me "
        "for checking whether the claims can be entailed. Please DO NOT use your own knowledge."
    )
    return extraction_prompts, checking_prompt


def test_evaluator_adapter_routes_stage_specific_reasoning_without_changing_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    extraction_prompts, checking_prompt = _ragchecker_role_prompts()

    class FakeResponses:
        def create(self, **request):
            calls.append(request)
            return SimpleNamespace(output_text=f"result-{len(calls)}")

    class FakeOpenAI:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "offline-test-key"
            self.responses = FakeResponses()

    openai_module = ModuleType("openai")
    openai_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")

    adapter = build_openai_batch_adapter()
    extraction_outputs = adapter(extraction_prompts)
    checking_outputs = adapter([checking_prompt])

    assert extraction_outputs == ["result-1", "result-2"]
    assert checking_outputs == ["result-3"]
    assert [call["reasoning"] for call in calls] == [
        {"effort": EXTRACTION_REASONING_EFFORT},
        {"effort": EXTRACTION_REASONING_EFFORT},
        {"effort": CHECKING_REASONING_EFFORT},
    ]
    assert {call["model"] for call in calls} == {EVALUATOR_MODEL}
    assert all(call["store"] is False for call in calls)


@pytest.mark.parametrize("prompts", [[], ["unknown evaluator prompt"]])
def test_evaluator_adapter_fails_closed_for_unknown_prompt_batches(
    monkeypatch: pytest.MonkeyPatch,
    prompts: list[str],
) -> None:
    provider_calls = 0

    class FakeResponses:
        def create(self, **_request):
            nonlocal provider_calls
            provider_calls += 1
            return SimpleNamespace(output_text="must not be returned")

    class FakeOpenAI:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "offline-test-key"
            self.responses = FakeResponses()

    openai_module = ModuleType("openai")
    openai_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")

    with pytest.raises(EvaluationBlocked, match="RAGCHECKER_PROMPT"):
        build_openai_batch_adapter()(prompts)

    assert provider_calls == 0


def test_evaluator_adapter_fails_closed_for_mixed_prompt_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction_prompts, checking_prompt = _ragchecker_role_prompts()
    openai_module = ModuleType("openai")
    openai_module.OpenAI = lambda **_kwargs: SimpleNamespace(
        responses=SimpleNamespace(create=lambda **_request: pytest.fail("provider must not run"))
    )
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")

    with pytest.raises(EvaluationBlocked, match="MIXED_ROLES"):
        build_openai_batch_adapter()([extraction_prompts[0], checking_prompt])


def test_formal_scoring_keeps_official_ragchecker_metric_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeRAGResult:
        def __init__(self, **values) -> None:
            self.__dict__.update(values)
            self.gt_answer_claims = None
            self.response_claims = None

    class FakeRAGResults:
        def __init__(self, *, results) -> None:
            self.results = results

    class FakeRAGChecker:
        def __init__(self, **configuration) -> None:
            captured["configuration"] = configuration

        def evaluate(self, results, *, metrics, save_path) -> None:
            captured["metrics"] = metrics
            captured["save_path"] = save_path
            assert all(result.gt_answer_claims is None for result in results.results)
            assert all(result.response_claims is None for result in results.results)

    class FakeRetrievedDoc:
        def __init__(self, *, doc_id, text) -> None:
            self.doc_id = doc_id
            self.text = text

    ragchecker_module = ModuleType("ragchecker")
    ragchecker_module.RAGChecker = FakeRAGChecker
    ragchecker_module.RAGResult = FakeRAGResult
    ragchecker_module.RAGResults = FakeRAGResults
    container_module = ModuleType("ragchecker.container")
    container_module.RetrievedDoc = FakeRetrievedDoc
    metrics_module = ModuleType("ragchecker.metrics")
    metrics_module.claim_recall = "claim_recall"
    metrics_module.context_precision = "context_precision"
    metrics_module.faithfulness = "faithfulness"
    metrics_module.f1 = "f1"
    monkeypatch.setitem(sys.modules, "ragchecker", ragchecker_module)
    monkeypatch.setitem(sys.modules, "ragchecker.container", container_module)
    monkeypatch.setitem(sys.modules, "ragchecker.metrics", metrics_module)
    def adapter(prompts: list[str]) -> list[str]:
        return ["unused" for _ in prompts]

    score_ragchecker(
        {
            "cases": [{
                "case_id": "ANS-001",
                "family": "answerable_definition",
                "query": "Mụn là gì?",
                "gold_answer": "Mụn là bệnh viêm.",
                "gold_claims": [{"text": "Mụn là bệnh viêm."}],
            }],
        },
        {
            "records": [{
                "case_id": "ANS-001",
                "answer": "Mụn là bệnh viêm.",
                "packed_contexts": [{"chunk_id": "chunk-1", "text": "Mụn là bệnh viêm."}],
                "pipeline_fingerprint": EXPECTED_PIPELINE_FINGERPRINT,
            }],
        },
        adapter,
    )

    assert captured["metrics"] == ["claim_recall", "context_precision", "faithfulness", "f1"]
    assert captured["configuration"]["custom_llm_api_func"] is adapter


def _patch_formal_run_paths(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> evaluation_support.EvaluationRunPaths:
    paths = evaluation_support.EvaluationRunPaths(
        run_id=POST_IMPROVEMENT_RUN_ID,
        directory=root / POST_IMPROVEMENT_RUN_ID,
        calibration_results=root / POST_IMPROVEMENT_RUN_ID / "evaluator_calibration_results.json",
        raw_results=root / POST_IMPROVEMENT_RUN_ID / "raw_results.json",
        case_metrics=root / POST_IMPROVEMENT_RUN_ID / "case_metrics.csv",
        metrics_summary=root / POST_IMPROVEMENT_RUN_ID / "metrics_summary.csv",
        ragchecker_checkpoint=root / POST_IMPROVEMENT_RUN_ID / "ragchecker_checkpoint.json",
    )
    monkeypatch.setattr(evaluation_support, "RAW_RESULTS_PATH", paths.raw_results)
    monkeypatch.setattr(evaluation_support, "CALIBRATION_RESULTS_PATH", paths.calibration_results)
    monkeypatch.setattr(evaluation_support, "CASE_METRICS_PATH", paths.case_metrics)
    monkeypatch.setattr(evaluation_support, "METRICS_SUMMARY_PATH", paths.metrics_summary)
    monkeypatch.setattr(evaluation_support, "RAGCHECKER_CHECKPOINT_PATH", paths.ragchecker_checkpoint)
    return paths


def _synthetic_formal_case() -> dict:
    return {
        "case_id": "SYNTHETIC-001",
        "family": "answerable_single_turn",
        "category": "synthetic",
        "query": "Synthetic offline query",
        "history": [],
        "expected": {"action": "generate", "reason": None},
    }


def _install_fake_agent(monkeypatch: pytest.MonkeyPatch, agent) -> None:
    graph_module = ModuleType("src.agent.graph")
    graph_module.run_clinical_agent = agent
    monkeypatch.setitem(sys.modules, "src.agent.graph", graph_module)


def test_post_improvement_execution_writes_only_to_its_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _patch_formal_run_paths(monkeypatch, tmp_path)
    provider_calls = 0

    async def fake_agent(**_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return {
            "answer": "Synthetic answer",
            "agent_decision": {"action": "generate", "reason_code": None},
            "pipeline_fingerprint": EXPECTED_PIPELINE_FINGERPRINT,
        }

    _install_fake_agent(monkeypatch, fake_agent)
    benchmark_sha = canonical_json_file_sha256(BENCHMARK_PATH)
    result = asyncio.run(
        run_formal_cases(
            {"cases": [_synthetic_formal_case()]},
            benchmark_sha,
            run_authorized=True,
            calibration_decision={"decision": CALIBRATION_READY},
        )
    )

    assert provider_calls == 1  # Fake production adapter only; no model provider was called.
    assert result["run_id"] == POST_IMPROVEMENT_RUN_ID
    assert paths.raw_results.is_file()
    assert not (tmp_path / "formal_run_baseline").exists()
    assert not (tmp_path / "raw_results.json").exists()


def test_same_benchmark_with_old_system_identity_cannot_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _patch_formal_run_paths(monkeypatch, tmp_path)
    paths.directory.mkdir(parents=True)
    benchmark_sha = canonical_json_file_sha256(BENCHMARK_PATH)
    atomic_write_json(paths.raw_results, {
        "run_id": POST_IMPROVEMENT_RUN_ID,
        "benchmark_sha256": benchmark_sha,
        "system_under_test_sha": "old-system",
        "active_kb_build_id": EXPECTED_KB_BUILD_ID,
        "records": [],
    })

    with pytest.raises(EvaluationBlocked, match="FORMAL_RUN_IDENTITY_MISMATCH"):
        asyncio.run(
            run_formal_cases(
                {"cases": []},
                benchmark_sha,
                run_authorized=True,
                calibration_decision={"decision": CALIBRATION_READY},
            )
        )


def test_matching_post_improvement_checkpoint_can_resume_without_rerunning_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _patch_formal_run_paths(monkeypatch, tmp_path)
    paths.directory.mkdir(parents=True)
    benchmark_sha = canonical_json_file_sha256(BENCHMARK_PATH)
    case = _synthetic_formal_case()
    payload = {
        "run_id": POST_IMPROVEMENT_RUN_ID,
        "benchmark_sha256": benchmark_sha,
        "system_under_test_sha": SYSTEM_UNDER_TEST_SHA,
        "active_kb_build_id": EXPECTED_KB_BUILD_ID,
        "records": [{
            "case_id": case["case_id"],
            "case_family": case["family"],
            "pipeline_fingerprint": EXPECTED_PIPELINE_FINGERPRINT,
            "infrastructure_error": None,
        }],
    }
    atomic_write_json(paths.raw_results, payload)

    async def unexpected_agent(**_kwargs):
        pytest.fail("a completed matching case must not be rerun")

    _install_fake_agent(monkeypatch, unexpected_agent)
    resumed = asyncio.run(
        run_formal_cases(
            {"cases": [case]},
            benchmark_sha,
            run_authorized=True,
            calibration_decision={"decision": CALIBRATION_READY},
        )
    )

    assert resumed == payload


def test_pipeline_fingerprint_mismatch_blocks_completion_and_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_sha = canonical_json_file_sha256(BENCHMARK_PATH)
    records = [
        {
            "case_id": f"CASE-{index:03d}",
            "pipeline_fingerprint": (
                "mixed-fingerprint" if index == 100 else EXPECTED_PIPELINE_FINGERPRINT
            ),
            "infrastructure_error": None,
        }
        for index in range(1, 101)
    ]
    raw = {
        "run_id": POST_IMPROVEMENT_RUN_ID,
        "benchmark_sha256": benchmark_sha,
        "system_under_test_sha": SYSTEM_UNDER_TEST_SHA,
        "active_kb_build_id": EXPECTED_KB_BUILD_ID,
        "records": records,
    }
    provider_calls = 0

    def adapter(_prompts: list[str]) -> list[str]:
        nonlocal provider_calls
        provider_calls += 1
        return []

    with pytest.raises(EvaluationBlocked, match="PIPELINE_FINGERPRINT_MISMATCH"):
        require_complete_formal_run(raw, benchmark_sha)
    with pytest.raises(EvaluationBlocked, match="PIPELINE_FINGERPRINT_MISMATCH"):
        score_ragchecker({"cases": []}, raw, adapter)
    assert provider_calls == 0


def test_formal_runtime_helper_requires_manual_and_calibration_gates() -> None:
    with pytest.raises(EvaluationBlocked, match="RUN_AUTHORIZATION_REQUIRED"):
        asyncio.run(
            run_formal_cases(
                {"cases": []},
                "hash",
                run_authorized=False,
                calibration_decision=None,
            )
        )

    with pytest.raises(EvaluationBlocked, match="BLOCKED_BY_EVALUATOR_CALIBRATION"):
        asyncio.run(
            run_formal_cases(
                {"cases": []},
                "hash",
                run_authorized=True,
                calibration_decision={"blocked": True},
            )
        )

    with pytest.raises(EvaluationBlocked, match="CALIBRATION_REVIEW_REQUIRED"):
        asyncio.run(
            run_formal_cases(
                {"cases": []},
                "hash",
                run_authorized=True,
                calibration_decision={"decision": CALIBRATION_REVIEW_REQUIRED},
            )
        )
