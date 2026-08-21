"""Executable support for the researcher-run formal evaluation notebook.

Nothing in this module runs at import time. The notebook must explicitly pass
the researcher gate, run evaluator calibration, and then call the live Agent.
Production modules are imported lazily so package validation remains offline.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from nltk.stem import LancasterStemmer


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT / "evaluation"
RESULTS_DIR = EVALUATION_DIR / "results"
BENCHMARK_PATH = EVALUATION_DIR / "benchmark_100.json"
MANIFEST_PATH = EVALUATION_DIR / "benchmark_manifest.json"
CALIBRATION_PATH = EVALUATION_DIR / "evaluator_calibration.json"
CALIBRATION_RESULTS_PATH = RESULTS_DIR / "evaluator_calibration_results.json"
RAW_RESULTS_PATH = RESULTS_DIR / "raw_results.json"
CASE_METRICS_PATH = RESULTS_DIR / "case_metrics.csv"
METRICS_SUMMARY_PATH = RESULTS_DIR / "metrics_summary.csv"
RAGCHECKER_CHECKPOINT_PATH = RESULTS_DIR / "ragchecker_checkpoint.json"

EXPECTED_BASE_SHA = "6a1809c4ddedbccab986ec76eb730321686ff3ff"
EXPECTED_KB_BUILD_ID = "94d613bc9b33628de3ef"
EVALUATOR_MODEL = "gpt-5.4-mini-2026-03-17"
RAGCHECKER_VERSION = "0.1.9"
EXTRACTION_REASONING_EFFORT = "medium"
CHECKING_REASONING_EFFORT = "low"
CALIBRATION_READY = "CALIBRATION_READY_FOR_FORMAL_RUN"
CALIBRATION_REVIEW_REQUIRED = "CALIBRATION_REVIEW_REQUIRED"
CALIBRATION_BLOCKED = "BLOCKED_BY_EVALUATOR_CALIBRATION"

_ENGLISH_STOPWORDS = frozenset({"a", "an", "and", "are", "be", "is", "of", "the", "to"})
_STEMMER = LancasterStemmer()
_RAGCHECKER_EXTRACTION_PROMPT_PREFIXES = (
    "Given a question and a response to the question, please extract a KG",
    "Given a question and a candidate answer to the question, please extract a KG",
    "Given an input text, please extract a KG",
    "You are an AI assistant, you can help to extract claims",
)
_RAGCHECKER_CHECKING_PROMPT_PREFIXES = (
    "I have a list of claims that made by a language model",
    "I have a claim that made by a language model",
)


class EvaluationBlocked(RuntimeError):
    """Raised when a methodology gate forbids the next evaluation stage."""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_pretty_json(temporary, value)
    temporary.replace(path)


def canonical_json_bytes(value: Any) -> bytes:
    """Return format-independent JSON identity bytes used by this benchmark.

    The final newline is retained from the approved benchmark identity contract,
    so presentation-only formatting cannot change an existing dataset SHA.
    """

    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (serialized + "\n").encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_json_file_sha256(path: Path) -> str:
    return canonical_json_sha256(load_json(path))


def write_pretty_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def load_evaluation_artifacts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return load_json(BENCHMARK_PATH), load_json(MANIFEST_PATH), load_json(CALIBRATION_PATH)


def validate_baseline(manifest: dict[str, Any]) -> dict[str, Any]:
    """Prove that production behavior/data still correspond to the locked SHA."""

    head = _git("rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_BASE_SHA, head],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    production_diff = _git(
        "diff",
        "--name-only",
        EXPECTED_BASE_SHA,
        "--",
        "src",
        "scripts",
        "sample_data",
        "data/sources",
        "data/taxonomy",
    ).splitlines()
    knowledge_manifest = load_json(ROOT / "data" / "knowledge_build_manifest.json")
    checks = {
        "evaluation_base_sha": manifest.get("evaluation_base_sha") == EXPECTED_BASE_SHA,
        "base_is_ancestor": ancestor,
        "production_diff_from_base": production_diff,
        "active_kb_build_id": knowledge_manifest.get("build_id") == EXPECTED_KB_BUILD_ID,
        "phase1_frozen": knowledge_manifest.get("phase1_frozen") is True,
        "manifest_status": knowledge_manifest.get("status") == "activated",
        "benchmark_manifest_kb": manifest.get("active_kb_build_id") == EXPECTED_KB_BUILD_ID,
    }
    if not all(value is True or value == [] for value in checks.values()):
        raise EvaluationBlocked(f"Baseline không khớp: {checks}")
    return {"current_head": head, **checks}


def validate_benchmark(
    benchmark: dict[str, Any], manifest: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, Any]:
    cases = benchmark.get("cases") or []
    ids = [case.get("case_id") for case in cases]
    queries = [str(case.get("query") or "").strip().casefold() for case in cases]
    family_counts = Counter(case.get("family") for case in cases)
    category_counts = Counter(case.get("category") for case in cases)
    answerable = [case for case in cases if str(case.get("family", "")).startswith("answerable")]
    gaps = [case for case in cases if case.get("family") == "evidence_gap"]
    errors: list[str] = []

    if len(cases) != 100:
        errors.append(f"expected 100 cases, got {len(cases)}")
    if len(ids) != len(set(ids)):
        errors.append("duplicate case_id")
    if len(queries) != len(set(queries)):
        errors.append("duplicate query")
    if len(answerable) != 70 or len(gaps) != 30:
        errors.append(f"unexpected family counts: {dict(family_counts)}")
    if manifest.get("benchmark_sha256") != canonical_json_file_sha256(BENCHMARK_PATH):
        errors.append("benchmark SHA-256 mismatch")
    if manifest.get("category_counts") != dict(sorted(category_counts.items())):
        errors.append("category counts do not match manifest")
    anti = benchmark.get("anti_contamination") or {}
    if any(
        anti.get(key) != 0
        for key in (
            "exact_matches",
            "near_matches_at_or_above_0_86",
            "manual_pattern_matches_at_or_above_0_86",
            "internal_query_near_duplicates_at_or_above_0_80",
        )
    ):
        errors.append(f"anti-contamination or internal deduplication failed: {anti}")

    for case in answerable:
        if not case.get("gold_claims") or not case.get("gold_answer") or not case.get("provenance"):
            errors.append(f"{case.get('case_id')}: incomplete answerable gold data")
        allowed_chunks = {item.get("chunk_id") for item in case.get("provenance", [])}
        if case.get("gold_answer") != " ".join(claim["text"] for claim in case.get("gold_claims", [])):
            errors.append(f"{case.get('case_id')}: gold answer is not composed from case gold claims")
        for claim in case.get("gold_claims", []):
            if not set(claim.get("source_chunk_ids", [])).issubset(allowed_chunks):
                errors.append(f"{case.get('case_id')}: claim source not in provenance")
            snippets = claim.get("evidence_snippets") or []
            if not snippets:
                errors.append(f"{case.get('case_id')}: claim has no evidence snippet")
            if any(item.get("chunk_id") not in claim.get("source_chunk_ids", []) or not item.get("text") for item in snippets):
                errors.append(f"{case.get('case_id')}: invalid claim evidence snippet")
            if claim.get("annotation_status") != "source_annotated_pending_researcher_review":
                errors.append(f"{case.get('case_id')}: invalid gold annotation status")
    for case in gaps:
        expected = case.get("expected") or {}
        audit = case.get("absence_verification") or {}
        if expected != {"action": "abstain", "reason": "evidence_gap"}:
            errors.append(f"{case.get('case_id')}: invalid evidence-gap expectation")
        if audit.get("candidate_search_completed") is not True:
            errors.append(f"{case.get('case_id')}: candidate search incomplete")
        if audit.get("absence_review_status") != "pending_researcher_review":
            errors.append(f"{case.get('case_id')}: invalid absence review status")
        if not audit.get("unsupported_factual_requirement"):
            errors.append(f"{case.get('case_id')}: missing unsupported requirement")
        if audit.get("provider_calls") != 0 or audit.get("datastore_writes") != 0:
            errors.append(f"{case.get('case_id')}: absence audit was not read-only")
        dense = audit.get("related_topic_dense_probe") or {}
        sparse = audit.get("bm25_candidate_search") or {}
        if len(dense.get("candidate_chunk_ids", [])) != 20 or not dense.get("review_candidates"):
            errors.append(f"{case.get('case_id')}: related-topic dense probe incomplete")
        if len(sparse.get("candidate_chunk_ids", [])) != 20 or not sparse.get("review_candidates"):
            errors.append(f"{case.get('case_id')}: BM25 candidate search incomplete")

    calibration_counts = calibration.get("counts") or {}
    if calibration_counts != {
        "total": 20,
        "claim_extraction": 8,
        "claim_checking": 12,
        "supported": 6,
        "not_supported": 6,
        "vi_to_vi": 6,
        "en_to_vi": 6,
    }:
        errors.append(f"unexpected calibration counts: {calibration_counts}")
    for item in calibration.get("claim_extraction", []):
        references = item.get("reference_claims") or []
        if not references or any(
            not reference.get("required_concept_groups") or not reference.get("text")
            for reference in references
        ):
            errors.append(f"{item.get('item_id')}: invalid atomic reference structure")
    for item in calibration.get("claim_checking", []):
        try:
            _checker_claim_triplets(item)
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise EvaluationBlocked("Benchmark validation failed:\n- " + "\n- ".join(errors))
    return {
        "total": len(cases),
        "answerable": len(answerable),
        "evidence_gap": len(gaps),
        "family_counts": dict(sorted(family_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "source_coverage": benchmark.get("coverage", {}).get("source_coverage", []),
        "benchmark_sha256": manifest["benchmark_sha256"],
        "duplicate_case_ids": 0,
        "duplicate_queries": 0,
        "provenance_reference_integrity": "passed",
        "gold_semantic_source_review": "pending_researcher_review",
        "evidence_gap_candidate_searches": f"{len(gaps)}/{len(gaps)} completed",
        "evidence_gap_absence_reviews": f"{len(gaps)}/{len(gaps)} pending researcher review",
        "evidence_gap_review_rows": [
            {
                "case_id": case["case_id"],
                "query": case["query"],
                "unsupported_requirement": case["absence_verification"]["unsupported_factual_requirement"],
                "top_candidate_snippets": [
                    item["snippet"]
                    for item in case["absence_verification"]["bm25_candidate_search"]["review_candidates"][:2]
                ],
                "absence_review_status": case["absence_verification"]["absence_review_status"],
            }
            for case in gaps
        ],
        "calibration_structure": "8 extraction + 12 checking",
        "researcher_review_status": manifest.get("researcher_review_status"),
    }


def require_researcher_review(approved: bool) -> None:
    if not approved:
        raise EvaluationBlocked(
            "RESEARCHER_REVIEW_REQUIRED: Hãy đối chiếu benchmark và calibration, sau đó "
            "đổi RESEARCHER_REVIEW_APPROVED = True để kiểm tra mô hình chấm điểm và chạy đánh giá."
        )


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.casefold())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _ragchecker_prompt_role(prompt: str) -> str:
    candidate = prompt.lstrip()
    if any(candidate.startswith(prefix) for prefix in _RAGCHECKER_EXTRACTION_PROMPT_PREFIXES):
        if "this is an EXTRACTION task" not in candidate:
            raise EvaluationBlocked("RAGCHECKER_EXTRACTION_PROMPT_CONTRACT_MISMATCH")
        return "extraction"
    if any(candidate.startswith(prefix) for prefix in _RAGCHECKER_CHECKING_PROMPT_PREFIXES):
        if "checking whether" not in candidate or "DO NOT use your own knowledge" not in candidate:
            raise EvaluationBlocked("RAGCHECKER_CHECKING_PROMPT_CONTRACT_MISMATCH")
        return "checking"
    raise EvaluationBlocked("RAGCHECKER_PROMPT_ROLE_UNKNOWN")


def _ragchecker_batch_role(prompts: list[str]) -> str:
    if not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise EvaluationBlocked("RAGCHECKER_PROMPT_BATCH_INVALID")
    roles = {_ragchecker_prompt_role(prompt) for prompt in prompts}
    if len(roles) != 1:
        raise EvaluationBlocked("RAGCHECKER_PROMPT_BATCH_MIXED_ROLES")
    return roles.pop()


def _evaluator_request_configuration() -> dict[str, Any]:
    return {
        "extraction": {"model": EVALUATOR_MODEL, "reasoning_effort": EXTRACTION_REASONING_EFFORT},
        "checking": {"model": EVALUATOR_MODEL, "reasoning_effort": CHECKING_REASONING_EFFORT},
    }


def build_openai_batch_adapter(model: str = EVALUATOR_MODEL) -> Callable[[list[str]], list[str]]:
    """Return a fail-closed RAGChecker callback with stage-specific reasoning effort."""

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EvaluationBlocked("OPENAI_API_KEY chưa được cấu hình tại notebook runtime.")
    from openai import OpenAI  # Imported only after the manual gate.

    client = OpenAI(api_key=api_key)

    def invoke(prompts: list[str]) -> list[str]:
        role = _ragchecker_batch_role(prompts)
        reasoning_effort = (
            EXTRACTION_REASONING_EFFORT if role == "extraction" else CHECKING_REASONING_EFFORT
        )
        outputs: list[str] = []
        for prompt in prompts:
            response = client.responses.create(
                model=model,
                input=prompt,
                reasoning={"effort": reasoning_effort},
                store=False,
            )
            outputs.append(response.output_text)
        return outputs

    return invoke


def _make_ragchecker(adapter: Callable[[list[str]], list[str]]):
    from ragchecker import RAGChecker

    return RAGChecker(
        extractor_name=EVALUATOR_MODEL,
        checker_name=EVALUATOR_MODEL,
        batch_size_extractor=8,
        batch_size_checker=8,
        joint_check=True,
        joint_check_num=5,
        custom_llm_api_func=adapter,
    )


def _concept_group_present(text: str, alternatives: list[str]) -> bool:
    normalized = _normalize(text)
    if any(_normalize(alternative) in normalized for alternative in alternatives):
        return True

    text_tokens = [
        _STEMMER.stem(token)
        for token in normalized.split()
        if token not in _ENGLISH_STOPWORDS
    ]
    for alternative in alternatives:
        alternative_tokens = [
            _STEMMER.stem(token)
            for token in _normalize(alternative).split()
            if token not in _ENGLISH_STOPWORDS
        ]
        if not alternative_tokens:
            continue
        width = len(alternative_tokens)
        if any(
            text_tokens[index : index + width] == alternative_tokens
            for index in range(len(text_tokens) - width + 1)
        ):
            return True
    return False


def _extraction_assessment(item: dict[str, Any], claims: list[str]) -> tuple[str, list[str]]:
    """Apply deterministic reference-aware checks and expose uncertain extras for review."""

    clean_claims = [str(claim).strip() for claim in claims if str(claim).strip()]
    reasons: list[str] = []
    if not clean_claims or len(clean_claims) != len(claims) or len(clean_claims) > 8:
        return "rejected", ["invalid_or_empty_claim_structure"]

    references = item["reference_claims"]
    mapped_claims: set[int] = set()
    complete_references_by_claim = [set() for _ in clean_claims]
    for reference in references:
        required = reference["required_concept_groups"]
        qualifiers = reference.get("qualifier_concept_groups") or []
        reference_id = reference["reference_id"]
        anchor_claims = {
            index
            for index, claim in enumerate(clean_claims)
            if _concept_group_present(claim, required[0])
        }
        required_matches = [
            {
                index
                for index in anchor_claims
                if _concept_group_present(clean_claims[index], group)
            }
            for group in required
        ]
        if not anchor_claims or any(not matches for matches in required_matches):
            reasons.append(f"missing_reference:{reference_id}")
            continue

        qualifier_matches = [
            {
                index
                for index in anchor_claims
                if _concept_group_present(clean_claims[index], group)
            }
            for group in qualifiers
        ]
        if any(not matches for matches in qualifier_matches):
            reasons.append(f"missing_qualifier:{reference_id}")
            continue

        contributing_claims = set().union(*required_matches[1:], *qualifier_matches)
        if not contributing_claims:
            contributing_claims = set(anchor_claims)
        mapped_claims.update(contributing_claims)
        for index in anchor_claims:
            if all(index in matches for matches in required_matches + qualifier_matches):
                complete_references_by_claim[index].add(reference_id)

    joined = " ".join(clean_claims)
    for marker in item["acceptance"].get("forbidden_inventions", []):
        if _concept_group_present(joined, [marker]):
            reasons.append(f"forbidden_invention:{marker}")
    if reasons:
        clear_failures = [
            reason
            for reason in reasons
            if reason.startswith(("missing_qualifier:", "forbidden_invention:"))
        ]
        if clear_failures:
            return "rejected", reasons
        return "review_required", reasons

    if any(len(reference_ids) > 1 for reference_ids in complete_references_by_claim):
        return "rejected", ["atomicity:independent_references_merged"]
    unassigned = sorted(set(range(len(clean_claims))) - mapped_claims)
    if unassigned:
        return "review_required", [f"unmapped_claims:{','.join(map(str, unassigned))}"]
    return "accepted", []


def _extraction_acceptable(item: dict[str, Any], claims: list[str]) -> tuple[bool, list[str]]:
    status, reasons = _extraction_assessment(item, claims)
    return status == "accepted", reasons


def _checker_label(raw: Any) -> str:
    values = raw if isinstance(raw, list) else [raw]
    labels: list[str] = []
    for value in values:
        normalized = re.sub(r"[^a-z_ ]+", "", str(value).strip().casefold()).replace("_", " ")
        normalized = " ".join(normalized.split())
        if normalized in {"entailment", "supported"}:
            labels.append("SUPPORTED")
        elif normalized in {"contradiction", "neutral", "not supported"}:
            labels.append("NOT_SUPPORTED")
        else:
            labels.append("UNPARSEABLE")
    if not labels or "UNPARSEABLE" in labels:
        return "UNPARSEABLE"
    return "SUPPORTED" if all(label == "SUPPORTED" for label in labels) else "NOT_SUPPORTED"


def _checker_claim_triplets(item: dict[str, Any]) -> list[list[str]]:
    triplets = item.get("claim_triplets")
    valid = (
        isinstance(triplets, list)
        and bool(triplets)
        and all(
            isinstance(triplet, list)
            and len(triplet) == 3
            and all(isinstance(part, str) and part.strip() for part in triplet)
            for triplet in triplets
        )
    )
    if not valid:
        raise ValueError(f"Calibration checker item requires atomic claim_triplets: {item.get('item_id')}")
    return [[part.strip() for part in triplet] for triplet in triplets]


def run_calibration_once(calibration: dict[str, Any], adapter: Callable[[list[str]], list[str]]) -> dict[str, Any]:
    from ragchecker import RAGResult, RAGResults

    evaluator = _make_ragchecker(adapter)
    extraction_rows = calibration["claim_extraction"]
    extraction_results = [
        RAGResult(
            query_id=item["item_id"],
            query="Extract atomic factual claims while preserving qualifiers.",
            gt_answer=item["input_text"],
            response="",
            retrieved_context=[],
        )
        for item in extraction_rows
    ]
    evaluator.extract_claims(extraction_results, extract_type="gt_answer")
    extraction_output = []
    for item, result in zip(extraction_rows, extraction_results, strict=True):
        claims = [str(claim) for claim in (result.gt_answer_claims or [])]
        status, reasons = _extraction_assessment(item, claims)
        extraction_output.append({
            "item_id": item["item_id"],
            "status": status,
            "acceptable": status == "accepted",
            "review_required": status == "review_required",
            "reasons": reasons,
            "claims": claims,
        })

    checking_rows = calibration["claim_checking"]
    checking_results = RAGResults(results=[
        RAGResult(
            query_id=item["item_id"],
            query="Does the evidence support the claim?",
            gt_answer=item["evidence"],
            response=item["claim"],
            retrieved_context=[],
            response_claims=_checker_claim_triplets(item),
        )
        for item in checking_rows
    ])
    evaluator.check_claims(checking_results, check_type="answer2response")
    checking_output = []
    for item, result in zip(checking_rows, checking_results.results, strict=True):
        actual = _checker_label(result.answer2response)
        checking_output.append({
            "item_id": item["item_id"],
            "expected": item["expected_label"],
            "actual": actual,
            "agreement": actual == item["expected_label"],
            "critical_dimensions": item["critical_dimensions"],
            "raw": result.answer2response,
        })
    return {"extraction": extraction_output, "checking": checking_output}


def evaluate_calibration_runs(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    extraction_pairs = list(zip(first["extraction"], second["extraction"], strict=True))
    checking_pairs = list(zip(first["checking"], second["checking"], strict=True))
    extraction_acceptable = sum(a["acceptable"] and b["acceptable"] for a, b in extraction_pairs)
    checker_agreement = sum(a["agreement"] and b["agreement"] for a, b in checking_pairs)
    repeat_extraction = sum(a["status"] == b["status"] for a, b in extraction_pairs)
    repeat_checking = sum(a["actual"] == b["actual"] for a, b in checking_pairs)

    critical_failures: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    for left, right in extraction_pairs:
        repeated_reasons = sorted(set(left["reasons"]) & set(right["reasons"]))
        critical = [
            reason
            for reason in repeated_reasons
            if reason.startswith((
                "missing_qualifier:",
                "forbidden_invention:",
                "atomicity:",
            ))
        ]
        if left["status"] == right["status"] == "rejected" and critical:
            critical_failures.append({"item_id": left["item_id"], "type": "extraction", "reasons": critical})
        elif left["status"] != "accepted" or right["status"] != "accepted" or left["status"] != right["status"]:
            disagreements.append({"item_id": left["item_id"], "type": "extraction", "first": left, "second": right})
    for left, right in checking_pairs:
        confirmed_wrong = (
            left["actual"] == right["actual"]
            and left["actual"] not in {left["expected"], "UNPARSEABLE"}
        )
        if confirmed_wrong:
            critical_failures.append({"item_id": left["item_id"], "type": "checking", "first": left["actual"], "second": right["actual"], "dimensions": left["critical_dimensions"]})
        elif not left["agreement"] or not right["agreement"] or left["actual"] != right["actual"]:
            disagreements.append({"item_id": left["item_id"], "type": "checking", "first": left, "second": right})

    if critical_failures:
        decision = CALIBRATION_BLOCKED
    elif disagreements:
        decision = CALIBRATION_REVIEW_REQUIRED
    else:
        decision = CALIBRATION_READY
    return {
        "claim_extraction_acceptable": extraction_acceptable,
        "claim_extraction_total": 8,
        "claim_checking_agreement": checker_agreement,
        "claim_checking_total": 12,
        "repeat_consistency": repeat_extraction + repeat_checking,
        "repeat_total": 20,
        "critical_semantic_failures": critical_failures,
        "disagreements": disagreements,
        "blocked": decision == CALIBRATION_BLOCKED,
        "requires_researcher_review": decision == CALIBRATION_REVIEW_REQUIRED,
        "formal_run_allowed": decision == CALIBRATION_READY,
        "decision": decision,
    }


def save_calibration_results(
    calibration: dict[str, Any],
    first: dict[str, Any],
    second: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Persist the two-run evaluator audit even when calibration blocks."""

    payload = {
        "evaluator_model": EVALUATOR_MODEL,
        "evaluator_request_configuration": _evaluator_request_configuration(),
        "run_timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "calibration_dataset_sha256": canonical_json_file_sha256(CALIBRATION_PATH),
        "calibration_schema_version": calibration.get("schema_version"),
        "run_1": first,
        "run_2": second,
        "claim_extraction_acceptable": decision["claim_extraction_acceptable"],
        "claim_checking_agreement": decision["claim_checking_agreement"],
        "repeat_consistency": decision["repeat_consistency"],
        "disagreements": decision["disagreements"],
        "critical_failures": decision["critical_semantic_failures"],
        "final_calibration_decision": decision["decision"],
    }
    atomic_write_json(CALIBRATION_RESULTS_PATH, payload)
    return payload


def _installed_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _formal_run_metadata(*, allow_model_fallback: bool, researcher_review_approved: bool) -> dict[str, Any]:
    return {
        "benchmark_sha256": canonical_json_file_sha256(BENCHMARK_PATH),
        "evaluation_base_sha": EXPECTED_BASE_SHA,
        "current_git_head": _git("rev-parse", "HEAD"),
        "active_kb_build_id": EXPECTED_KB_BUILD_ID,
        "run_timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "python_version": platform.python_version(),
        "ragchecker_version": _installed_version("ragchecker"),
        "openai_package_version": _installed_version("openai"),
        "spacy_version": _installed_version("spacy"),
        "spacy_model_identifier": "en_core_web_sm",
        "evaluator_model": EVALUATOR_MODEL,
        "evaluator_request_configuration": _evaluator_request_configuration(),
        "observed_production_requested_model_configuration": {
            "provider": os.getenv("LLM_PROVIDER", "gemini"),
            "google_model": os.getenv("GOOGLE_MODEL", "gemini-3.5-flash-lite"),
        },
        "observed_fallback_model_configuration": {
            "google_fallback_models": os.getenv("GOOGLE_FALLBACK_MODELS", "gemini-3.1-flash-lite"),
            "provider_fallback_enabled": os.getenv("LLM_PROVIDER_FALLBACK_ENABLED", "false"),
            "ollama_model": os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        },
        "allow_model_fallback": allow_model_fallback,
        "researcher_review_approved": researcher_review_approved,
    }


def _decision_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _extract_action_reason(result: dict[str, Any]) -> tuple[str | None, str | None]:
    decision = result.get("agent_decision")
    action = _decision_field(decision, "action")
    reason = _decision_field(decision, "reason_code")
    if not action:
        history = result.get("agent_decision_history") or []
        if history:
            action = _decision_field(history[-1], "action")
            reason = reason or _decision_field(history[-1], "reason_code")
    if result.get("fallback_type") == "evidence_gap":
        action = action or "abstain"
        reason = reason or "evidence_gap"
    return action, reason


def _packed_contexts(result: dict[str, Any]) -> list[dict[str, Any]]:
    packed = result.get("packed_context") or {}
    output = []
    for index, item in enumerate(packed.get("items") or []):
        if not isinstance(item, dict):
            continue
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        output.append({
            "position": index + 1,
            "chunk_id": payload.get("chunk_id") or item.get("item_id"),
            "source_id": payload.get("source_id"),
            "source_title": payload.get("source_title"),
            "source_url": payload.get("source_url"),
            "section_path": payload.get("section_path"),
            "text": item.get("text") or item.get("content") or "",
        })
    return output


def _safe_runtime_record(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    action, reason = _extract_action_reason(result)
    return {
        "case_id": case["case_id"],
        "case_family": case["family"],
        "category": case["category"],
        "query": case["query"],
        "history": case.get("history", []),
        "expected_action": case["expected"]["action"],
        "expected_reason": case["expected"].get("reason"),
        "answer": result.get("answer", ""),
        "actual_action": action,
        "actual_reason": reason,
        "packed_contexts": _packed_contexts(result),
        "sources": result.get("sources", []),
        "requested_provider": result.get("requested_provider"),
        "requested_model": result.get("requested_model"),
        "actual_provider": result.get("actual_provider"),
        "actual_model": result.get("actual_model"),
        "llm_fallback_used": result.get("llm_fallback_used", False),
        "fallback_provider": result.get("fallback_provider"),
        "fallback_model": result.get("fallback_model"),
        "fallback_chain": result.get("fallback_chain"),
        "retrieval_status": result.get("retrieval_status"),
        "retrieval_attempt": result.get("retrieval_attempt"),
        "retrieval_trace": result.get("retrieval_trace"),
        "evidence_assessment": result.get("evidence_assessment"),
        "agent_decision_history": result.get("agent_decision_history", []),
        "pipeline_fingerprint": result.get("pipeline_fingerprint"),
        "performance_timings": result.get("performance_timings"),
        "infrastructure_error": None,
    }


async def run_formal_cases(
    benchmark: dict[str, Any],
    benchmark_sha256: str,
    *,
    researcher_review_approved: bool,
    calibration_decision: dict[str, Any] | None,
    allow_model_fallback: bool = True,
) -> dict[str, Any]:
    """Run each fixed case once, checkpointing without selecting better outputs."""

    require_researcher_review(researcher_review_approved)
    decision = (calibration_decision or {}).get("decision")
    if decision != CALIBRATION_READY:
        raise EvaluationBlocked(decision or CALIBRATION_BLOCKED)

    from src.agent.graph import run_clinical_agent

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_RESULTS_PATH.exists():
        payload = load_json(RAW_RESULTS_PATH)
        if payload.get("benchmark_sha256") != benchmark_sha256:
            raise EvaluationBlocked("Checkpoint benchmark SHA không khớp; không được resume.")
    else:
        run_metadata = _formal_run_metadata(
            allow_model_fallback=allow_model_fallback,
            researcher_review_approved=researcher_review_approved,
        )
        if run_metadata["benchmark_sha256"] != benchmark_sha256:
            raise EvaluationBlocked("Benchmark SHA changed before formal execution.")
        payload = {
            "benchmark_sha256": benchmark_sha256,
            "evaluation_base_sha": EXPECTED_BASE_SHA,
            "active_kb_build_id": EXPECTED_KB_BUILD_ID,
            "run_metadata": run_metadata,
            "records": [],
        }
    completed = {record["case_id"] for record in payload["records"]}
    cases = benchmark["cases"]
    for index, case in enumerate(cases, start=1):
        if case["case_id"] in completed:
            print(f"[{index:03d}/100] {case['case_id']} RESUMED")
            continue
        try:
            result = await run_clinical_agent(
                message=case["query"],
                conversation_history=case.get("history") or [],
                allow_model_fallback=allow_model_fallback,
                bypass_cache=True,
                include_generation_diagnostics=False,
            )
            record = _safe_runtime_record(case, result)
            status = "OK"
        except Exception as exc:  # Preserve the failure; never rerun for a prettier output.
            record = {
                "case_id": case["case_id"],
                "case_family": case["family"],
                "category": case["category"],
                "query": case["query"],
                "history": case.get("history", []),
                "expected_action": case["expected"]["action"],
                "expected_reason": case["expected"].get("reason"),
                "answer": None,
                "actual_action": None,
                "actual_reason": None,
                "packed_contexts": [],
                "sources": [],
                "infrastructure_error": {"type": type(exc).__name__, "message": str(exc)[:500]},
            }
            status = "INFRASTRUCTURE_ERROR"
        payload["records"].append(record)
        atomic_write_json(RAW_RESULTS_PATH, payload)
        print(f"[{index:03d}/100] {case['case_id']} {status}")
    failures = sum(record.get("infrastructure_error") is not None for record in payload["records"])
    print(f"Raw results: {RAW_RESULTS_PATH}")
    print(f"Successful cases: {len(payload['records']) - failures}")
    print(f"Infrastructure failures: {failures}")
    print(f"Answerable: {sum(str(r['case_family']).startswith('answerable') for r in payload['records'])}; evidence-gap: {sum(r['case_family'] == 'evidence_gap' for r in payload['records'])}")
    return payload


def require_complete_formal_run(raw: dict[str, Any], benchmark_sha256: str) -> None:
    records = raw.get("records") or []
    failures = [record["case_id"] for record in records if record.get("infrastructure_error")]
    if raw.get("benchmark_sha256") != benchmark_sha256 or len(records) != 100 or failures:
        raise EvaluationBlocked(
            f"FORMAL_RUN_BLOCKED_BY_INFRASTRUCTURE: records={len(records)}, failures={failures}"
        )


def score_ragchecker(
    benchmark: dict[str, Any],
    raw: dict[str, Any],
    adapter: Callable[[list[str]], list[str]],
):
    """Use official RAGChecker metrics without changing their semantics."""

    from ragchecker import RAGChecker, RAGResult, RAGResults
    from ragchecker.container import RetrievedDoc
    from ragchecker.metrics import claim_recall, context_precision, f1, faithfulness

    cases = {case["case_id"]: case for case in benchmark["cases"]}
    results = []
    for record in raw["records"]:
        case = cases[record["case_id"]]
        if not str(case["family"]).startswith("answerable"):
            continue
        contexts = [
            RetrievedDoc(doc_id=item.get("chunk_id"), text=item.get("text", ""))
            for item in record.get("packed_contexts", [])
            if item.get("text")
        ]
        results.append(RAGResult(
            query_id=case["case_id"],
            query=case["query"],
            gt_answer=case["gold_answer"],
            response=record["answer"],
            retrieved_context=contexts,
        ))
    rag_results = RAGResults(results=results)
    evaluator = RAGChecker(
        extractor_name=EVALUATOR_MODEL,
        checker_name=EVALUATOR_MODEL,
        batch_size_extractor=8,
        batch_size_checker=8,
        joint_check=True,
        joint_check_num=5,
        custom_llm_api_func=adapter,
    )
    evaluator.evaluate(
        rag_results,
        metrics=[claim_recall, context_precision, faithfulness, f1],
        save_path=str(RAGCHECKER_CHECKPOINT_PATH),
    )
    return rag_results


def negative_rejection_rate(raw: dict[str, Any]) -> tuple[float, int]:
    negatives = [record for record in raw["records"] if record["case_family"] == "evidence_gap"]
    correct = sum(record.get("actual_action") == "abstain" and record.get("actual_reason") == "evidence_gap" for record in negatives)
    if len(negatives) != 30:
        raise EvaluationBlocked(f"NRR requires 30 evidence-gap cases, got {len(negatives)}")
    return (correct / 30) * 100.0, correct


def _ratio_to_percent(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationBlocked(f"Invalid RAGChecker per-case metric: {value!r}") from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise EvaluationBlocked(f"RAGChecker per-case metric outside ratio 0-1 scale: {numeric}")
    return numeric * 100.0


def _validate_aggregate_percent(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationBlocked(f"Invalid RAGChecker aggregate metric: {value!r}") from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 100.0:
        raise EvaluationBlocked(f"RAGChecker aggregate metric outside percent 0-100 scale: {numeric}")
    return numeric


def _aggregate_metric_percent(metrics: dict[str, Any], group: str, key: str) -> float | None:
    return _validate_aggregate_percent((metrics.get(group) or {}).get(key))


def export_metrics(
    benchmark: dict[str, Any], raw: dict[str, Any], rag_results: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = {case["case_id"]: case for case in benchmark["cases"]}
    rag_by_id = {item.query_id: item for item in rag_results.results}
    rows: list[dict[str, Any]] = []
    for record in raw["records"]:
        case = cases[record["case_id"]]
        rag_item = rag_by_id.get(record["case_id"])
        metrics = rag_item.metrics if rag_item else {}
        nrr_correct = None
        if case["family"] == "evidence_gap":
            nrr_correct = record.get("actual_action") == "abstain" and record.get("actual_reason") == "evidence_gap"
        rows.append({
            "case_id": case["case_id"],
            "case_family": case["family"],
            "category": case["category"],
            "expected_action": case["expected"]["action"],
            "expected_reason": case["expected"].get("reason"),
            "actual_action": record.get("actual_action"),
            "actual_reason": record.get("actual_reason"),
            "Claim Recall (%)": _ratio_to_percent(metrics.get("claim_recall")) if rag_item else None,
            "Context Precision (%)": _ratio_to_percent(metrics.get("context_precision")) if rag_item else None,
            "Faithfulness (%)": _ratio_to_percent(metrics.get("faithfulness")) if rag_item else None,
            "Claim F1 (%)": _ratio_to_percent(metrics.get("f1")) if rag_item else None,
            "NRR correct": nrr_correct,
            "Metric unit": "percent_0_100" if rag_item else ("boolean" if nrr_correct is not None else None),
            "infrastructure_error": (record.get("infrastructure_error") or {}).get("type"),
        })
    nrr, _ = negative_rejection_rate(raw)
    aggregate = rag_results.metrics
    summary = [
        {"Metric": "Claim Recall", "N cases": 70, "Score": _aggregate_metric_percent(aggregate, "retriever_metrics", "claim_recall"), "Unit": "percent_0_100", "Vietnamese interpretation/description": "Mức độ bằng chứng truy hồi bao phủ các claim trong gold answer."},
        {"Metric": "Context Precision", "N cases": 70, "Score": _aggregate_metric_percent(aggregate, "retriever_metrics", "context_precision"), "Unit": "percent_0_100", "Vietnamese interpretation/description": "Tỷ lệ context truy hồi có liên quan đến claim gold theo RAGChecker."},
        {"Metric": "Faithfulness", "N cases": 70, "Score": _aggregate_metric_percent(aggregate, "generator_metrics", "faithfulness"), "Unit": "percent_0_100", "Vietnamese interpretation/description": "Mức độ claim trong câu trả lời được context truy hồi hỗ trợ."},
        {"Metric": "Claim F1", "N cases": 70, "Score": _aggregate_metric_percent(aggregate, "overall_metrics", "f1"), "Unit": "percent_0_100", "Vietnamese interpretation/description": "RAGChecker overall claim-level F1 giữa gold answer và system response."},
        {"Metric": "Negative Rejection Rate", "N cases": 30, "Score": nrr, "Unit": "percent_0_100", "Vietnamese interpretation/description": "Tỷ lệ evidence-gap case trả structured action=abstain và reason=evidence_gap; đây là RGB-inspired adaptation."},
    ]
    _write_csv(CASE_METRICS_PATH, rows)
    _write_csv(METRICS_SUMMARY_PATH, summary)
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def vietnamese_analysis(summary: Iterable[dict[str, Any]]) -> str:
    values = {row["Metric"]: row["Score"] for row in summary}
    rendered = ", ".join(
        f"{name}: {float(score):.2f}%" if score is not None else f"{name}: NA"
        for name, score in values.items()
    )
    return (
        "Năm chỉ số của lần chạy hiện tại là " + rendered + ". "
        "Các số liệu này mô tả hiệu năng trên bộ dữ liệu đánh giá và kho kiến thức "
        "tham chiếu của nghiên cứu; chúng không phải bằng chứng xác nhận hiệu quả lâm sàng."
    )


__all__ = [
    "BENCHMARK_PATH",
    "CALIBRATION_BLOCKED",
    "CALIBRATION_PATH",
    "CALIBRATION_READY",
    "CALIBRATION_RESULTS_PATH",
    "CALIBRATION_REVIEW_REQUIRED",
    "CASE_METRICS_PATH",
    "CHECKING_REASONING_EFFORT",
    "EVALUATOR_MODEL",
    "EXTRACTION_REASONING_EFFORT",
    "EvaluationBlocked",
    "MANIFEST_PATH",
    "METRICS_SUMMARY_PATH",
    "RAGCHECKER_VERSION",
    "RAW_RESULTS_PATH",
    "build_openai_batch_adapter",
    "canonical_json_bytes",
    "canonical_json_file_sha256",
    "canonical_json_sha256",
    "evaluate_calibration_runs",
    "export_metrics",
    "load_evaluation_artifacts",
    "negative_rejection_rate",
    "require_complete_formal_run",
    "require_researcher_review",
    "run_calibration_once",
    "run_formal_cases",
    "save_calibration_results",
    "score_ragchecker",
    "validate_baseline",
    "validate_benchmark",
    "vietnamese_analysis",
    "write_pretty_json",
]
