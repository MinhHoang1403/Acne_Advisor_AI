"""Evaluation-only support for the final thesis run.

The module locks identities, executes the production graph without changing it,
and derives metrics only from the persisted system output. It intentionally
does not run at import time and never mutates the knowledge stores.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

from evaluation.formal_evaluation_support import (
    CALIBRATION_READY,
    EVALUATOR_MODEL,
    RAGCHECKER_VERSION,
    EvaluationBlocked,
    atomic_write_json,
    build_openai_batch_adapter,
    canonical_json_file_sha256,
    canonical_json_sha256,
    validate_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT / "evaluation"
BENCHMARK_PATH = EVALUATION_DIR / "benchmark_100.json"
BENCHMARK_MANIFEST_PATH = EVALUATION_DIR / "benchmark_manifest.json"
CALIBRATION_PATH = EVALUATION_DIR / "evaluator_calibration.json"
GOLD_PATH = EVALUATION_DIR / "locked_gold" / "current_kb_adjudication.json"
GOLD_MANIFEST_PATH = EVALUATION_DIR / "locked_gold" / "current_kb_adjudication_manifest.json"
RUN_ID = "thesis_evaluation_366f3ad8"
RESULTS_DIR = EVALUATION_DIR / "results" / RUN_ID
RUN_MANIFEST_PATH = RESULTS_DIR / "run_manifest.json"
RAW_RESULTS_PATH = RESULTS_DIR / "raw_results.json"
SYSTEM_OUTPUT_MANIFEST_PATH = RESULTS_DIR / "system_output_manifest.json"
DETERMINISTIC_METRICS_PATH = RESULTS_DIR / "deterministic_metrics.json"
CASE_METRICS_PATH = RESULTS_DIR / "case_metrics.csv"
CATEGORY_METRICS_PATH = RESULTS_DIR / "category_metrics.csv"
OPERATIONAL_METRICS_PATH = RESULTS_DIR / "operational_metrics.json"
RAGCHECKER_CHECKPOINT_PATH = RESULTS_DIR / "ragchecker_checkpoint.json"
METRICS_SUMMARY_PATH = RESULTS_DIR / "metrics_summary.csv"
REPORT_PATH = RESULTS_DIR / "THESIS_EVALUATION_REPORT.md"
ARTIFACT_MANIFEST_PATH = RESULTS_DIR / "artifact_manifest.json"

SYSTEM_UNDER_TEST_SHA = "366f3ad836c36be36565caad4a94ad17a7a6278f"
EXPECTED_PIPELINE_FINGERPRINT = "e4c420923b79b37c992cf36d"
EXPECTED_CACHE_VERSION = "v10"
EXPECTED_KB_BUILD_ID = "d4a1819fe7fb77fe1f40"
EXPECTED_QDRANT_COLLECTION = f"acne_knowledge__{EXPECTED_KB_BUILD_ID}"
EXPECTED_QDRANT_POINTS = 536
EXPECTED_BENCHMARK_SHA256 = "f61d6807c0ce39f902936844d810562c486f1bcadaa57a8a2da0e460ad7e534b"
GOLD_SOURCE_COMMIT = "6da9b67634aa231a3d9cd51891af6d1685ab25cc"
GOLD_SOURCE_PATH = "research/results/benchmark_current_kb_adjudication.json"
GOLD_MANIFEST_SOURCE_PATH = "research/results/benchmark_current_kb_manifest.json"
EXPECTED_GOLD_CANONICAL_SHA256 = "e0597523954311d954266adae59f4bb86530e6bce9b5ca6b59b0148ef7d3801e"
EXPECTED_GOLD_RAW_SHA256 = "68e025affb608d220eebd5988e76d19a1baebc2f60d35a986a87b2d93db6a02e"
EXPECTED_GOLD_MANIFEST_RAW_SHA256 = (
    "27aa2985bbff654403841e745a6142d196ebc23f88560cf0a155fb807983bbdc"
)
EXPECTED_GOLD_BLOB = "e91dbe5ccfcf455c8e7f6c61436afd39b58194c9"
EXPECTED_GOLD_MANIFEST_BLOB = "08f2ac84773b097c8d516acc8ea9934ee5271436"
CALIBRATION_APPROVAL_RUN_SHA = "2d5f0124dc532a4970a4b08dcd6cf846389a03ff"
CALIBRATION_RESULTS_PATH = (
    EVALUATION_DIR / "results" / "formal_run_2d5f0124" / "evaluator_calibration_results.json"
)
CALIBRATION_ADJUDICATION_PATH = (
    EVALUATION_DIR / "results" / "formal_run_2d5f0124" / "calibration_adjudication.json"
)
CALIBRATION_RAW_RUN_PATH = EVALUATION_DIR / "results" / "formal_run_2d5f0124" / "raw_results.json"

PRODUCTION_SENSITIVE_PATHS = (
    "src",
    "scripts",
    "data/sources",
    "data/taxonomy",
    "data/knowledge_build_manifest.json",
    "data/method_sources.json",
    "requirements.txt",
    "requirements.lock.txt",
    "pyproject.toml",
    ".python-version",
    "Dockerfile",
    "docker-compose.yml",
    ".env.example",
    "src/frontend",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _git_show_bytes(revision: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def production_sensitive_diff() -> list[str]:
    output = _git(
        "diff",
        "--name-only",
        SYSTEM_UNDER_TEST_SHA,
        "HEAD",
        "--",
        *PRODUCTION_SENSITIVE_PATHS,
    )
    return output.splitlines() if output else []


def _json_request(url: str, *, body: dict[str, Any] | None = None) -> Any:
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"} if encoded is not None else {},
        method="POST" if encoded is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _qdrant_url() -> str:
    return os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")


def validate_benchmark_and_gold() -> dict[str, Any]:
    benchmark = load_json(BENCHMARK_PATH)
    benchmark_contract = validate_benchmark(
        benchmark,
        load_json(BENCHMARK_MANIFEST_PATH),
        load_json(CALIBRATION_PATH),
    )
    gold = load_json(GOLD_PATH)
    gold_manifest = load_json(GOLD_MANIFEST_PATH)
    cases = benchmark.get("cases") or []
    answerable = [case for case in cases if str(case.get("family", "")).startswith("answerable")]
    single = [case for case in answerable if case.get("family") == "answerable_single_turn"]
    multi = [case for case in answerable if case.get("family") == "answerable_multi_turn"]
    gaps = [case for case in cases if case.get("family") == "evidence_gap"]
    claims = gold.get("gold_claim_adjudications") or []
    negatives = gold.get("negative_adjudications") or []
    policy = gold.get("policy_adjudications") or []
    required_ids = sorted(
        {
            str(evidence_id)
            for claim in claims
            for evidence_id in claim.get("current_evidence_ids") or []
            if evidence_id
        }
    )

    source_gold_blob = _git("rev-parse", f"{GOLD_SOURCE_COMMIT}:{GOLD_SOURCE_PATH}")
    source_manifest_blob = _git("rev-parse", f"{GOLD_SOURCE_COMMIT}:{GOLD_MANIFEST_SOURCE_PATH}")
    source_gold_canonical = canonical_json_sha256(
        json.loads(_git_show_bytes(GOLD_SOURCE_COMMIT, GOLD_SOURCE_PATH))
    )
    source_manifest_canonical = canonical_json_sha256(
        json.loads(_git_show_bytes(GOLD_SOURCE_COMMIT, GOLD_MANIFEST_SOURCE_PATH))
    )
    current_manifest_canonical = canonical_json_file_sha256(GOLD_MANIFEST_PATH)
    checks = {
        "benchmark_canonical_sha256": canonical_json_file_sha256(BENCHMARK_PATH),
        "benchmark_total": len(cases),
        "answerable": len(answerable),
        "answerable_single_turn": len(single),
        "answerable_multi_turn": len(multi),
        "evidence_gap": len(gaps),
        "gold_source_commit": GOLD_SOURCE_COMMIT,
        "gold_canonical_sha256": canonical_json_file_sha256(GOLD_PATH),
        "gold_raw_sha256": raw_sha256(GOLD_PATH),
        "gold_manifest_raw_sha256": raw_sha256(GOLD_MANIFEST_PATH),
        "source_gold_blob": source_gold_blob,
        "source_manifest_blob": source_manifest_blob,
        "source_gold_canonical_sha256": source_gold_canonical,
        "source_manifest_canonical_sha256": source_manifest_canonical,
        "current_manifest_canonical_sha256": current_manifest_canonical,
        "reviewed_claims": len(claims),
        "technically_clear_claims": sum(
            claim.get("review_status") == "TECHNICALLY_CLEAR" for claim in claims
        ),
        "direct_support_claims": sum(
            claim.get("technical_evidence_status") == "DIRECT_SUPPORT" for claim in claims
        ),
        "confirmed_absence": sum(
            item.get("negative_status") == "CONFIRMED_ABSENCE" for item in negatives
        ),
        "policy_ambiguous": sum(
            item.get("negative_status") == "POLICY_AMBIGUOUS" for item in negatives
        ),
        "policy_rows": len(policy),
        "required_evidence_ids": len(required_ids),
        "kb_build_id": (gold.get("identity") or {}).get("current_kb_build_id"),
        "qdrant_collection": (gold.get("identity") or {}).get("qdrant_collection"),
        "qdrant_points": (gold.get("identity") or {}).get("qdrant_points"),
        "manifest_benchmark_sha256": gold_manifest.get("benchmark_source_sha256"),
        "manifest_kb_build_id": gold_manifest.get("knowledge_build_id"),
        "source_canonical_match": source_gold_canonical == canonical_json_file_sha256(GOLD_PATH),
        "source_manifest_canonical_match": source_manifest_canonical == current_manifest_canonical,
        "benchmark_schema_validation": "passed",
    }
    expected = {
        "benchmark_canonical_sha256": EXPECTED_BENCHMARK_SHA256,
        "benchmark_total": 100,
        "answerable": 70,
        "answerable_single_turn": 55,
        "answerable_multi_turn": 15,
        "evidence_gap": 30,
        "gold_canonical_sha256": EXPECTED_GOLD_CANONICAL_SHA256,
        "gold_raw_sha256": EXPECTED_GOLD_RAW_SHA256,
        "gold_manifest_raw_sha256": EXPECTED_GOLD_MANIFEST_RAW_SHA256,
        "source_gold_blob": EXPECTED_GOLD_BLOB,
        "source_manifest_blob": EXPECTED_GOLD_MANIFEST_BLOB,
        "source_gold_canonical_sha256": EXPECTED_GOLD_CANONICAL_SHA256,
        "reviewed_claims": 104,
        "technically_clear_claims": 104,
        "direct_support_claims": 104,
        "confirmed_absence": 24,
        "policy_ambiguous": 6,
        "policy_rows": 6,
        "kb_build_id": EXPECTED_KB_BUILD_ID,
        "qdrant_collection": EXPECTED_QDRANT_COLLECTION,
        "qdrant_points": EXPECTED_QDRANT_POINTS,
        "manifest_benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
        "manifest_kb_build_id": EXPECTED_KB_BUILD_ID,
        "source_canonical_match": True,
        "source_manifest_canonical_match": True,
        "benchmark_schema_validation": "passed",
    }
    mismatches = {
        key: {"expected": value, "actual": checks.get(key)}
        for key, value in expected.items()
        if checks.get(key) != value
    }
    case_ids = [case.get("case_id") for case in cases]
    claim_ids = [claim.get("claim_id") for claim in claims]
    if len(case_ids) != len(set(case_ids)):
        mismatches["case_ids"] = "duplicate"
    if len(claim_ids) != len(set(claim_ids)):
        mismatches["claim_ids"] = "duplicate"
    if any(not claim.get("current_evidence_ids") for claim in claims):
        mismatches["claim_evidence"] = "missing current_evidence_ids"
    if mismatches:
        raise EvaluationBlocked(f"FINAL_EVALUATION_GOLD_BLOCKER: {mismatches}")
    return {
        **checks,
        "benchmark_contract": benchmark_contract,
        "required_evidence_id_values": required_ids,
    }


def validate_live_qdrant(required_ids: Iterable[str]) -> dict[str, Any]:
    base = _qdrant_url()
    aliases = (_json_request(f"{base}/aliases").get("result") or {}).get("aliases") or []
    alias_map = {str(item.get("alias_name")): str(item.get("collection_name")) for item in aliases}
    collection = _json_request(f"{base}/collections/acne_knowledge").get("result") or {}
    ids = list(required_ids)
    points = (
        _json_request(
            f"{base}/collections/acne_knowledge/points",
            body={"ids": ids, "with_payload": True, "with_vector": False},
        ).get("result")
        or []
    )
    returned = {str(item.get("id")) for item in points}
    build_ids = {str((item.get("payload") or {}).get("build_id") or "") for item in points}
    report = {
        "active_target": alias_map.get("acne_knowledge"),
        "collection_status": collection.get("status"),
        "points_count": int(collection.get("points_count") or 0),
        "required_ids": len(ids),
        "returned_ids": len(returned),
        "missing_ids": sorted(set(ids) - returned),
        "payload_build_ids": sorted(build_ids),
    }
    if report != {
        "active_target": EXPECTED_QDRANT_COLLECTION,
        "collection_status": "green",
        "points_count": EXPECTED_QDRANT_POINTS,
        "required_ids": len(ids),
        "returned_ids": len(ids),
        "missing_ids": [],
        "payload_build_ids": [EXPECTED_KB_BUILD_ID],
    }:
        raise EvaluationBlocked(f"FINAL_EVALUATION_GOLD_BLOCKER: live Qdrant {report}")
    return report


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _runtime_environment() -> dict[str, Any]:
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    capability = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
    bfloat16_supported = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependency_lock_sha256": raw_sha256(ROOT / "requirements.lock.txt"),
        "torch": _package_version("torch"),
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "gpu_compute_capability": capability,
        "gpu_bfloat16_supported": bfloat16_supported,
        "sentence_transformers": _package_version("sentence-transformers"),
        "transformers": _package_version("transformers"),
        "ragchecker": _package_version("ragchecker"),
        "openai": _package_version("openai"),
        "spacy": _package_version("spacy"),
    }


def build_run_manifest(
    *, gold_report: dict[str, Any], qdrant_report: dict[str, Any]
) -> dict[str, Any]:
    from src.observability.versioning import (
        current_pipeline_fingerprint,
        get_answer_cache_version,
    )

    head = _git("rev-parse", "HEAD")
    ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", SYSTEM_UNDER_TEST_SHA, head],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )
    fingerprint = current_pipeline_fingerprint()
    cache_version = get_answer_cache_version()
    manifest = {
        "schema": "thesis_evaluation_run_manifest",
        "run_id": RUN_ID,
        "prepared_at": utc_now(),
        "system_under_test_sha": SYSTEM_UNDER_TEST_SHA,
        "repository_head_at_preflight": head,
        "system_under_test_is_ancestor": ancestor,
        "pipeline_fingerprint": fingerprint,
        "answer_cache_version": cache_version,
        "cache_mode": "isolated_cold_per_case",
        "production_cache_enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
        "kb_build_id": EXPECTED_KB_BUILD_ID,
        "qdrant_collection": qdrant_report["active_target"],
        "qdrant_points": qdrant_report["points_count"],
        "benchmark": {
            "path": str(BENCHMARK_PATH.relative_to(ROOT)).replace("\\", "/"),
            "canonical_sha256": gold_report["benchmark_canonical_sha256"],
            "total": gold_report["benchmark_total"],
            "answerable": gold_report["answerable"],
            "single_turn": gold_report["answerable_single_turn"],
            "multi_turn": gold_report["answerable_multi_turn"],
            "evidence_gap": gold_report["evidence_gap"],
        },
        "gold": {
            "source_commit": GOLD_SOURCE_COMMIT,
            "path": str(GOLD_PATH.relative_to(ROOT)).replace("\\", "/"),
            "canonical_sha256": gold_report["gold_canonical_sha256"],
            "raw_sha256": gold_report["gold_raw_sha256"],
            "reviewed_claims": gold_report["reviewed_claims"],
            "confirmed_absence": gold_report["confirmed_absence"],
            "policy_ambiguous": gold_report["policy_ambiguous"],
            "required_evidence_ids": gold_report["required_evidence_ids"],
        },
        "execution_policy": {
            "case_order": "stored benchmark order",
            "one_execution_per_case": True,
            "successful_case_rerun": False,
            "resume_only_after_external_infrastructure_failure": True,
            "production_retry_and_fallback_enabled": True,
            "qdrant_read_only": True,
            "knowledge_base_read_only": True,
            "production_redis_flushes": 0,
        },
        "generation": {
            "provider": os.getenv("LLM_PROVIDER", "gemini"),
            "model": os.getenv("GOOGLE_MODEL", "gemini-3.5-flash-lite"),
            "provider_fallback_enabled": os.getenv("LLM_PROVIDER_FALLBACK_ENABLED", "false"),
            "fallback_models": os.getenv("GOOGLE_FALLBACK_MODELS", "gemini-3.1-flash-lite"),
        },
        "reranker": {
            "enabled": os.getenv("RERANKER_ENABLED", "true"),
            "model": os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            "device": os.getenv("RERANKER_DEVICE", "cuda"),
            "precision": os.getenv("RERANKER_PRECISION", "bfloat16"),
        },
        "semantic_evaluator": {
            "package": f"ragchecker=={RAGCHECKER_VERSION}",
            "model": EVALUATOR_MODEL,
            "api": "OpenAI Responses API",
            "store": False,
            "ragas_excluded": True,
        },
        "environment": _runtime_environment(),
        "git": {
            "branch": _git("branch", "--show-current"),
            "dirty_at_preflight": bool(_git("status", "--short")),
            "production_sensitive_diff": production_sensitive_diff(),
        },
        "write_audit": {
            "qdrant_writes": 0,
            "knowledge_base_writes": 0,
            "production_redis_flushes": 0,
        },
        "protocol_commit": None,
        "system_output_commit": None,
        "final_results_commit": None,
    }
    identity_errors = {}
    for name, actual, expected in (
        ("pipeline_fingerprint", fingerprint, EXPECTED_PIPELINE_FINGERPRINT),
        ("answer_cache_version", cache_version, EXPECTED_CACHE_VERSION),
        ("ragchecker", manifest["environment"]["ragchecker"], RAGCHECKER_VERSION),
    ):
        if actual != expected:
            identity_errors[name] = {"expected": expected, "actual": actual}
    if not ancestor:
        identity_errors["system_under_test_is_ancestor"] = False
    if production_sensitive_diff():
        identity_errors["production_sensitive_diff"] = production_sensitive_diff()
    if not manifest["production_cache_enabled"]:
        identity_errors["production_cache_enabled"] = False
    if identity_errors:
        raise EvaluationBlocked(f"FINAL_EVALUATION_PREFLIGHT_BLOCKED: {identity_errors}")
    return manifest


def run_preflight() -> dict[str, Any]:
    load_dotenv(ROOT / ".env", override=False)
    if _git("branch", "--show-current") != "evaluation/thesis-final-evaluation":
        raise EvaluationBlocked("FINAL_EVALUATION_PREFLIGHT_BLOCKED: wrong branch")
    gold_report = validate_benchmark_and_gold()
    qdrant_report = validate_live_qdrant(gold_report["required_evidence_id_values"])
    manifest = build_run_manifest(gold_report=gold_report, qdrant_report=qdrant_report)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(RUN_MANIFEST_PATH, manifest)
    return {
        "status": "PREFLIGHT_READY",
        "run_manifest": str(RUN_MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
        "benchmark_and_gold": gold_report,
        "qdrant": qdrant_report,
        "production_sensitive_diff": production_sensitive_diff(),
    }


def _decision_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _extract_action_reason(result: dict[str, Any]) -> tuple[str | None, str | None]:
    decision = result.get("agent_decision")
    action = _decision_field(decision, "action")
    reason = _decision_field(decision, "reason_code")
    history = result.get("agent_decision_history") or []
    if history:
        action = action or _decision_field(history[-1], "action")
        reason = reason or _decision_field(history[-1], "reason_code")
    if result.get("fallback_type") == "evidence_gap":
        action = action or "abstain"
        reason = reason or "evidence_gap"
    return action, reason


def _ordered_unique(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _context_records(
    packed_context: dict[str, Any] | None,
    *,
    generation_invoked: bool,
    generation_trace: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], str, str]:
    if not generation_invoked:
        return [], "", text_sha256("")
    packed = packed_context if isinstance(packed_context, dict) else {}
    items = packed.get("items") or []
    context_text = str(packed.get("context_text") or "")
    from src.retrieval.context_packer import _render_block_from_item
    from src.retrieval.contracts import ContextItem

    records: list[dict[str, Any]] = []
    rendered_blocks: list[str] = []
    for position, raw_item in enumerate(items, start=1):
        item = ContextItem.model_validate(raw_item)
        rendered = _render_block_from_item(item, position)
        rendered_blocks.append(rendered)
        payload = item.payload
        records.append(
            {
                "position": position,
                "chunk_id": item.item_id,
                "source_id": payload.get("source_id")
                or payload.get("source_path")
                or payload.get("source_file")
                or payload.get("document_id"),
                "source_title": payload.get("source_title"),
                "source_url": payload.get("source_url"),
                "section_path": payload.get("section_path"),
                "text": item.text,
                "text_sha256": text_sha256(item.text),
                "rendered_text": rendered,
                "rendered_text_sha256": text_sha256(rendered),
            }
        )
    reconstructed = "\n\n".join(rendered_blocks)
    if reconstructed != context_text:
        raise EvaluationBlocked("FINAL_EVALUATION_CONTEXT_BLOCKER: rendered context mismatch")
    packed_ids = [item["chunk_id"] for item in records]
    trace_items = (generation_trace or {}).get("packed_evidence") or []
    trace_ids = [item.get("item_id") for item in trace_items]
    if packed_ids != trace_ids:
        raise EvaluationBlocked(
            "FINAL_EVALUATION_CONTEXT_BLOCKER: generation trace IDs differ from packed IDs"
        )
    limits = (packed.get("debug") or {}).get("limits") or {}
    if len(records) > 9 or len(context_text) > 7000:
        raise EvaluationBlocked("FINAL_EVALUATION_CONTEXT_BLOCKER: context budget exceeded")
    if limits and (limits.get("max_items") != 9 or limits.get("max_chars") != 7000):
        raise EvaluationBlocked(
            f"FINAL_EVALUATION_CONTEXT_BLOCKER: unexpected packer limits {limits}"
        )
    return records, context_text, text_sha256(context_text)


def _record_from_result(
    case: dict[str, Any],
    result: dict[str, Any],
    *,
    cache_empty_before: bool,
    cache_size_after: int,
    reranker_events: list[dict[str, Any]],
) -> dict[str, Any]:
    action, reason = _extract_action_reason(result)
    attempts = result.get("retrieval_attempt_traces") or []
    acquired = _ordered_unique(
        evidence_id
        for attempt in attempts
        for evidence_id in (attempt.get("retry_evidence") or {}).get("acquired_candidate_ids", [])
    )
    merged = _ordered_unique(
        evidence_id
        for attempt in attempts
        for evidence_id in (attempt.get("retry_evidence") or {}).get("eligible_candidate_ids", [])
    )
    generation_invoked = bool(result.get("generation_invoked"))
    contexts, context_text, context_hash = _context_records(
        result.get("packed_context"),
        generation_invoked=generation_invoked,
        generation_trace=result.get("generation_evidence_trace"),
    )
    history = case.get("history") or []
    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "category": case["category"],
        "query": case["query"],
        "history": history,
        "history_sha256": canonical_json_sha256(history),
        "history_message_count": len(history),
        "expected_benchmark_class": case.get("expected"),
        "final_action": action,
        "final_reason": reason,
        "final_answer": result.get("answer", ""),
        "acquisition_candidate_ids": acquired,
        "merged_retry_candidate_ids": merged,
        "retrieval_attempt_traces": attempts,
        "generation_contexts": contexts,
        "generation_context_text": context_text,
        "generation_context_text_sha256": context_hash,
        "sources": result.get("sources") or [],
        "source_allowlist": result.get("source_allowlist") or [],
        "source_validation": result.get("source_validation"),
        "structural_verification_report": result.get("structural_verification_report"),
        "answer_quality_report": result.get("answer_quality_report"),
        "retrieval_status": result.get("retrieval_status"),
        "retrieval_attempt_count": int(result.get("retrieval_attempt") or 0),
        "retry_history": result.get("retry_history") or [],
        "retrieval_trace": result.get("retrieval_trace"),
        "reranker_events": reranker_events,
        "requested_provider": result.get("requested_provider"),
        "requested_model": result.get("requested_model"),
        "actual_provider": result.get("actual_provider"),
        "actual_model": result.get("actual_model"),
        "generation_invoked": generation_invoked,
        "generation_provider": result.get("generation_provider"),
        "generation_model": result.get("generation_model"),
        "llm_fallback_used": bool(result.get("llm_fallback_used")),
        "fallback_provider": result.get("fallback_provider"),
        "fallback_model": result.get("fallback_model"),
        "fallback_chain": result.get("fallback_chain"),
        "fallback_applied": bool(result.get("fallback_applied")),
        "fallback_type": result.get("fallback_type"),
        "pipeline_fingerprint": result.get("pipeline_fingerprint"),
        "cache": {
            "enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
            "empty_before_case": cache_empty_before,
            "checked": result.get("cache_checked"),
            "hit": bool(result.get("cache_hit")),
            "reason": result.get("cache_reason"),
            "size_after_case": cache_size_after,
        },
        "timings_ms": result.get("performance_timings") or {},
        "runtime_resilience": result.get("runtime_resilience"),
        "error": None,
    }


def _allowed_live_status_paths() -> set[str]:
    return {
        str(RAW_RESULTS_PATH.relative_to(ROOT)).replace("\\", "/"),
        str(SYSTEM_OUTPUT_MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
    }


def validate_checkpoint_prefix(
    payload: dict[str, Any], cases: list[dict[str, Any]], protocol_commit: str
) -> int:
    """Resume only an identity-matched, duplicate-free benchmark-order prefix."""

    identity = payload.get("identity") or {}
    if identity.get("protocol_commit") != protocol_commit:
        raise EvaluationBlocked("Checkpoint protocol commit mismatch")
    records = payload.get("records") or []
    completed_ids = [record.get("case_id") for record in records]
    expected_prefix = [case["case_id"] for case in cases[: len(records)]]
    if completed_ids != expected_prefix or len(completed_ids) != len(set(completed_ids)):
        raise EvaluationBlocked("Checkpoint is not an exact benchmark-order prefix")
    if any(record.get("error") for record in records):
        raise EvaluationBlocked("Checkpoint contains a completed infrastructure-error row")
    return len(records)


def _require_live_gate() -> str:
    load_dotenv(ROOT / ".env", override=False)
    if _git("branch", "--show-current") != "evaluation/thesis-final-evaluation":
        raise EvaluationBlocked("FINAL_EVALUATION_PREFLIGHT_BLOCKED: wrong branch")
    head = _git("rev-parse", "HEAD")
    if head == SYSTEM_UNDER_TEST_SHA:
        raise EvaluationBlocked("FINAL_EVALUATION_PREFLIGHT_BLOCKED: protocol is not committed")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", SYSTEM_UNDER_TEST_SHA, head],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise EvaluationBlocked("SYSTEM_UNDER_TEST_SHA_CHANGED")
    if production_sensitive_diff():
        raise EvaluationBlocked("FINAL_EVALUATION_INVALID: production-sensitive diff")
    unexpected = []
    for line in _git("status", "--short").splitlines():
        path = line[3:].replace("\\", "/")
        if path not in _allowed_live_status_paths():
            unexpected.append(line)
    if unexpected:
        raise EvaluationBlocked(
            f"FINAL_EVALUATION_PREFLIGHT_BLOCKED: unexpected worktree state {unexpected}"
        )
    if os.getenv("THESIS_EVALUATION_CACHE_ISOLATED", "").lower() != "true":
        raise EvaluationBlocked(
            "FINAL_EVALUATION_PREFLIGHT_BLOCKED: isolated cache attestation missing"
        )
    if os.getenv("CACHE_ENABLED", "true").lower() != "true":
        raise EvaluationBlocked(
            "FINAL_EVALUATION_PREFLIGHT_BLOCKED: production cache must remain enabled"
        )
    return head


async def _cold_cache() -> tuple[Any, bool]:
    from src.cache.redis_cache import get_redis

    client = await get_redis()
    if client is None:
        raise EvaluationBlocked("FINAL_EVALUATION_PREFLIGHT_BLOCKED: Redis unavailable")
    await client.flushdb()
    empty = int(await client.dbsize()) == 0
    if not empty:
        raise EvaluationBlocked("FINAL_EVALUATION_PREFLIGHT_BLOCKED: cache not empty")
    return client, empty


async def run_system_cases() -> dict[str, Any]:
    protocol_commit = _require_live_gate()
    from src.agent.graph import run_clinical_agent
    import src.retrieval.service as retrieval_service

    benchmark = load_json(BENCHMARK_PATH)
    cases = benchmark["cases"]
    if RAW_RESULTS_PATH.exists():
        payload = load_json(RAW_RESULTS_PATH)
        validate_checkpoint_prefix(payload, cases, protocol_commit)
    else:
        payload = {
            "schema": "thesis_system_outputs",
            "identity": {
                "run_id": RUN_ID,
                "protocol_commit": protocol_commit,
                "system_under_test_sha": SYSTEM_UNDER_TEST_SHA,
                "pipeline_fingerprint": EXPECTED_PIPELINE_FINGERPRINT,
                "kb_build_id": EXPECTED_KB_BUILD_ID,
                "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
                "gold_canonical_sha256": EXPECTED_GOLD_CANONICAL_SHA256,
                "cache_mode": "isolated_cold_per_case",
            },
            "started_at": utc_now(),
            "completed_at": None,
            "incidents": [],
            "records": [],
        }
    original_rerank = retrieval_service.rerank_candidates
    start_index = len(payload["records"])
    for index, case in enumerate(cases[start_index:], start=start_index + 1):
        client, cache_empty = await _cold_cache()
        reranker_events: list[dict[str, Any]] = []

        async def observed_rerank(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            outcome = await original_rerank(*args, **kwargs)
            scorer = kwargs.get("scorer")
            reranker_events.append(
                {
                    "status": outcome.status,
                    "enabled": outcome.enabled,
                    "model": outcome.model_name,
                    "device": getattr(scorer, "device", None),
                    "precision": getattr(scorer, "precision", None),
                    "fallback_used": outcome.fallback_used,
                    "fallback_reason": outcome.fallback_reason,
                    "reported_elapsed_ms": outcome.elapsed_ms,
                    "observed_elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
            return outcome

        retrieval_service.rerank_candidates = observed_rerank
        try:
            result = await run_clinical_agent(
                message=case["query"],
                conversation_history=case.get("history") or [],
                allow_model_fallback=True,
                bypass_cache=False,
                include_generation_diagnostics=False,
                request_id=f"thesis-{index:03d}-{case['case_id']}",
            )
            cache_size_after = int(await client.dbsize())
            record = _record_from_result(
                case,
                result,
                cache_empty_before=cache_empty,
                cache_size_after=cache_size_after,
                reranker_events=reranker_events,
            )
            if record["pipeline_fingerprint"] != EXPECTED_PIPELINE_FINGERPRINT:
                raise EvaluationBlocked(
                    "FINAL_EVALUATION_INVALID: per-case pipeline fingerprint mismatch"
                )
        except Exception as exc:
            incident = {
                "timestamp": utc_now(),
                "case_id": case["case_id"],
                "last_completed_case": (
                    payload["records"][-1]["case_id"] if payload["records"] else None
                ),
                "resume_position": index,
                "type": type(exc).__name__,
                "message": str(exc)[:1000],
            }
            payload["incidents"].append(incident)
            atomic_write_json(RAW_RESULTS_PATH, payload)
            raise
        finally:
            retrieval_service.rerank_candidates = original_rerank
        payload["records"].append(record)
        atomic_write_json(RAW_RESULTS_PATH, payload)
        print(f"[{index:03d}/100] {case['case_id']} {record['final_action']}", flush=True)

    payload["completed_at"] = utc_now()
    atomic_write_json(RAW_RESULTS_PATH, payload)
    validate_complete_raw(payload, cases)
    output_manifest = {
        "schema": "thesis_system_output_manifest",
        "run_id": RUN_ID,
        "locked_at": utc_now(),
        "protocol_commit": protocol_commit,
        "record_count": 100,
        "case_order_sha256": canonical_json_sha256(
            [record["case_id"] for record in payload["records"]]
        ),
        "raw_results_canonical_sha256": canonical_json_file_sha256(RAW_RESULTS_PATH),
        "raw_results_raw_sha256": raw_sha256(RAW_RESULTS_PATH),
        "system_answers_locked": True,
    }
    atomic_write_json(SYSTEM_OUTPUT_MANIFEST_PATH, output_manifest)
    return output_manifest


def validate_complete_raw(raw: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    records = raw.get("records") or []
    ids = [record.get("case_id") for record in records]
    expected_ids = [case.get("case_id") for case in cases]
    errors = [record.get("case_id") for record in records if record.get("error")]
    if ids != expected_ids or len(set(ids)) != 100 or errors:
        raise EvaluationBlocked(
            f"FORMAL_RUN_BLOCKED_BY_INFRASTRUCTURE: records={len(records)}, errors={errors}"
        )


def _gold_indexes() -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    gold = load_json(GOLD_PATH)
    claims_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in gold["gold_claim_adjudications"]:
        claims_by_case[claim["case_id"]].append(claim)
    negative_status = {
        item["case_id"]: item["negative_status"] for item in gold["negative_adjudications"]
    }
    return dict(claims_by_case), negative_status


def _proportion(count: int, denominator: int) -> dict[str, Any]:
    return {
        "count": count,
        "denominator": denominator,
        "percentage": round((count / denominator) * 100, 1) if denominator else None,
    }


def _latency_stats(values: Iterable[Any]) -> dict[str, Any]:
    clean = sorted(
        float(value) for value in values if value is not None and math.isfinite(float(value))
    )
    if not clean:
        return {"count": 0, "min_ms": None, "median_ms": None, "p95_ms": None, "max_ms": None}
    p95 = clean[max(0, math.ceil(0.95 * len(clean)) - 1)]
    return {
        "count": len(clean),
        "min_ms": round(clean[0], 1),
        "median_ms": round(statistics.median(clean), 1),
        "p95_ms": round(p95, 1),
        "max_ms": round(clean[-1], 1),
    }


def _case_deterministic_row(
    case: dict[str, Any],
    record: dict[str, Any],
    claims: list[dict[str, Any]],
    negative_status: str | None,
) -> dict[str, Any]:
    required = _ordered_unique(
        evidence_id for claim in claims for evidence_id in claim.get("current_evidence_ids") or []
    )
    acquisition = set(record.get("acquisition_candidate_ids") or [])
    packed_ids = [item["chunk_id"] for item in record.get("generation_contexts") or []]
    packed = set(packed_ids)
    acquired_count = sum(evidence_id in acquisition for evidence_id in required)
    packed_count = sum(evidence_id in packed for evidence_id in required)
    complete_claims = sum(
        set(claim.get("current_evidence_ids") or []).issubset(packed) for claim in claims
    )
    complete_case = bool(required) and set(required).issubset(packed)
    strict_relevant = sum(evidence_id in set(required) for evidence_id in packed_ids)

    source_ids = _ordered_unique(record.get("sources") or [])
    allowlist_ids = {
        str(item.get("source_id"))
        for item in record.get("source_allowlist") or []
        if item.get("source_id")
    }
    citation_valid_count = sum(source_id in allowlist_ids for source_id in source_ids)
    contexts = record.get("generation_contexts") or []
    provenance_valid_count = sum(
        bool(item.get("chunk_id") and item.get("text") and item.get("source_id"))
        for item in contexts
    )
    structural = record.get("structural_verification_report") or {}
    source_validation = record.get("source_validation") or {}
    timings = record.get("timings_ms") or {}
    retrieval_ms = sum(
        float(value)
        for key, value in timings.items()
        if key.startswith("retrieval_attempt_") and value is not None
    )
    reranker_ms = sum(
        float(item.get("reported_elapsed_ms") or 0.0)
        for item in record.get("reranker_events") or []
    )
    pre_decision_ms = sum(
        float(timings.get(key) or 0.0) for key in ("prepare", "guard", "agent_decision_1")
    )
    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "category": case["category"],
        "negative_status": negative_status,
        "final_action": record.get("final_action"),
        "final_reason": record.get("final_reason"),
        "required_evidence_count": len(required),
        "acquired_required_evidence_count": acquired_count,
        "packed_required_evidence_count": packed_count,
        "reviewed_claim_count": len(claims),
        "complete_reviewed_claim_count": complete_claims,
        "complete_case": complete_case if claims else None,
        "packed_context_count": len(packed_ids),
        "strict_reviewed_id_context_count": strict_relevant,
        "confirmed_gap_rejection": (
            record.get("final_action") == "abstain" and record.get("final_reason") == "evidence_gap"
            if negative_status == "CONFIRMED_ABSENCE"
            else None
        ),
        "citation_id_count": len(source_ids),
        "valid_citation_id_count": citation_valid_count,
        "source_allowlist_valid": (
            int(source_validation.get("invalid_source_name_count") or 0) == 0
        ),
        "provenance_item_count": len(contexts),
        "valid_provenance_item_count": provenance_valid_count,
        "structural_verifier_pass": structural.get("passed") is True,
        "retrieval_attempt_count": record.get("retrieval_attempt_count", 0),
        "retried": int(record.get("retrieval_attempt_count") or 0) > 1,
        "provider_fallback": bool(record.get("llm_fallback_used")),
        "reranker_fallback": any(
            item.get("fallback_used") for item in record.get("reranker_events") or []
        ),
        "cache_hit": bool((record.get("cache") or {}).get("hit")),
        "error": record.get("error") is not None,
        "timeout": (record.get("error") or {}).get("type") in {"TimeoutError", "AgentTimeoutError"}
        if isinstance(record.get("error"), dict)
        else False,
        "end_to_end_ms": timings.get("agent_total"),
        "pre_decision_ms": round(pre_decision_ms, 3),
        "retrieval_ms": round(retrieval_ms, 3),
        "reranker_ms": round(reranker_ms, 3),
        "generation_ms": timings.get("llm_generation"),
        "finalization_ms": timings.get("finalize"),
        "generation_provider": record.get("generation_provider"),
        "generation_model": record.get("generation_model"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise EvaluationBlocked(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = sum(int(row["required_evidence_count"]) for row in rows)
    acquired = sum(int(row["acquired_required_evidence_count"]) for row in rows)
    packed = sum(int(row["packed_required_evidence_count"]) for row in rows)
    claims = sum(int(row["reviewed_claim_count"]) for row in rows)
    complete_claims = sum(int(row["complete_reviewed_claim_count"]) for row in rows)
    packed_total = sum(int(row["packed_context_count"]) for row in rows)
    strict_relevant = sum(int(row["strict_reviewed_id_context_count"]) for row in rows)
    return {
        "acquisition_required_evidence_recall": _proportion(acquired, required),
        "packed_required_evidence_recall": _proportion(packed, required),
        "complete_reviewed_claims": _proportion(complete_claims, claims),
        "complete_answerable_cases": _proportion(
            sum(row["complete_case"] is True for row in rows), len(rows)
        ),
        "strict_reviewed_id_context_precision": _proportion(strict_relevant, packed_total),
    }


def compute_deterministic_metrics() -> dict[str, Any]:
    benchmark = load_json(BENCHMARK_PATH)
    raw = load_json(RAW_RESULTS_PATH)
    output_manifest = load_json(SYSTEM_OUTPUT_MANIFEST_PATH)
    validate_complete_raw(raw, benchmark["cases"])
    if output_manifest["raw_results_canonical_sha256"] != canonical_json_file_sha256(
        RAW_RESULTS_PATH
    ) or output_manifest["raw_results_raw_sha256"] != raw_sha256(RAW_RESULTS_PATH):
        raise EvaluationBlocked("System output no longer matches its locked manifest")
    claims_by_case, negative_status = _gold_indexes()
    records_by_id = {record["case_id"]: record for record in raw["records"]}
    rows = [
        _case_deterministic_row(
            case,
            records_by_id[case["case_id"]],
            claims_by_case.get(case["case_id"], []),
            negative_status.get(case["case_id"]),
        )
        for case in benchmark["cases"]
    ]
    answerable = [row for row in rows if row["family"].startswith("answerable")]
    single = [row for row in answerable if row["family"] == "answerable_single_turn"]
    multi = [row for row in answerable if row["family"] == "answerable_multi_turn"]
    confirmed = [row for row in rows if row["negative_status"] == "CONFIRMED_ABSENCE"]
    ambiguous = [row for row in rows if row["negative_status"] == "POLICY_AMBIGUOUS"]
    if (len(answerable), len(single), len(multi), len(confirmed), len(ambiguous)) != (
        70,
        55,
        15,
        24,
        6,
    ):
        raise EvaluationBlocked("Deterministic metric denominator mismatch")
    evidence = _aggregate_evidence(answerable)
    if evidence["complete_reviewed_claims"]["denominator"] != 104:
        raise EvaluationBlocked("Reviewed claim denominator mismatch")

    citation_count = sum(row["citation_id_count"] for row in rows)
    provenance_count = sum(row["provenance_item_count"] for row in rows)
    ambiguous_distribution = Counter(row["final_action"] or "other/error" for row in ambiguous)
    operational = {
        "latency": {
            "end_to_end": _latency_stats(row["end_to_end_ms"] for row in rows),
            "pre_decision": _latency_stats(row["pre_decision_ms"] for row in rows),
            "retrieval": _latency_stats(row["retrieval_ms"] for row in rows),
            "reranker": _latency_stats(row["reranker_ms"] for row in rows),
            "generation": _latency_stats(row["generation_ms"] for row in rows),
            "finalization": _latency_stats(row["finalization_ms"] for row in rows),
            "p95_method": "nearest-rank",
        },
        "retrieval_attempt_distribution": dict(
            sorted(Counter(str(row["retrieval_attempt_count"]) for row in rows).items())
        ),
        "retry_rate": _proportion(sum(row["retried"] for row in rows), 100),
        "provider_fallback_rate": _proportion(sum(row["provider_fallback"] for row in rows), 100),
        "reranker_fallback_rate": _proportion(sum(row["reranker_fallback"] for row in rows), 100),
        "timeout_count": sum(row["timeout"] for row in rows),
        "error_count": sum(row["error"] for row in rows),
        "cache_hit_count": sum(row["cache_hit"] for row in rows),
        "generation_provider_model_distribution": dict(
            sorted(
                Counter(
                    f"{row['generation_provider'] or 'none'}::{row['generation_model'] or 'none'}"
                    for row in rows
                ).items()
            )
        ),
    }
    category_rows = []
    for category in sorted({row["category"] for row in rows}):
        group = [row for row in rows if row["category"] == category]
        group_answerable = [row for row in group if row["family"].startswith("answerable")]
        group_evidence = _aggregate_evidence(group_answerable) if group_answerable else None
        category_rows.append(
            {
                "category": category,
                "total_n": len(group),
                "answerable_n": len(group_answerable),
                "confirmed_gap_n": sum(
                    row["negative_status"] == "CONFIRMED_ABSENCE" for row in group
                ),
                "policy_ambiguous_n": sum(
                    row["negative_status"] == "POLICY_AMBIGUOUS" for row in group
                ),
                "acquisition_recall_percent": (
                    group_evidence["acquisition_required_evidence_recall"]["percentage"]
                    if group_evidence
                    else None
                ),
                "packed_recall_percent": (
                    group_evidence["packed_required_evidence_recall"]["percentage"]
                    if group_evidence
                    else None
                ),
                "complete_case_percent": (
                    group_evidence["complete_answerable_cases"]["percentage"]
                    if group_evidence
                    else None
                ),
                "confirmed_gap_rejection_percent": (
                    _proportion(
                        sum(row["confirmed_gap_rejection"] is True for row in group),
                        sum(row["negative_status"] == "CONFIRMED_ABSENCE" for row in group),
                    )["percentage"]
                ),
                "claim_recall_percent": None,
                "context_precision_percent": None,
                "faithfulness_percent": None,
                "claim_f1_percent": None,
            }
        )
    deterministic = {
        "schema": "thesis_deterministic_metrics",
        "computed_at": utc_now(),
        "source_raw_results_canonical_sha256": output_manifest["raw_results_canonical_sha256"],
        "denominators": {
            "all_cases": 100,
            "answerable": 70,
            "single_turn_answerable": 55,
            "multi_turn_answerable": 15,
            "reviewed_claims": 104,
            "confirmed_absence": 24,
            "policy_ambiguous": 6,
        },
        "answerable_action_distribution": dict(
            sorted(Counter(row["final_action"] or "other/error" for row in answerable).items())
        ),
        "answerable_completion_rate": _proportion(
            sum(row["final_action"] == "generate" for row in answerable), 70
        ),
        "evidence": evidence,
        "confirmed_evidence_gap_rejection": _proportion(
            sum(row["confirmed_gap_rejection"] is True for row in confirmed), 24
        ),
        "confirmed_evidence_gap_cases": [
            {
                "case_id": row["case_id"],
                "action": row["final_action"],
                "reason": row["final_reason"],
                "passed": row["confirmed_gap_rejection"],
            }
            for row in confirmed
        ],
        "policy_ambiguous_behavior": {
            "denominator": 6,
            "no_pass_fail_aggregate": True,
            "distribution": dict(sorted(ambiguous_distribution.items())),
            "cases": [
                {
                    "case_id": row["case_id"],
                    "action": row["final_action"],
                    "reason": row["final_reason"],
                }
                for row in ambiguous
            ],
        },
        "structural_provenance": {
            "citation_id_validity": _proportion(
                sum(row["valid_citation_id_count"] for row in rows), citation_count
            ),
            "source_allowlist_validity": _proportion(
                sum(row["source_allowlist_valid"] for row in rows), 100
            ),
            "provenance_validity": _proportion(
                sum(row["valid_provenance_item_count"] for row in rows),
                provenance_count,
            ),
            "structural_verifier_pass_rate": _proportion(
                sum(row["structural_verifier_pass"] for row in rows), 100
            ),
            "scope": "technical identity and structure only; no semantic entailment claim",
        },
        "subgroups": {
            "single_turn": {
                "denominator": 55,
                "evidence": _aggregate_evidence(single),
                "action_distribution": dict(
                    sorted(Counter(row["final_action"] or "other/error" for row in single).items())
                ),
            },
            "multi_turn": {
                "denominator": 15,
                "evidence": _aggregate_evidence(multi),
                "action_distribution": dict(
                    sorted(Counter(row["final_action"] or "other/error" for row in multi).items())
                ),
            },
        },
    }
    atomic_write_json(DETERMINISTIC_METRICS_PATH, deterministic)
    atomic_write_json(OPERATIONAL_METRICS_PATH, operational)
    _write_csv(CASE_METRICS_PATH, rows)
    _write_csv(CATEGORY_METRICS_PATH, category_rows)
    return deterministic


def _adapter_contract_ast(source: str) -> str:
    module = ast.parse(source)
    names = {
        "EVALUATOR_MODEL",
        "RAGCHECKER_VERSION",
        "EXTRACTION_REASONING_EFFORT",
        "CHECKING_REASONING_EFFORT",
    }
    functions = {
        "_ragchecker_prompt_role",
        "_ragchecker_batch_role",
        "build_openai_batch_adapter",
        "_make_ragchecker",
    }
    selected = []
    for node in module.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id in names for target in targets):
                selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in functions:
            selected.append(node)
    return ast.dump(ast.Module(body=selected, type_ignores=[]), include_attributes=False)


def adapter_contract_sha256(revision: str | None = None) -> str:
    if revision:
        source = _git_show_bytes(revision, "evaluation/formal_evaluation_support.py").decode(
            "utf-8"
        )
    else:
        source = (EVALUATION_DIR / "formal_evaluation_support.py").read_text(encoding="utf-8")
    return text_sha256(_adapter_contract_ast(source))


def validate_calibration_reuse() -> dict[str, Any]:
    results = load_json(CALIBRATION_RESULTS_PATH)
    adjudication = load_json(CALIBRATION_ADJUDICATION_PATH)
    historical_raw = load_json(CALIBRATION_RAW_RUN_PATH)
    runtime = _runtime_environment()
    historical_contract = adapter_contract_sha256(CALIBRATION_APPROVAL_RUN_SHA)
    current_contract = adapter_contract_sha256()
    current_calibration_sha = canonical_json_file_sha256(CALIBRATION_PATH)
    historical_metadata = historical_raw.get("run_metadata") or {}
    checks = {
        "calibration_sha_matches_results": current_calibration_sha
        == results.get("calibration_dataset_sha256"),
        "calibration_sha_matches_adjudication": current_calibration_sha
        == adjudication.get("calibration_dataset_sha256"),
        "evaluator_model_matches_results": EVALUATOR_MODEL == results.get("evaluator_model"),
        "evaluator_model_matches_adjudication": EVALUATOR_MODEL
        == adjudication.get("evaluator_model"),
        "ragchecker_version_matches_approved_run": RAGCHECKER_VERSION
        == historical_metadata.get("ragchecker_version"),
        "ragchecker_version_matches_runtime": RAGCHECKER_VERSION == runtime["ragchecker"],
        "adapter_contract_matches": historical_contract == current_contract,
        "researcher_approved": adjudication.get("researcher_adjudication") == "APPROVED",
        "effective_decision_ready": adjudication.get("effective_decision") == CALIBRATION_READY,
    }
    if not all(checks.values()):
        raise EvaluationBlocked(
            f"RAGCHECKER_CALIBRATION_REVIEW_REQUIRED: exact reuse not proven {checks}"
        )
    return {
        "decision": CALIBRATION_READY,
        "method": "reused_exact_researcher_approved_adjudication",
        "calibration_dataset_sha256": current_calibration_sha,
        "ragchecker_version": RAGCHECKER_VERSION,
        "evaluator_model": EVALUATOR_MODEL,
        "adapter_contract_sha256": current_contract,
        "approved_run_sha": CALIBRATION_APPROVAL_RUN_SHA,
        "adjudication_raw_sha256": raw_sha256(CALIBRATION_ADJUDICATION_PATH),
        "checks": checks,
    }


def _ratio_to_percent(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise EvaluationBlocked(f"Invalid RAGChecker per-case ratio: {value!r}")
    return round(numeric * 100.0, 1)


def _aggregate_percent(metrics: dict[str, Any], group: str, key: str) -> float | None:
    value = (metrics.get(group) or {}).get(key)
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 100.0:
        raise EvaluationBlocked(f"Invalid RAGChecker aggregate percentage: {value!r}")
    return round(numeric, 1)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _committed_raw_output_sha() -> str:
    relative = str(RAW_RESULTS_PATH.relative_to(ROOT)).replace("\\", "/")
    try:
        committed = _git_show_bytes("HEAD", relative)
    except subprocess.CalledProcessError as exc:
        raise EvaluationBlocked("Raw system output must be committed before scoring") from exc
    if committed != RAW_RESULTS_PATH.read_bytes():
        raise EvaluationBlocked("Working raw system output differs from committed bytes")
    return _git("log", "-1", "--format=%H", "--", relative)


def run_ragchecker_scoring() -> dict[str, Any]:
    """Score the committed outputs with the exact approved evaluator contract."""

    system_output_commit = _committed_raw_output_sha()
    calibration = validate_calibration_reuse()
    benchmark = load_json(BENCHMARK_PATH)
    raw = load_json(RAW_RESULTS_PATH)
    output_manifest = load_json(SYSTEM_OUTPUT_MANIFEST_PATH)
    validate_complete_raw(raw, benchmark["cases"])
    if output_manifest["raw_results_canonical_sha256"] != canonical_json_file_sha256(
        RAW_RESULTS_PATH
    ):
        raise EvaluationBlocked("RAGChecker input does not match system output manifest")

    from ragchecker import RAGChecker, RAGResult, RAGResults
    from ragchecker.container import RetrievedDoc
    from ragchecker.metrics import claim_recall, context_precision, f1, faithfulness

    cases = {case["case_id"]: case for case in benchmark["cases"]}
    answerable_records = [
        record for record in raw["records"] if record["family"].startswith("answerable")
    ]
    if len(answerable_records) != 70:
        raise EvaluationBlocked("RAGChecker answerable denominator is not 70")
    results = []
    for record in answerable_records:
        case = cases[record["case_id"]]
        rendered = [item["rendered_text"] for item in record["generation_contexts"]]
        if "\n\n".join(rendered) != record["generation_context_text"]:
            raise EvaluationBlocked(
                f"FINAL_EVALUATION_CONTEXT_BLOCKER: {record['case_id']} context changed"
            )
        contexts = [
            RetrievedDoc(doc_id=item["chunk_id"], text=item["rendered_text"])
            for item in record["generation_contexts"]
        ]
        results.append(
            RAGResult(
                query_id=case["case_id"],
                query=case["query"],
                gt_answer=case["gold_answer"],
                response=record["final_answer"],
                retrieved_context=contexts,
            )
        )

    base_adapter = build_openai_batch_adapter(EVALUATOR_MODEL)
    call_audit = {"batch_calls": 0, "prompt_requests": 0, "latency_ms": 0.0}

    def observed_adapter(prompts: list[str]) -> list[str]:
        started = time.perf_counter()
        outputs = base_adapter(prompts)
        call_audit["batch_calls"] += 1
        call_audit["prompt_requests"] += len(prompts)
        call_audit["latency_ms"] += (time.perf_counter() - started) * 1000
        return outputs

    evaluator = RAGChecker(
        extractor_name=EVALUATOR_MODEL,
        checker_name=EVALUATOR_MODEL,
        batch_size_extractor=8,
        batch_size_checker=8,
        joint_check=True,
        joint_check_num=5,
        custom_llm_api_func=observed_adapter,
    )
    rag_results = RAGResults(results=results)
    try:
        evaluator.evaluate(
            rag_results,
            metrics=[claim_recall, context_precision, faithfulness, f1],
            save_path=str(RAGCHECKER_CHECKPOINT_PATH),
        )
    except Exception as exc:
        message = str(exc).casefold()
        if "model" in message and any(
            marker in message for marker in ("not found", "does not exist", "unsupported")
        ):
            raise EvaluationBlocked(f"RAGCHECKER_EVALUATOR_BLOCKER: {exc}") from exc
        raise

    if len(rag_results.results) != 70:
        raise EvaluationBlocked("RAGChecker checkpoint does not contain 70 results")
    aggregate = {
        "Claim Recall": _aggregate_percent(
            rag_results.metrics, "retriever_metrics", "claim_recall"
        ),
        "Context Precision": _aggregate_percent(
            rag_results.metrics, "retriever_metrics", "context_precision"
        ),
        "Faithfulness": _aggregate_percent(
            rag_results.metrics, "generator_metrics", "faithfulness"
        ),
        "Claim F1": _aggregate_percent(rag_results.metrics, "overall_metrics", "f1"),
    }
    rag_by_id = {item.query_id: item for item in rag_results.results}
    case_rows = _read_csv(CASE_METRICS_PATH)
    for row in case_rows:
        result = rag_by_id.get(row["case_id"])
        metrics = result.metrics if result else {}
        row["claim_recall_percent"] = _ratio_to_percent(metrics.get("claim_recall"))
        row["context_precision_percent"] = _ratio_to_percent(metrics.get("context_precision"))
        row["faithfulness_percent"] = _ratio_to_percent(metrics.get("faithfulness"))
        row["claim_f1_percent"] = _ratio_to_percent(metrics.get("f1"))
    _write_csv(CASE_METRICS_PATH, case_rows)

    category_rows = _read_csv(CATEGORY_METRICS_PATH)
    for row in category_rows:
        group = [
            item
            for item in case_rows
            if item["category"] == row["category"] and item["case_id"] in rag_by_id
        ]
        for key in (
            "claim_recall_percent",
            "context_precision_percent",
            "faithfulness_percent",
            "claim_f1_percent",
        ):
            values = [float(item[key]) for item in group if item.get(key) not in {None, ""}]
            row[key] = round(statistics.mean(values), 1) if values else None
    _write_csv(CATEGORY_METRICS_PATH, category_rows)

    deterministic = load_json(DETERMINISTIC_METRICS_PATH)
    operational = load_json(OPERATIONAL_METRICS_PATH)
    summary_rows = [
        {
            "section": "Answerable semantic quality",
            "metric": name,
            "n": 70,
            "count": "",
            "score": score,
            "unit": "percent_0_100",
            "description_vi": description,
        }
        for name, score, description in (
            (
                "Claim Recall",
                aggregate["Claim Recall"],
                "Mức bằng chứng truy hồi bao phủ claim gold theo RAGChecker.",
            ),
            (
                "Context Precision",
                aggregate["Context Precision"],
                "Mức context liên quan tới claim gold theo RAGChecker.",
            ),
            (
                "Faithfulness",
                aggregate["Faithfulness"],
                "Mức claim trong final answer được exact generation context hỗ trợ.",
            ),
            (
                "Claim F1",
                aggregate["Claim F1"],
                "Overall claim-level F1 theo RAGChecker.",
            ),
        )
    ]
    evidence_labels = {
        "acquisition_required_evidence_recall": "Acquisition required-evidence recall",
        "packed_required_evidence_recall": "Packed required-evidence recall",
        "complete_reviewed_claims": "Complete reviewed claims",
        "complete_answerable_cases": "Complete answerable cases",
        "strict_reviewed_id_context_precision": "Strict reviewed-ID context precision",
    }
    for key, label in evidence_labels.items():
        value = deterministic["evidence"][key]
        summary_rows.append(
            {
                "section": "Deterministic retrieval/evidence",
                "metric": label,
                "n": value["denominator"],
                "count": value["count"],
                "score": value["percentage"],
                "unit": "percent_0_100",
                "description_vi": "Metric xác định từ reviewed evidence ID.",
            }
        )
    negative = deterministic["confirmed_evidence_gap_rejection"]
    summary_rows.append(
        {
            "section": "Evidence-gap behavior",
            "metric": "Confirmed evidence-gap rejection",
            "n": negative["denominator"],
            "count": negative["count"],
            "score": negative["percentage"],
            "unit": "percent_0_100",
            "description_vi": "Structured abstain + evidence_gap trên confirmed absence.",
        }
    )
    for key, value in deterministic["structural_provenance"].items():
        if isinstance(value, dict) and "denominator" in value:
            summary_rows.append(
                {
                    "section": "Structural/provenance",
                    "metric": key.replace("_", " ").title(),
                    "n": value["denominator"],
                    "count": value["count"],
                    "score": value["percentage"],
                    "unit": "percent_0_100",
                    "description_vi": "Kiểm tra contract kỹ thuật; không phải semantic entailment.",
                }
            )
    for key in ("end_to_end", "retrieval", "reranker", "generation"):
        value = operational["latency"][key]
        summary_rows.append(
            {
                "section": "Operational",
                "metric": f"{key.replace('_', ' ').title()} median",
                "n": value["count"],
                "count": "",
                "score": value["median_ms"],
                "unit": "milliseconds",
                "description_vi": "Median latency; bảng operational có min/p95/max.",
            }
        )
    _write_csv(METRICS_SUMMARY_PATH, summary_rows)

    manifest = load_json(RUN_MANIFEST_PATH)
    manifest["protocol_commit"] = output_manifest["protocol_commit"]
    manifest["system_output_commit"] = system_output_commit
    manifest["calibration"] = calibration
    manifest["ragchecker"] = {
        "completed_at": utc_now(),
        "checkpoint_path": str(RAGCHECKER_CHECKPOINT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "aggregate_percent": aggregate,
        "call_audit": {
            **call_audit,
            "latency_ms": round(float(call_audit["latency_ms"]), 1),
            "input_tokens": "not_exposed_by_existing_adapter",
            "output_tokens": "not_exposed_by_existing_adapter",
        },
    }
    atomic_write_json(RUN_MANIFEST_PATH, manifest)
    return {
        "calibration": calibration,
        "aggregate_percent": aggregate,
        "call_audit": call_audit,
    }


async def run_cache_smoke() -> dict[str, Any]:
    _require_live_gate()
    _committed_raw_output_sha()
    from src.agent.graph import run_clinical_agent

    client, empty = await _cold_cache()
    query = "Trong kiểm thử cache kỹ thuật, hãy định nghĩa ngắn gọn mụn trứng cá."
    first = await run_clinical_agent(
        message=query,
        allow_model_fallback=True,
        bypass_cache=False,
        request_id="thesis-cache-smoke-first",
    )
    second = await run_clinical_agent(
        message=query,
        allow_model_fallback=True,
        bypass_cache=False,
        request_id="thesis-cache-smoke-second",
    )
    result = {
        "executed_at": utc_now(),
        "isolated_cache_empty_before": empty,
        "first_cache_hit": bool(first.get("cache_hit")),
        "second_cache_hit": bool(second.get("cache_hit")),
        "first_answer_sha256": text_sha256(str(first.get("answer") or "")),
        "second_answer_sha256": text_sha256(str(second.get("answer") or "")),
        "answers_match": first.get("answer") == second.get("answer"),
        "passed": (
            not first.get("cache_hit")
            and bool(second.get("cache_hit"))
            and first.get("answer") == second.get("answer")
        ),
    }
    await client.flushdb()
    result["isolated_cache_empty_after"] = int(await client.dbsize()) == 0
    if not result["passed"] or not result["isolated_cache_empty_after"]:
        raise EvaluationBlocked(f"Cache smoke test failed: {result}")
    manifest = load_json(RUN_MANIFEST_PATH)
    manifest["cache_smoke"] = result
    atomic_write_json(RUN_MANIFEST_PATH, manifest)
    return result


def safety_regression_summary() -> dict[str, Any]:
    from src.agent.safety_policy import evaluate_safety

    fixture_path = ROOT / "tests" / "fixtures" / "safety_emergency_regression.json"
    fixture = load_json(fixture_path)
    rows = []
    for case in fixture["cases"]:
        decision = evaluate_safety(case["question"])
        rows.append(
            {
                "case_id": case["id"],
                "matched": decision is not None,
                "rule_id": decision.rule_id if decision else None,
            }
        )
    return {
        "label": "deterministic safety regression coverage",
        "fixture_sha256": canonical_json_file_sha256(fixture_path),
        "matched": sum(row["matched"] for row in rows),
        "denominator": len(rows),
        "percentage": round(sum(row["matched"] for row in rows) / len(rows) * 100, 1),
        "not_clinical_accuracy": True,
        "cases": rows,
    }


def _format_proportion(value: dict[str, Any]) -> str:
    return f"{value['count']}/{value['denominator']} ({value['percentage']:.1f}%)"


def _semantic_subgroup(case_rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    rows = [row for row in case_rows if row["family"] == family]
    result: dict[str, Any] = {"n": len(rows)}
    for key in (
        "claim_recall_percent",
        "context_precision_percent",
        "faithfulness_percent",
        "claim_f1_percent",
    ):
        values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
        result[key] = round(statistics.mean(values), 1) if values else None
    return result


def write_report_and_artifact_manifest() -> dict[str, Any]:
    run_manifest = load_json(RUN_MANIFEST_PATH)
    output_manifest = load_json(SYSTEM_OUTPUT_MANIFEST_PATH)
    deterministic = load_json(DETERMINISTIC_METRICS_PATH)
    operational = load_json(OPERATIONAL_METRICS_PATH)
    checkpoint = load_json(RAGCHECKER_CHECKPOINT_PATH)
    case_rows = _read_csv(CASE_METRICS_PATH)
    category_rows = _read_csv(CATEGORY_METRICS_PATH)
    if len(case_rows) != 100 or len(category_rows) != 13:
        raise EvaluationBlocked("Final CSV row-count validation failed")
    if len(checkpoint.get("results") or []) != 70:
        raise EvaluationBlocked("Final RAGChecker checkpoint denominator is not 70")
    semantic = run_manifest["ragchecker"]["aggregate_percent"]
    single = _semantic_subgroup(case_rows, "answerable_single_turn")
    multi = _semantic_subgroup(case_rows, "answerable_multi_turn")
    safety = safety_regression_summary()
    evidence = deterministic["evidence"]
    structural = deterministic["structural_provenance"]
    negative = deterministic["confirmed_evidence_gap_rejection"]
    ambiguous = deterministic["policy_ambiguous_behavior"]
    latency = operational["latency"]

    category_table = "\n".join(
        "| {category} | {answerable_n} | {claim_recall_percent} | "
        "{context_precision_percent} | {faithfulness_percent} | {claim_f1_percent} | "
        "{packed_recall_percent} |".format(**row)
        for row in category_rows
    )
    ambiguous_distribution = ", ".join(
        f"{key}={value}" for key, value in ambiguous["distribution"].items()
    )
    checksum_paths = [
        RUN_MANIFEST_PATH,
        SYSTEM_OUTPUT_MANIFEST_PATH,
        RAW_RESULTS_PATH,
        DETERMINISTIC_METRICS_PATH,
        CASE_METRICS_PATH,
        CATEGORY_METRICS_PATH,
        OPERATIONAL_METRICS_PATH,
        RAGCHECKER_CHECKPOINT_PATH,
        METRICS_SUMMARY_PATH,
    ]
    checksum_table = "\n".join(
        "| `{path}` | `{digest}` | {size} |".format(
            path=str(path.relative_to(ROOT)).replace("\\", "/"),
            digest=raw_sha256(path),
            size=path.stat().st_size,
        )
        for path in checksum_paths
    )
    report = f"""# Báo cáo đánh giá luận văn cuối cùng

## Tóm tắt phương pháp

System-under-test là commit `{SYSTEM_UNDER_TEST_SHA}` với pipeline fingerprint `{EXPECTED_PIPELINE_FINGERPRINT}`, cache `{EXPECTED_CACHE_VERSION}` và KB `{EXPECTED_KB_BUILD_ID}`. Benchmark `{EXPECTED_BENCHMARK_SHA256}` có 100 ca; gold từ commit `{GOLD_SOURCE_COMMIT}` có 104 claim đã review. Giao thức được commit tại `{run_manifest["protocol_commit"]}` trước request chính thức; raw output được commit tại `{run_manifest["system_output_commit"]}` trước semantic scoring.

RAGChecker 0.1.9 dùng evaluator `{EVALUATOR_MODEL}` và exact packed generation context. Calibration researcher-approved được tái sử dụng sau khi khớp calibration SHA, package/model và adapter AST contract. RAGAS không được dùng vì nghiên cứu trước dừng ở `RAGAS_CALIBRATION_BLOCKER`.

Cache production vẫn bật nhưng 100 ca dùng Redis evaluation riêng, rỗng trước từng ca (`isolated_cold_per_case`). Qdrant/KB chỉ đọc; write audit: Qdrant 0, KB 0, production Redis flush 0.

## TABLE A — Final semantic RAG quality

| Metric | N | Score |
|---|---:|---:|
| Claim Recall | 70 | {semantic["Claim Recall"]:.1f}% |
| Context Precision | 70 | {semantic["Context Precision"]:.1f}% |
| Faithfulness | 70 | {semantic["Faithfulness"]:.1f}% |
| Claim F1 | 70 | {semantic["Claim F1"]:.1f}% |

## TABLE B — Retrieval/evidence coverage

| Metric | Count / denominator | Score |
|---|---:|---:|
| Acquisition required-evidence recall | {evidence["acquisition_required_evidence_recall"]["count"]}/{evidence["acquisition_required_evidence_recall"]["denominator"]} | {evidence["acquisition_required_evidence_recall"]["percentage"]:.1f}% |
| Packed required-evidence recall | {evidence["packed_required_evidence_recall"]["count"]}/{evidence["packed_required_evidence_recall"]["denominator"]} | {evidence["packed_required_evidence_recall"]["percentage"]:.1f}% |
| Complete reviewed claims | {evidence["complete_reviewed_claims"]["count"]}/104 | {evidence["complete_reviewed_claims"]["percentage"]:.1f}% |
| Complete answerable cases | {evidence["complete_answerable_cases"]["count"]}/70 | {evidence["complete_answerable_cases"]["percentage"]:.1f}% |
| Strict reviewed-ID context precision | {evidence["strict_reviewed_id_context_precision"]["count"]}/{evidence["strict_reviewed_id_context_precision"]["denominator"]} | {evidence["strict_reviewed_id_context_precision"]["percentage"]:.1f}% |

Strict reviewed-ID precision có thể phạt context hữu ích không nằm trong gold ID; không được diễn giải là relevance truth phổ quát.

## TABLE C — Evidence-gap handling

| Nhóm | N | Kết quả |
|---|---:|---|
| Confirmed evidence-gap rejection | 24 | {_format_proportion(negative)} |
| Policy-ambiguous behavior | 6 | {ambiguous_distribution}; không aggregate pass/fail |

## TABLE D — Single-turn vs multi-turn

| Nhóm | N | Claim Recall | Context Precision | Faithfulness | Claim F1 | Packed recall |
|---|---:|---:|---:|---:|---:|---:|
| Single-turn | 55 | {single["claim_recall_percent"]:.1f}% | {single["context_precision_percent"]:.1f}% | {single["faithfulness_percent"]:.1f}% | {single["claim_f1_percent"]:.1f}% | {deterministic["subgroups"]["single_turn"]["evidence"]["packed_required_evidence_recall"]["percentage"]:.1f}% |
| Multi-turn | 15 | {multi["claim_recall_percent"]:.1f}% | {multi["context_precision_percent"]:.1f}% | {multi["faithfulness_percent"]:.1f}% | {multi["claim_f1_percent"]:.1f}% | {deterministic["subgroups"]["multi_turn"]["evidence"]["packed_required_evidence_recall"]["percentage"]:.1f}% |

Các subgroup semantic là macro mean của metric per-case do package RAGChecker tạo; metric không được định nghĩa lại.

## TABLE E — Category-level performance

| Category | Answerable N | Claim Recall | Context Precision | Faithfulness | Claim F1 | Packed recall |
|---|---:|---:|---:|---:|---:|---:|
{category_table}

## TABLE F — Operational performance

| Stage | N | Min ms | Median ms | P95 ms | Max ms |
|---|---:|---:|---:|---:|---:|
| End-to-end | {latency["end_to_end"]["count"]} | {latency["end_to_end"]["min_ms"]} | {latency["end_to_end"]["median_ms"]} | {latency["end_to_end"]["p95_ms"]} | {latency["end_to_end"]["max_ms"]} |
| Pre-decision | {latency["pre_decision"]["count"]} | {latency["pre_decision"]["min_ms"]} | {latency["pre_decision"]["median_ms"]} | {latency["pre_decision"]["p95_ms"]} | {latency["pre_decision"]["max_ms"]} |
| Retrieval | {latency["retrieval"]["count"]} | {latency["retrieval"]["min_ms"]} | {latency["retrieval"]["median_ms"]} | {latency["retrieval"]["p95_ms"]} | {latency["retrieval"]["max_ms"]} |
| Reranker | {latency["reranker"]["count"]} | {latency["reranker"]["min_ms"]} | {latency["reranker"]["median_ms"]} | {latency["reranker"]["p95_ms"]} | {latency["reranker"]["max_ms"]} |
| Generation | {latency["generation"]["count"]} | {latency["generation"]["min_ms"]} | {latency["generation"]["median_ms"]} | {latency["generation"]["p95_ms"]} | {latency["generation"]["max_ms"]} |
| Finalization | {latency["finalization"]["count"]} | {latency["finalization"]["min_ms"]} | {latency["finalization"]["median_ms"]} | {latency["finalization"]["p95_ms"]} | {latency["finalization"]["max_ms"]} |

P95 dùng nearest-rank.

## TABLE G — Reliability / fallback / timeout

| Metric | Count / denominator | Score |
|---|---:|---:|
| Retry | {operational["retry_rate"]["count"]}/100 | {operational["retry_rate"]["percentage"]:.1f}% |
| Provider fallback | {operational["provider_fallback_rate"]["count"]}/100 | {operational["provider_fallback_rate"]["percentage"]:.1f}% |
| Reranker fallback | {operational["reranker_fallback_rate"]["count"]}/100 | {operational["reranker_fallback_rate"]["percentage"]:.1f}% |
| Timeout | {operational["timeout_count"]}/100 | — |
| Error | {operational["error_count"]}/100 | — |
| Cache hit trong formal cases | {operational["cache_hit_count"]}/100 | — |

## Provenance, cấu trúc và safety regression

- Citation ID validity: {_format_proportion(structural["citation_id_validity"])}.
- Source allowlist validity: {_format_proportion(structural["source_allowlist_validity"])}.
- Provenance validity: {_format_proportion(structural["provenance_validity"])}.
- Structural verifier pass: {_format_proportion(structural["structural_verifier_pass_rate"])}.
- Deterministic safety regression coverage: {safety["matched"]}/{safety["denominator"]} ({safety["percentage"]:.1f}%). Đây là regression coverage, không phải clinical safety accuracy.

## Tính toàn vẹn thực thi và khả năng tái lập

Đủ 100 case ID duy nhất theo stored order; final answer được lấy sau finalize/verifier. Raw-results canonical SHA-256 là `{output_manifest["raw_results_canonical_sha256"]}` và raw-byte SHA-256 là `{output_manifest["raw_results_raw_sha256"]}`. Exact generation contexts được giữ nguyên cả rendered text và hash, đồng thời kiểm tra trùng generation trace. Cache smoke: `{run_manifest.get("cache_smoke", {}).get("passed")}`.

Môi trường: Python {run_manifest["environment"]["python"]}, torch {run_manifest["environment"]["torch"]}, CUDA {run_manifest["environment"]["torch_cuda"]}, GPU `{run_manifest["environment"]["gpu"]}`, sentence-transformers {run_manifest["environment"]["sentence_transformers"]}, transformers {run_manifest["environment"]["transformers"]}.

### Checksum artifact

| Artifact | Raw SHA-256 | Bytes |
|---|---|---:|
{checksum_table}

`THESIS_EVALUATION_REPORT.md` và `artifact_manifest.json` không nằm trong bảng tự tham chiếu này; hash cuối của báo cáo được ghi trong artifact manifest, còn manifest tự loại chính nó để tránh vòng lặp mật mã.

## Hạn chế và threats to validity

- Benchmark chỉ có 100 ca, chuyên biệt miền acne và có composition 70 answerable/30 evidence-gap.
- Chỉ 24 negative được xác nhận absence; 6 ca policy-ambiguous được tách riêng.
- Model sinh có tính stochastic; đây là một lần chạy theo protocol đã khóa.
- RAGChecker phụ thuộc LLM evaluator và snapshot evaluator đã chọn.
- Strict reviewed-ID precision có thể đánh giá thấp context hữu ích ngoài gold ID.
- Whole-chunk packer 9/7000 là resource policy, chưa được chứng minh tối ưu toàn cục.
- Bằng chứng cải thiện retry còn giới hạn; BF16 phụ thuộc phần cứng, thư viện và model cache.
- So sánh lịch sử không cho phép suy luận nhân quả vì runtime/KB/pipeline khác nhau.
- Kết quả không thiết lập clinical efficacy hoặc clinical safety accuracy.

## Kết luận dùng cho luận văn

Trên bộ benchmark và KB đã khóa, hệ thống được mô tả bằng các metric semantic, evidence, provenance và operational nêu trên. Kết quả phản ánh hiệu năng kỹ thuật của đúng snapshot SUT trong điều kiện cache lạnh cô lập; chúng không cấu thành bằng chứng về hiệu quả lâm sàng. Mọi kết quả âm và giới hạn denominator được giữ nguyên trong báo cáo.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

    required = [
        RUN_MANIFEST_PATH,
        SYSTEM_OUTPUT_MANIFEST_PATH,
        RAW_RESULTS_PATH,
        DETERMINISTIC_METRICS_PATH,
        CASE_METRICS_PATH,
        CATEGORY_METRICS_PATH,
        OPERATIONAL_METRICS_PATH,
        RAGCHECKER_CHECKPOINT_PATH,
        METRICS_SUMMARY_PATH,
        REPORT_PATH,
    ]
    artifacts = [
        {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "raw_sha256": raw_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in required
    ]
    artifact_manifest = {
        "schema": "thesis_evaluation_artifact_manifest",
        "created_at": utc_now(),
        "run_id": RUN_ID,
        "system_output_canonical_sha256": output_manifest["raw_results_canonical_sha256"],
        "artifacts": artifacts,
        "self_hash_exclusion": (
            "artifact_manifest.json is excluded from its own list because a file cannot "
            "contain its final cryptographic hash without a circular definition"
        ),
    }
    atomic_write_json(ARTIFACT_MANIFEST_PATH, artifact_manifest)
    return {
        "report": str(REPORT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "artifact_manifest_raw_sha256": raw_sha256(ARTIFACT_MANIFEST_PATH),
        "artifact_count": len(artifacts),
        "safety": safety,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "preflight",
            "run-system",
            "deterministic",
            "calibration-check",
            "ragchecker",
            "cache-smoke",
            "finalize",
        ),
    )
    args = parser.parse_args()
    if args.command == "preflight":
        result = run_preflight()
    elif args.command == "run-system":
        result = asyncio.run(run_system_cases())
    elif args.command == "deterministic":
        result = compute_deterministic_metrics()
    elif args.command == "calibration-check":
        result = validate_calibration_reuse()
    elif args.command == "ragchecker":
        result = run_ragchecker_scoring()
    elif args.command == "cache-smoke":
        result = asyncio.run(run_cache_smoke())
    else:
        result = write_report_and_artifact_manifest()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
