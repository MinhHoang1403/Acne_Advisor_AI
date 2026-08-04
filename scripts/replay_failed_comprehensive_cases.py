"""Replay selected canonical comprehensive-evaluation cases through public ``/chat``.

The runner is intentionally diagnostic only: it reads the frozen V1 artifacts,
uses a fresh replay session per request, and writes its output outside the
canonical evaluation report. The public API persists chat sessions, so
``--no-persistence`` is rejected rather than silently claiming no side effects.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "notebooks" / "eval_data" / "acne_rag_eval_comprehensive_v1.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "evaluation" / "20260804T072950Z_final_comprehensive_v1"


def _load_cases(dataset_path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(case["id"]): case
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for case in [json.loads(line)]
    }


def _previous_results(report_dir: Path) -> dict[str, dict[str, str]]:
    with (report_dir / "case_results.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        return {str(row["case_id"]): row for row in csv.DictReader(handle)}


def _select_cases(
    cases: dict[str, dict[str, Any]],
    *,
    previous_results: dict[str, dict[str, str]],
    failures_only: bool,
    categories: set[str],
    case_ids: set[str],
    previous_origins: set[str],
    previous_route_mismatch: bool,
    stratified_per_category: int | None,
) -> list[dict[str, Any]]:
    failed_ids = {
        case_id
        for case_id, result in previous_results.items()
        if str(result.get("failure_reasons") or "").strip() not in {"", "[]"}
    }
    selected: list[dict[str, Any]] = []
    for case in cases.values():
        case_id = str(case["id"])
        previous = previous_results.get(case_id, {})
        if failures_only and case_id not in failed_ids:
            continue
        if categories and str(case.get("category")) not in categories:
            continue
        if case_ids and case_id not in case_ids:
            continue
        if previous_origins and str(previous.get("actual_origin") or "") not in previous_origins:
            continue
        if previous_route_mismatch and str(previous.get("route_match") or "") != "False":
            continue
        selected.append(case)
    if not stratified_per_category:
        return sorted(selected, key=lambda item: str(item["id"]))

    by_category: dict[str, list[dict[str, Any]]] = {}
    for case in selected:
        by_category.setdefault(str(case["category"]), []).append(case)
    stratified: list[dict[str, Any]] = []
    for category in sorted(by_category):
        category_cases = sorted(
            by_category[category],
            key=lambda case: (str(case["id"]) not in failed_ids, str(case["id"])),
        )
        stratified.extend(category_cases[:stratified_per_category])
    return stratified


def _replay_case(case: dict[str, Any], args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    payload = {
        "message": case["question"],
        "session_id": f"comprehensive-replay-{run_id}-{case['id']}",
        "user_id": "comprehensive-replay",
        "conversation_history": case.get("conversation_history") or [],
        "llm_provider": args.provider,
        "llm_model": args.model,
        "allow_model_fallback": False,
        "bypass_cache": True,
    }
    started = time.perf_counter()
    try:
        response = requests.post(
            f"{args.api_base_url.rstrip('/')}/chat",
            json=payload,
            timeout=args.timeout_seconds,
        )
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {"raw_text": response.text[:1000]}
        metadata = body.get("metadata") if isinstance(body, dict) else {}
        return {
            "case_id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "ok": response.ok,
            "http_status": response.status_code,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "answer": body.get("answer") if isinstance(body, dict) else None,
            "actual_provider": metadata.get("provider") if isinstance(metadata, dict) else None,
            "actual_model": metadata.get("model") if isinstance(metadata, dict) else None,
            "response_origin": metadata.get("response_origin") if isinstance(metadata, dict) else None,
            "fallback_reason": metadata.get("fallback_reason") if isinstance(metadata, dict) else None,
            "medical_severity": metadata.get("medical_severity") if isinstance(metadata, dict) else None,
            "severity_guard_modified": metadata.get("severity_guard_modified") if isinstance(metadata, dict) else None,
            "error": None if response.ok else str(body)[:500],
        }
    except requests.RequestException as exc:
        return {
            "case_id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "ok": False,
            "http_status": None,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "answer": None,
            "actual_provider": None,
            "actual_model": None,
            "response_origin": None,
            "fallback_reason": None,
            "medical_severity": None,
            "severity_guard_modified": None,
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reference-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--failures-only", action="store_true")
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--previous-origin", action="append", default=[])
    parser.add_argument("--previous-route-mismatch", action="store_true")
    parser.add_argument("--stratified-per-category", type=int)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--no-persistence", action="store_true")
    args = parser.parse_args()
    if args.no_persistence:
        raise SystemExit(
            "BLOCKED: --no-persistence is not supported by the public /chat contract; no runtime bypass was added."
        )

    dataset_path = args.dataset.resolve()
    report_dir = args.reference_report.resolve()
    if args.stratified_per_category is not None and args.stratified_per_category <= 0:
        raise SystemExit("--stratified-per-category must be a positive integer.")
    cases = _load_cases(dataset_path)
    previous_results = _previous_results(report_dir)
    selected = _select_cases(
        cases,
        previous_results=previous_results,
        failures_only=args.failures_only,
        categories=set(args.category),
        case_ids=set(args.case_id),
        previous_origins=set(args.previous_origin),
        previous_route_mismatch=args.previous_route_mismatch,
        stratified_per_category=args.stratified_per_category,
    )
    if not selected:
        raise SystemExit("No canonical cases matched the requested replay filters.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (args.output_dir or ROOT / "reports" / "audit_final" / f"{timestamp}_replay").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    run_id = output_dir.name
    results = [_replay_case(case, args, run_id) for case in selected]
    output_path = output_dir / "replay_results.jsonl"
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )
    summary = {
        "reference_report": str(report_dir),
        "dataset": str(dataset_path),
        "provider": args.provider,
        "model": args.model,
        "bypass_cache": True,
        "case_count": len(results),
        "success_count": sum(bool(row["ok"]) for row in results),
        "failed_case_ids": [row["case_id"] for row in results if not row["ok"]],
        "selection": {
            "failures_only": args.failures_only,
            "categories": args.category,
            "case_ids": args.case_id,
            "previous_origins": args.previous_origin,
            "previous_route_mismatch": args.previous_route_mismatch,
            "stratified_per_category": args.stratified_per_category,
            "prioritizes_previous_failures": args.stratified_per_category is not None,
        },
        "persistence_note": "Public /chat creates replay sessions in PostgreSQL; no cache or database cleanup was performed.",
    }
    (output_dir / "replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"REPLAY_RESULTS={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
