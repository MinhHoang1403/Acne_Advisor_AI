"""Checkpointed two-stage CLI runner for the canonical Evaluation V3 framework."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoint import append_jsonl, assert_resume_compatible, atomic_write_json, completed_ids, read_jsonl
from .deterministic import deterministic_result, judge_agrees_with_deterministic, summarize_metrics
from .judge_gemini import judge_case, summarize_judge
from .live_eval import assert_isolated, component_checks, run_live_case
from .models import DATASET_SCHEMA_VERSION, JUDGE_RUBRIC_VERSION, METRICS_VERSION, EvaluationConfig
from .plots import create_plots
from .report_vi import render_report
from .validators import load_cases, sha256_file, validate_cases


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["case_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


class FinalEvaluationRunner:
    """Owns one frozen V3 run directory and its stage-specific checkpoints."""

    def __init__(self, config: EvaluationConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root.resolve()
        self.dataset_path = config.dataset_path.resolve()
        self.cases = load_cases(self.dataset_path)
        validate_cases(self.cases, self.project_root)
        self.dataset_sha256 = sha256_file(self.dataset_path)

    def _selected_cases(self, limit: int) -> list[dict[str, Any]]:
        if self.config.case_ids:
            by_id = {str(case["id"]): case for case in self.cases}
            missing = [case_id for case_id in self.config.case_ids if case_id not in by_id]
            if missing:
                raise ValueError("Unknown Evaluation V3 case IDs: " + ", ".join(missing))
            return [by_id[case_id] for case_id in self.config.case_ids]
        if limit >= len(self.cases):
            return list(self.cases)
        if limit <= 0:
            raise ValueError("Case limit must be positive")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in self.cases:
            grouped[str(case["category"])].append(case)
        if limit % len(grouped) == 0:
            per_category = limit // len(grouped)
            return [case for category in sorted(grouped) for case in grouped[category][:per_category]]
        return list(self.cases[:limit])

    def _new_run_dir(self) -> Path:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_final_evaluation_v3"
        path = self.config.report_root / run_id
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _latest_run_dir(self) -> Path:
        candidates = sorted(
            path
            for path in self.config.report_root.glob("*_final_evaluation_v3")
            if (path / "evaluation_manifest.json").exists()
        )
        if not candidates:
            raise FileNotFoundError("No Evaluation V3 run is available to resume")
        return candidates[-1]

    def _run_dir(self, *, resume: bool) -> Path:
        return self._latest_run_dir() if resume else self._new_run_dir()

    def _manifest(self, run_dir: Path) -> dict[str, Any]:
        path = run_dir / "evaluation_manifest.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _write_manifest(self, run_dir: Path, manifest: dict[str, Any]) -> None:
        manifest["updated_at"] = _utc_now()
        atomic_write_json(run_dir / "evaluation_manifest.json", manifest)

    def _ensure_run(self, *, resume: bool) -> tuple[Path, dict[str, Any]]:
        run_dir = self._run_dir(resume=resume)
        manifest = self._manifest(run_dir)
        if manifest:
            current_config_sha = _json_hash(self.config.as_json())
            if manifest.get("config_sha256") != current_config_sha:
                raise ValueError("Resume rejected because the saved evaluation configuration differs.")
            assert_resume_compatible(
                manifest,
                dataset_sha256=self.dataset_sha256,
                provider=self.config.live_provider,
                model=self.config.live_model,
                version=METRICS_VERSION,
                stage="live",
            )
            return run_dir, manifest
        config_data = self.config.as_json()
        manifest = {
            "run_id": run_dir.name,
            "created_at": _utc_now(),
            "git_commit": _git_commit(self.project_root),
            "dataset_path": str(self.dataset_path),
            "dataset_sha256": self.dataset_sha256,
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "metrics_version": METRICS_VERSION,
            "judge_rubric_version": JUDGE_RUBRIC_VERSION,
            "live_provider": self.config.live_provider,
            "live_model": self.config.live_model,
            "judge_provider": self.config.judge_provider,
            "judge_model": self.config.judge_model,
            "config_sha256": _json_hash(config_data),
            "live_stage": "not_started",
            "judge_stage": "not_started",
            "finalized": False,
        }
        atomic_write_json(run_dir / "run_config.json", config_data)
        self._write_manifest(run_dir, manifest)
        return run_dir, manifest

    def run_live(self, *, resume: bool = False) -> Path:
        assert_isolated(self.config)
        run_dir, manifest = self._ensure_run(resume=resume)
        if manifest.get("live_stage") == "live_completed":
            return run_dir
        manifest["live_stage"] = "live_running"
        manifest["live_started_at"] = manifest.get("live_started_at") or _utc_now()
        self._write_manifest(run_dir, manifest)
        raw_path = run_dir / "raw_responses.jsonl"
        result_path = run_dir / "case_results.jsonl"
        done = completed_ids(raw_path)
        cases = self._selected_cases(self.config.question_limit)
        by_id = {case["id"]: case for case in cases}
        for case in cases:
            if case["id"] in done:
                continue
            raw = run_live_case(case, self.config)
            result = deterministic_result(raw, case, self.config.live_provider, self.config.live_model)
            append_jsonl(raw_path, raw)
            append_jsonl(result_path, result)
            manifest["live_completed_case_count"] = len(completed_ids(raw_path))
            self._write_manifest(run_dir, manifest)
        raw_rows = [row for row in read_jsonl(raw_path) if row.get("case_id") in by_id]
        results = [row for row in read_jsonl(result_path) if row.get("case_id") in by_id]
        if len(raw_rows) != len(cases) or len(results) != len(cases):
            raise RuntimeError("Live checkpoint is incomplete after execution")
        metrics = summarize_metrics(results)
        checks = component_checks()
        checks["case_count"] = {"passed": len(cases) == self.config.question_limit, "value": len(cases)}
        _write_csv(run_dir / "case_results.csv", results)
        self._write_category_summary(run_dir, results)
        atomic_write_json(run_dir / "retrieval_metrics.json", metrics["retrieval_and_grounding"])
        atomic_write_json(run_dir / "safety_metrics.json", metrics["safety_and_scope"])
        atomic_write_json(run_dir / "performance_metrics.json", metrics["performance"])
        atomic_write_json(run_dir / "naturalness_metrics.json", metrics["naturalness_user_experience"])
        atomic_write_json(run_dir / "summary_metrics.json", metrics)
        atomic_write_json(run_dir / "component_checks.json", checks)
        full_run = len(cases) == 300
        stage_status = {
            "stage": "live_completed",
            "case_count": len(cases),
            "raw_response_count": len(raw_rows),
            "deterministic_result_count": len(results),
            "persistence_enabled": False,
            "cache_read_enabled": False,
            "cache_write_enabled": False,
            "full_run": full_run,
            "hard_gates": metrics["hard_gates"],
            "ready_for_full_judge": full_run and metrics["hard_gates_passed"],
        }
        atomic_write_json(run_dir / "live_stage_status.json", stage_status)
        manifest.update({"live_stage": "live_completed", "live_finished_at": _utc_now(), "live_result_count": len(results)})
        self._write_manifest(run_dir, manifest)
        return run_dir

    def run_judge(self, *, resume: bool = True) -> Path:
        run_dir, manifest = self._ensure_run(resume=resume)
        if manifest.get("live_stage") != "live_completed":
            raise RuntimeError("Judge stage requires a completed live stage")
        assert_resume_compatible(
            manifest,
            dataset_sha256=self.dataset_sha256,
            provider=self.config.judge_provider,
            model=self.config.judge_model,
            version=JUDGE_RUBRIC_VERSION,
            stage="judge",
        )
        if manifest.get("judge_stage") == "judge_completed":
            return run_dir
        results = read_jsonl(run_dir / "case_results.jsonl")
        selected = self._select_judge_cases(results)
        result_by_id = {str(row["case_id"]): row for row in results}
        if any(case["id"] not in result_by_id for case in selected):
            raise RuntimeError("Judge stage cannot find every selected live result")
        manifest["judge_stage"] = "judge_running"
        manifest["judge_started_at"] = manifest.get("judge_started_at") or _utc_now()
        self._write_manifest(run_dir, manifest)
        judge_path = run_dir / "judge_results.jsonl"
        done = completed_ids(judge_path)
        for case in selected:
            if case["id"] in done:
                continue
            row = judge_case(case, result_by_id[case["id"]], self.config)
            append_jsonl(judge_path, row)
            manifest["judge_completed_case_count"] = len(completed_ids(judge_path))
            self._write_manifest(run_dir, manifest)
        judge_rows = [row for row in read_jsonl(judge_path) if row.get("case_id") in {case["id"] for case in selected}]
        if len(judge_rows) != len(selected):
            raise RuntimeError("Judge checkpoint is incomplete after execution")
        summary = summarize_judge(judge_rows, judge_agrees_with_deterministic)
        _write_csv(run_dir / "judge_results.csv", judge_rows)
        atomic_write_json(run_dir / "judge_summary.json", summary)
        disagreements = [row for row in judge_rows if row.get("status") == "ok" and not judge_agrees_with_deterministic(row)]
        _write_csv(run_dir / "judge_disagreements.csv", disagreements)
        manifest.update({"judge_stage": "judge_completed", "judge_finished_at": _utc_now(), "judge_result_count": len(judge_rows)})
        self._write_manifest(run_dir, manifest)
        return run_dir

    def _select_judge_cases(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Choose a small judge sample across response routes before filling strata."""

        live_cases = self._selected_cases(self.config.question_limit)
        if self.config.case_ids or self.config.judge_limit >= len(live_cases):
            return live_cases[: self.config.judge_limit]
        result_by_id = {str(row["case_id"]): row for row in results}
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        for origin in ("llm_generated", "safe_fallback", "guardrail", "emergency_response"):
            case = next(
                (
                    item
                    for item in live_cases
                    if item["id"] not in selected_ids
                    and result_by_id.get(str(item["id"]), {}).get("actual_origin") == origin
                ),
                None,
            )
            if case is not None and len(selected) < self.config.judge_limit:
                selected.append(case)
                selected_ids.add(str(case["id"]))
        for case in live_cases:
            if len(selected) >= self.config.judge_limit:
                break
            if case["id"] not in selected_ids:
                selected.append(case)
                selected_ids.add(str(case["id"]))
        return selected

    def finalize(self, *, resume: bool = True) -> Path:
        run_dir, manifest = self._ensure_run(resume=resume)
        if manifest.get("live_stage") != "live_completed" or manifest.get("judge_stage") != "judge_completed":
            raise RuntimeError("Finalize requires completed live and judge stages")
        results = read_jsonl(run_dir / "case_results.jsonl")
        judge_rows = read_jsonl(run_dir / "judge_results.jsonl")
        if len(results) != 300 or len(judge_rows) != 300:
            raise RuntimeError("Finalization requires exactly 300 live results and 300 judge results")
        metrics = summarize_metrics(results)
        judge_summary = summarize_judge(judge_rows, judge_agrees_with_deterministic)
        summary = {"deterministic": metrics, "judge": judge_summary}
        atomic_write_json(run_dir / "summary_metrics.json", summary)
        _write_csv(run_dir / "failure_cases.csv", [row for row in results if row.get("failure_reasons")])
        _write_csv(run_dir / "judge_disagreements.csv", [row for row in judge_rows if row.get("status") == "ok" and not judge_agrees_with_deterministic(row)])
        plots = create_plots(run_dir, metrics, results, judge_rows)
        render_report(run_dir, manifest, metrics, judge_summary, results, plots)
        (run_dir / "FINALIZED").write_text("finalized\n", encoding="utf-8")
        manifest.update({"finalized": True, "finalized_at": _utc_now()})
        self._write_manifest(run_dir, manifest)
        return run_dir

    @staticmethod
    def _write_category_summary(run_dir: Path, results: list[dict[str, Any]]) -> None:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in results:
            grouped[str(row["category"])].append(row)
        rows = []
        for category, items in sorted(grouped.items()):
            rows.append(
                {
                    "category": category,
                    "case_count": len(items),
                    "behavior_match_rate": round(100 * sum(bool(item.get("behavior_match")) for item in items) / len(items), 2),
                    "deterministic_score_average": round(sum(float(item.get("deterministic_score") or 0) for item in items) / len(items), 2),
                }
            )
        _write_csv(run_dir / "category_summary.csv", rows)


__all__ = ["FinalEvaluationRunner"]
