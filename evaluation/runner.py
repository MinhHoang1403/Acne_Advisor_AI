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

from .checkpoint import (
    append_jsonl,
    assert_resume_compatible,
    atomic_write_json,
    completed_ids,
    read_jsonl,
    rows_by_case_id,
)
from .deterministic import deterministic_result, judge_agrees_with_deterministic, summarize_metrics
from .judge_gemini import judge_case, summarize_judge
from .live_eval import assert_isolated, component_checks, run_live_case
from .models import (
    CHECKPOINT_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION,
    JUDGE_RUBRIC_VERSION,
    METRICS_VERSION,
    EvaluationConfig,
)
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
        if self.config.run_dir is not None:
            path = self._validated_explicit_run_dir()
            if path.exists():
                raise FileExistsError(f"Evaluation run directory already exists: {path}")
            path.mkdir(parents=True, exist_ok=False)
            return path
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_final_evaluation_v3"
        path = self.config.report_root / run_id
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _validated_explicit_run_dir(self) -> Path:
        if self.config.run_dir is None:
            raise ValueError("No explicit evaluation run directory was provided")
        report_root = self.config.report_root.resolve()
        path = self.config.run_dir.resolve()
        try:
            path.relative_to(report_root)
        except ValueError as exc:
            raise ValueError(f"Evaluation run directory must be inside report root: {report_root}") from exc
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
        if self.config.run_dir is not None:
            path = self._validated_explicit_run_dir()
            if resume:
                if not path.is_dir() or not (path / "evaluation_manifest.json").exists():
                    raise FileNotFoundError(f"Explicit evaluation run directory does not exist: {path}")
                return path
            return self._new_run_dir()
        return self._latest_run_dir() if resume else self._new_run_dir()

    def _manifest(self, run_dir: Path) -> dict[str, Any]:
        path = run_dir / "evaluation_manifest.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _write_manifest(self, run_dir: Path, manifest: dict[str, Any]) -> None:
        manifest["updated_at"] = _utc_now()
        atomic_write_json(run_dir / "evaluation_manifest.json", manifest)

    def _resume_identity(self) -> dict[str, Any]:
        """Fingerprint semantic run inputs, never one invocation's local limits."""

        return {
            **self.config.resume_identity_json(),
            "dataset_sha256": self.dataset_sha256,
        }

    def _record_invocation(self, manifest: dict[str, Any], stage: str) -> None:
        manifest.setdefault("invocations", []).append(
            {
                "stage": stage,
                "requested_at": _utc_now(),
                "config": self.config.as_json(),
            }
        )

    def _ensure_run(self, *, resume: bool) -> tuple[Path, dict[str, Any]]:
        run_dir = self._run_dir(resume=resume)
        manifest = self._manifest(run_dir)
        if manifest:
            saved_schema = manifest.get("checkpoint_schema_version")
            if saved_schema != CHECKPOINT_SCHEMA_VERSION:
                raise ValueError(
                    "Resume rejected because the saved checkpoint schema is incompatible: "
                    f"{saved_schema!r}; expected {CHECKPOINT_SCHEMA_VERSION!r}."
                )
            current_identity = self._resume_identity()
            if manifest.get("resume_fingerprint") != _json_hash(current_identity):
                raise ValueError("Resume rejected because the saved semantic evaluation identity differs.")
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
        resume_identity = self._resume_identity()
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
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "config_sha256": _json_hash(config_data),
            "resume_identity": resume_identity,
            "resume_fingerprint": _json_hash(resume_identity),
            "invocations": [],
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
        self._record_invocation(manifest, "live")
        self._write_manifest(run_dir, manifest)
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

        results = read_jsonl(run_dir / "case_results.jsonl")
        canonical_cases = self._canonical_judge_cases(results)
        target_total = len(canonical_cases)
        if self.config.judge_limit <= 0:
            raise ValueError("Judge limit must be positive")
        invocation_cases = canonical_cases[: min(self.config.judge_limit, target_total)]
        result_by_id = {str(row["case_id"]): row for row in results}
        judge_path = run_dir / "judge_results.jsonl"
        indexed_rows = rows_by_case_id(judge_path)
        checkpoint = self._judge_checkpoint(manifest, canonical_cases)

        if manifest.get("judge_stage") == "judge_completed":
            self._assert_completed_judge_state(indexed_rows, checkpoint, canonical_cases)
            self._record_invocation(manifest, "judge")
            self._write_manifest(run_dir, manifest)
            return run_dir

        manifest["judge_stage"] = "judge_in_progress"
        manifest["judge_started_at"] = manifest.get("judge_started_at") or _utc_now()
        self._record_invocation(manifest, "judge")
        self._write_manifest(run_dir, manifest)

        for case in invocation_cases:
            case_id = str(case["id"])
            previous = indexed_rows.get(case_id)
            if previous and previous.get("status") == "ok":
                continue
            if previous and previous.get("checkpoint_status") == "final_error":
                continue

            checkpoint["cases"][case_id] = {"status": "in_progress", "updated_at": _utc_now()}
            self._write_manifest(run_dir, manifest)
            row = judge_case(case, result_by_id[case_id], self.config)
            checkpoint_status = str(row.get("checkpoint_status") or "final_error")
            if row.get("status") == "ok":
                row["checkpoint_status"] = "success"
                append_jsonl(judge_path, row)
                indexed_rows[case_id] = row
                checkpoint["cases"][case_id] = {"status": "success", "updated_at": _utc_now()}
            elif checkpoint_status == "transient_error":
                checkpoint["cases"][case_id] = {
                    "status": "transient_error",
                    "error": row.get("final_error"),
                    "updated_at": _utc_now(),
                }
            else:
                row["checkpoint_status"] = "final_error"
                append_jsonl(judge_path, row)
                indexed_rows[case_id] = row
                checkpoint["cases"][case_id] = {
                    "status": "final_error",
                    "error": row.get("final_error"),
                    "updated_at": _utc_now(),
                }
            manifest["judge_completed_case_count"] = len(
                {case_id for case_id, row in indexed_rows.items() if row.get("status") == "ok"}
            )
            self._write_manifest(run_dir, manifest)

        self._write_judge_outputs(run_dir, manifest, canonical_cases, indexed_rows)
        return run_dir

    def _canonical_judge_cases(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return the fixed full target set; judge_limit never changes this set."""

        canonical_cases = self._selected_cases(self.config.question_limit)
        result_by_id = {str(row.get("case_id")): row for row in results}
        expected_ids = [str(case["id"]) for case in canonical_cases]
        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError("Canonical judge target contains duplicate case IDs")
        missing = [case_id for case_id in expected_ids if case_id not in result_by_id]
        if missing:
            raise RuntimeError("Judge stage cannot find every selected live result: " + ", ".join(missing))
        return canonical_cases

    def _judge_checkpoint(self, manifest: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
        checkpoint = manifest.setdefault(
            "judge_checkpoint",
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "target_total_cases": len(cases),
                "cases": {},
            },
        )
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Judge checkpoint schema is incompatible")
        if checkpoint.get("target_total_cases") != len(cases):
            raise ValueError("Judge checkpoint target set differs from the current canonical run")
        states = checkpoint.setdefault("cases", {})
        for case in cases:
            states.setdefault(str(case["id"]), {"status": "pending"})
        return checkpoint

    def _write_judge_outputs(
        self,
        run_dir: Path,
        manifest: dict[str, Any],
        canonical_cases: list[dict[str, Any]],
        indexed_rows: dict[str, dict[str, Any]],
    ) -> None:
        target_ids = {str(case["id"]) for case in canonical_cases}
        rows = [row for case_id, row in indexed_rows.items() if case_id in target_ids]
        success_ids = {case_id for case_id, row in indexed_rows.items() if case_id in target_ids and row.get("status") == "ok"}
        final_errors = [row for row in rows if row.get("checkpoint_status") == "final_error"]
        checkpoint_states = manifest["judge_checkpoint"]["cases"]
        transient_errors = [
            case_id for case_id in target_ids if checkpoint_states.get(case_id, {}).get("status") == "transient_error"
        ]
        complete = len(success_ids) == len(target_ids) and not final_errors and not transient_errors

        summary = summarize_judge(rows, judge_agrees_with_deterministic)
        summary.update(
            {
                "target_total_cases": len(target_ids),
                "successful_unique_cases": len(success_ids),
                "remaining_case_count": len(target_ids) - len(success_ids),
                "final_error_count": len(final_errors),
                "transient_error_count": len(transient_errors),
                "is_complete": complete,
            }
        )
        _write_csv(run_dir / "judge_results.csv", rows)
        atomic_write_json(run_dir / "judge_summary.json", summary)
        disagreements = [row for row in rows if row.get("status") == "ok" and not judge_agrees_with_deterministic(row)]
        _write_csv(run_dir / "judge_disagreements.csv", disagreements)

        manifest.update(
            {
                "judge_target_total_cases": len(target_ids),
                "judge_successful_unique_cases": len(success_ids),
                "judge_result_count": len(rows),
                "judge_final_error_count": len(final_errors),
            }
        )
        if complete:
            manifest.update({"judge_stage": "judge_completed", "judge_finished_at": _utc_now()})
        elif final_errors:
            manifest["judge_stage"] = "judge_blocked"
        else:
            manifest["judge_stage"] = "judge_in_progress"
        self._write_manifest(run_dir, manifest)

    @staticmethod
    def _assert_completed_judge_state(
        indexed_rows: dict[str, dict[str, Any]], checkpoint: dict[str, Any], canonical_cases: list[dict[str, Any]]
    ) -> None:
        target_ids = {str(case["id"]) for case in canonical_cases}
        successful_ids = {
            case_id for case_id, row in indexed_rows.items() if row.get("status") == "ok" and case_id in target_ids
        }
        if successful_ids != target_ids or checkpoint.get("target_total_cases") != len(target_ids):
            raise RuntimeError("Completed judge stage does not contain the full successful canonical target set")

    def finalize(self, *, resume: bool = True) -> Path:
        run_dir, manifest = self._ensure_run(resume=resume)
        self._record_invocation(manifest, "finalize")
        self._write_manifest(run_dir, manifest)
        if manifest.get("live_stage") != "live_completed" or manifest.get("judge_stage") != "judge_completed":
            raise RuntimeError("Finalize requires completed live and judge stages")
        results = read_jsonl(run_dir / "case_results.jsonl")
        judge_rows_by_id = rows_by_case_id(run_dir / "judge_results.jsonl")
        judge_rows = list(judge_rows_by_id.values())
        canonical_ids = {str(case["id"]) for case in self._selected_cases(300)}
        result_ids = {str(row.get("case_id")) for row in results}
        target_total = int(manifest.get("judge_target_total_cases") or 0)
        final_errors = [row for row in judge_rows if row.get("checkpoint_status") == "final_error"]
        if (
            len(results) != 300
            or len(judge_rows) != 300
            or result_ids != canonical_ids
            or set(judge_rows_by_id) != canonical_ids
            or target_total != 300
            or final_errors
        ):
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
        manifest.update({"finalized": True, "finalized_at": _utc_now(), "run_status": "completed"})
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
