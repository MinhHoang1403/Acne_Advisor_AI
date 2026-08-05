"""Durable JSONL checkpoints for the two independent Evaluation V3 stages."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL checkpoint {path} line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Checkpoint {path} line {line_number} is not an object")
        rows.append(value)
    return rows


def completed_ids(path: Path) -> set[str]:
    return {str(row["case_id"]) for row in read_jsonl(path) if row.get("case_id")}


def assert_resume_compatible(
    manifest: dict[str, Any],
    *,
    dataset_sha256: str,
    provider: str,
    model: str,
    version: str,
    stage: str,
) -> None:
    expected = {
        "dataset_sha256": dataset_sha256,
        f"{stage}_provider": provider,
        f"{stage}_model": model,
    }
    version_key = "metrics_version" if stage == "live" else "judge_rubric_version"
    expected[version_key] = version
    mismatches = [
        key
        for key, value in expected.items()
        if manifest.get(key) not in {None, value}
    ]
    if mismatches:
        raise ValueError(
            "Resume rejected because the saved run differs in: " + ", ".join(mismatches)
        )
