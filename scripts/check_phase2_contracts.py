#!/usr/bin/env python3
"""Run the full offline/read-only runtime implementation contract suite."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKS = [
    ["scripts/inspect_phase2_readiness.py"],
    ["scripts/check_answer_contracts.py"],
    ["scripts/check_safe_fallback_flow.py"],
    ["scripts/smoke_phase2_runtime.py", "--mode", "offline"],
    ["scripts/inspect_cache_versions.py"],
]


def run_phase2_contracts(timeout_seconds: int = 120) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for command in CHECKS:
        started = time.perf_counter()
        full_command = [sys.executable, *command]
        try:
            completed = subprocess.run(
                full_command,
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
            stdout = completed.stdout.strip()
            stderr = completed.stderr.strip()
            checks.append(
                {
                    "name": " ".join(command),
                    "command": full_command,
                    "passed": completed.returncode == 0,
                    "returncode": completed.returncode,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "summary": _parse_json_or_excerpt(stdout),
                    "stderr": stderr[-1000:] if stderr else "",
                }
            )
        except subprocess.TimeoutExpired as exc:
            checks.append(
                {
                    "name": " ".join(command),
                    "command": full_command,
                    "passed": False,
                    "returncode": None,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "summary": {"error": f"Timed out after {timeout_seconds}s"},
                    "stderr": str(exc),
                }
            )

    passed_checks = sum(1 for check in checks if check["passed"])
    return {
        "passed": passed_checks == len(checks),
        "total_checks": len(checks),
        "passed_checks": passed_checks,
        "failed_checks": len(checks) - passed_checks,
        "checks": checks,
    }


def _parse_json_or_excerpt(text: str) -> Any:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"output_excerpt": text[-2000:]}


def main() -> int:
    summary = run_phase2_contracts()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
