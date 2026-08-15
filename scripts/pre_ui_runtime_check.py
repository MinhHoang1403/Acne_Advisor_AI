#!/usr/bin/env python3
"""Read-only backend readiness check used before starting the UI."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=False)

from scripts.inspect_phase2_readiness import inspect_readiness  # noqa: E402
from src.observability.versioning import get_answer_cache_version  # noqa: E402


def check(name: str, passed: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details or {}}


def frontend_config_status(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    frontend = root / "src" / "frontend"
    package = frontend / "package.json"
    api_source = frontend / "src" / "api" / "chatApi.js"
    return {
        "frontend_exists": frontend.is_dir(),
        "package_json": package.is_file(),
        "api_contract_source": api_source.is_file(),
        "vite_api_url": os.getenv("VITE_API_URL", "http://127.0.0.1:8000"),
    }


def environment_summary() -> dict[str, Any]:
    """Return non-secret endpoint/config identities only."""

    return {
        "qdrant_url": _redact_url(os.getenv("QDRANT_URL", "http://localhost:6333")),
        "neo4j_uri": _redact_url(os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")),
        "redis_url": _redact_url(os.getenv("REDIS_URL", "redis://localhost:6379/0")),
        "answer_cache_version": get_answer_cache_version(),
        "llm_provider": os.getenv("LLM_PROVIDER", "gemini"),
    }


def run_pre_ui_check() -> dict[str, Any]:
    readiness = inspect_readiness()
    frontend = frontend_config_status()
    checks = [
        check("backend_readiness", readiness["passed"], {"checks": readiness["checks"]}),
        check("cache_version", get_answer_cache_version() == "v8", {"answer_cache_version": get_answer_cache_version()}),
        check(
            "frontend_config",
            all(frontend[key] for key in ("frontend_exists", "package_json", "api_contract_source")),
            frontend,
        ),
    ]
    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "environment": environment_summary(),
    }


def _redact_url(value: str) -> str:
    if "://" not in value or "@" not in value:
        return value
    scheme, remainder = value.split("://", 1)
    return f"{scheme}://[REDACTED]@{remainder.rsplit('@', 1)[-1]}"


def main() -> int:
    report = run_pre_ui_check()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
