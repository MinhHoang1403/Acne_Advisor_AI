from __future__ import annotations

import asyncio
import os
from pathlib import Path
import subprocess
import sys

import pytest
from httpx import ASGITransport, AsyncClient

from scripts import check_release_readiness as checker
from src.api.app import app
from src.observability.versioning import (
    build_pipeline_version_manifest,
    compute_pipeline_fingerprint,
    get_answer_cache_version,
)


def test_release_readiness_version_changes_fingerprint_without_cache_bump() -> None:
    old_manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "END_TO_END_RELEASE_READINESS_VERSION": "end_to_end_release_readiness_v0",
        }
    )
    new_manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "END_TO_END_RELEASE_READINESS_VERSION": "end_to_end_release_readiness_v1",
        }
    )

    assert new_manifest["end_to_end_release_readiness_version"] == "end_to_end_release_readiness_v1"
    assert compute_pipeline_fingerprint(old_manifest) != compute_pipeline_fingerprint(new_manifest)
    assert get_answer_cache_version({"CACHE_ANSWER_VERSION": "v5"}) == "v7"


@pytest.mark.asyncio
async def test_release_checker_offline_passes_without_live_calls() -> None:
    report = await checker.run_offline(run_pip_check=False)

    assert report["status"] == "PASS"
    assert report["passed"] is True


def test_release_checker_invalid_env_fixture_fails(tmp_path: Path) -> None:
    env_example = tmp_path / ".env.example"
    env_example.write_text(
        "CACHE_ANSWER_VERSION=v4\n"
        "GOOGLE_GENAI_SDK_VERSION=google_genai_sdk_v1\n"
        "REPRODUCIBLE_ENVIRONMENT_VERSION=reproducible_environment_v1\n",
        encoding="utf-8",
    )

    report = checker.inspect_env_contract(env_example)

    assert report["passed"] is False
    assert "CACHE_ANSWER_VERSION" in report["missing"]
    assert "END_TO_END_RELEASE_READINESS_VERSION" in report["missing"]


def test_extract_data_counts_reads_current_readiness_schema() -> None:
    summary = {
        "checks": [
            {
                "name": "frozen_phase1_manifest",
                "details": {"counts": {"entities": 32}},
            },
            {
                "name": "qdrant_frozen_knowledge",
                "details": {"points_count": 512},
            },
            {
                "name": "neo4j_frozen_graph",
                "details": {"nodes": 32, "relationships": 27},
            },
        ]
    }

    counts = checker.extract_data_counts(summary)

    assert counts["passed"] is True
    assert counts["acne_knowledge"] == 512
    assert counts["acne_entities"] == 32
    assert counts["neo4j_nodes"] == 32
    assert counts["neo4j_relationships"] == 27


def test_release_checker_utf8_stdio_survives_cp1252_console() -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"
    code = (
        "from scripts.check_release_readiness import _configure_utf8_stdio; "
        "_configure_utf8_stdio(); "
        "print('mụn viêm đỏ, đau và sưng')"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        check=False,
    )

    stderr = completed.stderr.decode("utf-8", errors="replace")
    stdout = completed.stdout.decode("utf-8", errors="replace")
    assert completed.returncode == 0
    assert "UnicodeEncodeError" not in stderr
    assert "mụn viêm đỏ" in stdout


@pytest.mark.asyncio
async def test_http_structured_503_504_and_safe_fallback(monkeypatch) -> None:
    monkeypatch.setenv("RELEASE_READINESS_TEST_MODE", "http_double")
    monkeypatch.setenv("RELEASE_READINESS_HEALTH_DOUBLE", "true")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://release.test") as client:
        err503 = await client.post("/chat", json={"message": "__release_readiness_503__"})
        err504 = await client.post("/chat", json={"message": "__release_readiness_504__"})
        fallback = await client.post("/chat", json={"message": "__release_readiness_safe_fallback__"})
        unicode_response = await client.post("/chat", json={"message": "Mụn viêm đỏ, đau và sưng nên xử lý thế nào?"})
        invalid = await client.post("/chat", json={"message": "   "})

    assert checker.structured_error_ok(err503, 503, "provider_unavailable")
    assert checker.structured_error_ok(err504, 504, "agent_timeout")
    assert checker.safe_fallback_ok(fallback)
    assert unicode_response.status_code == 200
    assert "tiếng Việt" in unicode_response.json()["answer"]
    assert invalid.status_code == 400
    assert invalid.json()["detail"]["code"] == "empty_message"


@pytest.mark.asyncio
async def test_api_subprocess_cleanup_and_contract() -> None:
    async with checker.api_server(test_mode=True, health_double=True) as base_url:
        checks = await checker.http_boundary_checks(base_url)

    failed = [item["name"] for item in checks if not item["passed"]]
    assert failed == []

    await asyncio.sleep(0.1)


def test_frontend_contract_reads_structured_errors() -> None:
    report = checker.inspect_frontend_contract()

    assert report["passed"] is True
    assert report["package_manager"] == "npm"
    assert report["structured_error_parser"] is True
    assert report["no_sensitive_markers"] is True


def test_tracked_secret_scan_is_clean() -> None:
    report = checker.tracked_secret_scan()

    assert report["passed"] is True
