from __future__ import annotations

from pathlib import Path

from scripts import check_reproducible_environment as checker
from src.observability.versioning import (
    build_pipeline_version_manifest,
    compute_pipeline_fingerprint,
    get_answer_cache_version,
)


def test_python_version_file_is_pinned() -> None:
    assert (checker.PROJECT_ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.11.9"


def test_lock_file_is_exact_and_has_no_legacy_google_sdk() -> None:
    report = checker.inspect_lock_file(checker.PROJECT_ROOT / "requirements.lock.txt")

    assert report["checks"]["exact_pins"] is True
    assert report["checks"]["no_local_paths"] is True
    assert report["checks"]["no_legacy_google_sdk"] is True
    assert report["checks"]["google_genai_pinned"] is True
    assert report["details"]["exact_pins"]["package_count"] > 100


def test_compose_images_are_digest_pinned_without_latest() -> None:
    report = checker.inspect_compose_images(checker.PROJECT_ROOT / "docker-compose.yml")

    assert report["passed"] is True
    assert report["invalid"] == {}
    assert all("@sha256:" in image for image in report["images"].values())
    assert all(":latest" not in image for image in report["images"].values())


def test_env_example_has_reproducible_version_contract() -> None:
    report = checker.inspect_env_example(checker.PROJECT_ROOT / ".env.example")

    assert report["passed"] is True
    assert report["missing"] == {}
    assert report["duplicates"] == []
    assert report["malformed"] == []


def test_reproducible_environment_version_changes_fingerprint() -> None:
    old_manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "REPRODUCIBLE_ENVIRONMENT_VERSION": "reproducible_environment_v0",
        }
    )
    new_manifest = build_pipeline_version_manifest(
        {
            "CACHE_ANSWER_VERSION": "v5",
            "REPRODUCIBLE_ENVIRONMENT_VERSION": "reproducible_environment_v1",
        }
    )

    assert new_manifest["reproducible_environment_version"] == "reproducible_environment_v1"
    assert compute_pipeline_fingerprint(old_manifest) != compute_pipeline_fingerprint(new_manifest)
    assert get_answer_cache_version({"CACHE_ANSWER_VERSION": "v5"}) == "v6"


def test_checker_passes_without_network_or_pip_check() -> None:
    report = checker.check_reproducible_environment(run_pip_check=False)

    assert report["passed"] is True


def test_checker_reports_invalid_fixture(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("google-genai>=1\n", encoding="utf-8")
    legacy_package = "google" + "-" + "generativeai"
    (tmp_path / "requirements.lock.txt").write_text(
        "google-genai>=1\n"
        f"{legacy_package}==0.8.6\n"
        "local-pkg @ file:///C:/Users/example/pkg\n",
        encoding="utf-8",
    )
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n  qdrant:\n    image: qdrant/qdrant:latest\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.example").write_text("```dotenv\nCACHE_ANSWER_VERSION=v5\n", encoding="utf-8")
    (tmp_path / ".python-version").write_text("3.12.0\n", encoding="utf-8")

    report = checker.check_reproducible_environment(tmp_path, run_pip_check=False)

    assert report["passed"] is False
    failed = {item["name"] for item in report["checks"] if not item["passed"]}
    assert "lock_exact_pins" in failed
    assert "lock_no_local_paths" in failed
    assert "lock_no_legacy_google_sdk" in failed
    assert "compose_images_pinned" in failed
    assert "env_example_contract" in failed
    assert "python_version_file" in failed


def test_env_file_is_not_tracked_by_git() -> None:
    assert not checker._git_ls_files_env(checker.PROJECT_ROOT)
