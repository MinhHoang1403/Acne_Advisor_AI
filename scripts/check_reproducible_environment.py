"""Kiểm tra read-only về lock file, image pinning, env template và imports.

Script so sánh contract cấu hình chứ không cài package hoặc sửa môi trường. Các
pattern local path/secret ngăn artifact phụ thuộc máy cá nhân hay lộ credential.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_ENV_EXAMPLE_KEYS = {
    "GOOGLE_GENAI_SDK_VERSION": "google_genai_sdk_v1",
    "SAFE_FALLBACK_FLOW_VERSION": "safe_fallback_flow_v4",
    "CACHE_ANSWER_VERSION": "v10",
    "EVIDENCE_GROUNDING_VERSION": "evidence_grounded_runtime_v2",
    "REPRODUCIBLE_ENVIRONMENT_VERSION": "reproducible_environment_v1",
    "END_TO_END_RELEASE_READINESS_VERSION": "end_to_end_release_readiness_v1",
}

IMPORTANT_LOCK_PACKAGES = {
    "fastapi",
    "google-genai",
    "langgraph",
    "qdrant-client",
    "neo4j",
    "redis",
    "sqlalchemy",
    "pytest",
}

RUNTIME_IMPORTS = [
    "google.genai",
    "fastapi",
    "langgraph",
    "qdrant_client",
    "neo4j",
    "redis",
    "sqlalchemy",
    "src.agent.llm.provider",
    "src.database.vector_store",
    "src.api.app",
]

BAD_LOCK_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"file:///",
        r"\bC:\\",
        r"\bC:/Users/",
        r"\bUsers\\",
        r"site-packages",
        "AI" + "za",
        r"GOOGLE_API_KEY\s*=\s*\S+",
        r"LLAMA_CLOUD_API_KEY\s*=\s*\S+",
        r"api[_-]?key.{0,20}[A-Za-z0-9_-]{20,}",
    )
]


def check_reproducible_environment(root: Path = PROJECT_ROOT, *, run_pip_check: bool = True) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, details: dict[str, Any] | None = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details or {}})

    python_ok = sys.version_info[:2] == (3, 11)
    add("python_3_11", python_ok, {"version": sys.version.split()[0]})

    requirements = root / "requirements.txt"
    lock = root / "requirements.lock.txt"
    add("requirements_txt_exists", requirements.exists())
    add("requirements_lock_exists", lock.exists())

    lock_report = inspect_lock_file(lock)
    for key, value in lock_report["checks"].items():
        add(f"lock_{key}", bool(value), lock_report.get("details", {}).get(key, {}))

    compose_report = inspect_compose_images(root / "docker-compose.yml")
    add("compose_images_pinned", compose_report["passed"], compose_report)

    env_report = inspect_env_example(root / ".env.example")
    add("env_example_contract", env_report["passed"], env_report)

    python_version = (root / ".python-version").read_text(encoding="utf-8").strip() if (root / ".python-version").exists() else ""
    add("python_version_file", python_version == "3.11.9", {"python_version": python_version})

    add("env_not_tracked", not _git_ls_files_env(root), {})

    import_report = check_runtime_imports()
    add("runtime_imports", import_report["passed"], import_report)
    legacy_module = "google" + "." + "generativeai"
    add("legacy_google_sdk_not_importable", importlib.util.find_spec(legacy_module) is None)

    if run_pip_check:
        pip_report = run_pip_check_command(root)
        add("pip_check", pip_report["passed"], pip_report)

    passed = all(item["passed"] for item in checks)
    return {"passed": passed, "checks": checks}


def inspect_lock_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "checks": {
                "exact_pins": False,
                "no_local_paths": False,
                "no_legacy_google_sdk": False,
                "google_genai_pinned": False,
                "important_directs_present": False,
                "no_conflicting_duplicates": False,
            },
            "details": {},
        }

    text = path.read_text(encoding="utf-8")
    pins: dict[str, str] = {}
    conflicts: dict[str, set[str]] = {}
    invalid_specs: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("--"):
            continue
        if "==" not in line:
            invalid_specs.append(raw_line.strip())
            continue
        name, version = line.split("==", 1)
        package = _canonical_package_name(name)
        version = version.strip()
        if any(operator in version for operator in (">=", "<=", "~=", ">", "<")):
            invalid_specs.append(raw_line.strip())
        if package in pins and pins[package] != version:
            conflicts.setdefault(package, set()).update({pins[package], version})
        pins[package] = version

    bad_patterns = [pattern.pattern for pattern in BAD_LOCK_PATTERNS if pattern.search(text)]
    checks = {
        "exact_pins": not invalid_specs and bool(pins),
        "no_local_paths": not bad_patterns,
        "no_legacy_google_sdk": _legacy_google_package_name() not in pins
        and _legacy_google_package_name() not in text.lower(),
        "google_genai_pinned": "google-genai" in pins,
        "important_directs_present": IMPORTANT_LOCK_PACKAGES.issubset(set(pins)),
        "no_conflicting_duplicates": not conflicts,
    }
    return {
        "checks": checks,
        "details": {
            "exact_pins": {"invalid_specs": invalid_specs[:10], "package_count": len(pins)},
            "no_local_paths": {"bad_patterns": bad_patterns},
            "google_genai_pinned": {"version": pins.get("google-genai")},
            "important_directs_present": {
                "missing": sorted(IMPORTANT_LOCK_PACKAGES.difference(set(pins))),
            },
            "no_conflicting_duplicates": {
                "conflicts": {key: sorted(value) for key, value in conflicts.items()},
            },
        },
    }


def inspect_compose_images(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"passed": False, "images": {}, "error": "docker-compose.yml missing"}

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    images: dict[str, str] = {
        service: str(config.get("image", ""))
        for service, config in (data.get("services") or {}).items()
        if isinstance(config, dict)
    }
    invalid = {
        service: image
        for service, image in images.items()
        if "@sha256:" not in image or ":latest" in image
    }
    return {"passed": bool(images) and not invalid, "images": images, "invalid": invalid}


def inspect_env_example(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"passed": False, "missing": sorted(REQUIRED_ENV_EXAMPLE_KEYS), "duplicates": []}

    text = path.read_text(encoding="utf-8-sig")
    values: dict[str, str] = {}
    duplicates: list[str] = []
    malformed: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("```") or "=" not in line:
            malformed.append(raw_line)
            continue
        name, value = line.split("=", 1)
        if name.strip() != name or not name:
            malformed.append(raw_line)
            continue
        if name in values:
            duplicates.append(name)
        values[name] = value

    missing = {
        key: expected
        for key, expected in REQUIRED_ENV_EXAMPLE_KEYS.items()
        if values.get(key) != expected
    }
    secret_like = [
        name
        for name, value in values.items()
        if any(marker in name for marker in ("API_KEY", "PASSWORD"))
        and value
        and not value.startswith("dummy")
        and value not in {"password", "neo4j/password"}
    ]
    return {
        "passed": not missing and not duplicates and not malformed and not secret_like,
        "missing": missing,
        "duplicates": sorted(set(duplicates)),
        "malformed": malformed[:10],
        "secret_like": sorted(secret_like),
    }


def check_runtime_imports() -> dict[str, Any]:
    imported: list[str] = []
    failed: dict[str, str] = {}
    for module_name in RUNTIME_IMPORTS:
        try:
            importlib.import_module(module_name)
            imported.append(module_name)
        except Exception as exc:
            failed[module_name] = exc.__class__.__name__
    return {"passed": not failed, "imported": imported, "failed": failed}


def run_pip_check_command(root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _git_ls_files_env(root: Path) -> bool:
    completed = subprocess.run(
        ["git", "ls-files", ".env"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    return bool(completed.stdout.strip())


def _canonical_package_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _legacy_google_package_name() -> str:
    return "google" + "-" + "generativeai"


def main() -> int:
    report = check_reproducible_environment()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["passed"]:
        print("REPRODUCIBLE_ENVIRONMENT: PASS")
        return 0
    print("REPRODUCIBLE_ENVIRONMENT: FAIL", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
