#!/usr/bin/env python3
"""End-to-end release readiness checks with offline, local-services, and live modes."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

EXPECTED_VERSION = "end_to_end_release_readiness_v1"
EXPECTED_CACHE_VERSION = "v5"
EXPECTED_FINGERPRINT_BEFORE_STEP10 = "c8507401e35043380fd119e7"
EXPECTED_COUNTS = {
    "acne_knowledge": 641,
    "acne_entities_v1": {20, 22, 32},
    "neo4j_nodes": {21, 23, 32},
    "neo4j_relationships": {15, 18, 27},
}
SECRET_PATTERNS = ("AI" + "za",)


def _configure_utf8_stdio() -> None:
    """Best-effort UTF-8 output for Windows consoles and CI runners."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            continue


def check(name: str, passed: bool, details: dict[str, Any] | None = None, severity: str = "error") -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "severity": severity,
        "details": sanitize(details or {}),
    }


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            key_lower = str(key).lower()
            if any(marker in key_lower for marker in ("api_key", "password", "secret", "token", "authorization")):
                sanitized[key] = "<REDACTED>" if item else "<EMPTY>"
            else:
                sanitized[key] = sanitize(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        result = value
        for marker in SECRET_PATTERNS:
            if marker in result:
                result = result.replace(marker, "<REDACTED>")
        return result
    return value


def finalize(mode: str, checks: list[dict[str, Any]], blocked: bool = False) -> dict[str, Any]:
    failed = [item for item in checks if not item["passed"] and item.get("severity") == "error"]
    status = "BLOCKED" if blocked else ("FAIL" if failed else "PASS")
    return {
        "name": "END_TO_END_RELEASE_READINESS",
        "mode": mode,
        "status": status,
        "passed": status == "PASS",
        "total_checks": len(checks),
        "passed_checks": sum(1 for item in checks if item["passed"]),
        "failed_checks": len(failed),
        "checks": checks,
    }


def run_command(command: list[str], *, timeout: float = 60.0, cwd: Path = PROJECT_ROOT) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {"passed": False, "returncode": None, "stdout": "", "stderr": f"timeout after {exc.timeout}s"}
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


async def run_offline(*, run_pip_check: bool = True) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    from scripts import check_reproducible_environment as reproducible
    from src.observability.versioning import (
        build_pipeline_version_manifest,
        compute_pipeline_fingerprint,
        get_answer_cache_version,
    )

    checks.append(check("python_3_11", sys.version_info[:2] == (3, 11), {"version": sys.version.split()[0]}))
    if run_pip_check:
        pip_report = run_command([sys.executable, "-m", "pip", "check"], timeout=90)
        checks.append(check("pip_check", pip_report["passed"], pip_report))

    repro_report = reproducible.check_reproducible_environment(run_pip_check=run_pip_check)
    checks.append(check("reproducible_environment", repro_report["passed"], {"failed": failed_names(repro_report)}))

    compose_report = reproducible.inspect_compose_images(PROJECT_ROOT / "docker-compose.yml")
    checks.append(check("compose_pinned_images", compose_report["passed"], compose_report))

    env_report = inspect_env_contract(PROJECT_ROOT / ".env.example")
    checks.append(check("env_contract", env_report["passed"], env_report))

    manifest = build_pipeline_version_manifest()
    fingerprint = compute_pipeline_fingerprint(manifest)
    checks.append(
        check(
            "release_readiness_version",
            manifest.get("end_to_end_release_readiness_version") == EXPECTED_VERSION,
            {"end_to_end_release_readiness_version": manifest.get("end_to_end_release_readiness_version")},
        )
    )
    checks.append(
        check(
            "pipeline_fingerprint_changed",
            bool(fingerprint) and fingerprint != EXPECTED_FINGERPRINT_BEFORE_STEP10,
            {"fingerprint": fingerprint, "previous": EXPECTED_FINGERPRINT_BEFORE_STEP10},
        )
    )
    checks.append(check("cache_version_v5", get_answer_cache_version() == EXPECTED_CACHE_VERSION, {"cache_version": get_answer_cache_version()}))

    api_report = await offline_api_contract_checks()
    checks.extend(api_report)

    frontend_report = inspect_frontend_contract()
    checks.append(check("frontend_contract", frontend_report["passed"], frontend_report))

    secret_report = tracked_secret_scan()
    checks.append(check("tracked_secret_scan", secret_report["passed"], secret_report))

    return finalize("offline", checks)


async def run_local_services() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    compose_ps = run_command(["docker", "compose", "ps"], timeout=30)
    checks.append(check("docker_compose_ps", compose_ps["passed"], {"stdout": compose_ps["stdout"]}))

    readiness = run_json_script(["scripts/inspect_phase2_readiness.py"], timeout=120)
    checks.append(check("phase2_readiness", readiness["passed"], readiness["summary"]))
    counts = extract_data_counts(readiness["summary"])
    checks.append(check("data_counts", counts["passed"], counts))

    ollama = await check_ollama()
    checks.append(check("ollama_local_model", ollama["passed"], ollama))

    async with api_server(test_mode=True, health_double=False) as base_url:
        http_checks = await http_boundary_checks(base_url)
        checks.extend(http_checks)

    return finalize("local-services", checks)


async def run_live() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    blocked = False

    gemini = await check_gemini_live()
    checks.append(check("gemini_live_generation", gemini["passed"], gemini))
    if gemini.get("blocked"):
        blocked = True

    ollama = await check_ollama(generate=True)
    checks.append(check("ollama_live_generation", ollama["passed"], ollama))
    if ollama.get("blocked"):
        blocked = True

    async with api_server(test_mode=True, health_double=False) as base_url:
        checks.extend(await http_boundary_checks(base_url))
        fallback = await http_chat(base_url, "__release_readiness_fallback__")
        fallback_ok = (
            fallback["status_code"] == 200
            and fallback["json"].get("metadata", {}).get("fallback_used") is True
            and fallback["json"].get("metadata", {}).get("fallback_provider") == "ollama"
        )
        checks.append(check("http_gemini_to_ollama_fallback", fallback_ok, fallback))

    return finalize("live", checks, blocked=blocked)


def inspect_env_contract(path: Path) -> dict[str, Any]:
    required = {
        "CACHE_ANSWER_VERSION": EXPECTED_CACHE_VERSION,
        "GOOGLE_GENAI_SDK_VERSION": "google_genai_sdk_v1",
        "REPRODUCIBLE_ENVIRONMENT_VERSION": "reproducible_environment_v1",
        "END_TO_END_RELEASE_READINESS_VERSION": EXPECTED_VERSION,
    }
    values: dict[str, str] = {}
    duplicates: list[str] = []
    malformed: list[str] = []
    if not path.exists():
        return {"passed": False, "missing": required, "duplicates": [], "malformed": ["missing .env.example"]}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            malformed.append(f"{line_number}:{stripped[:30]}")
            continue
        key, value = stripped.split("=", 1)
        if key in values:
            duplicates.append(key)
        values[key] = value
    missing = {key: expected for key, expected in required.items() if values.get(key) != expected}
    return {"passed": not missing and not duplicates and not malformed, "missing": missing, "duplicates": duplicates, "malformed": malformed}


async def offline_api_contract_checks() -> list[dict[str, Any]]:
    from httpx import ASGITransport, AsyncClient

    from src.api.app import app

    checks: list[dict[str, Any]] = []
    old_mode = os.environ.get("RELEASE_READINESS_TEST_MODE")
    old_health = os.environ.get("RELEASE_READINESS_HEALTH_DOUBLE")
    os.environ["RELEASE_READINESS_TEST_MODE"] = "http_double"
    os.environ["RELEASE_READINESS_HEALTH_DOUBLE"] = "true"
    try:
        schema = app.openapi()
        checks.append(check("openapi_contract", "/chat" in schema.get("paths", {}) and "/health" in schema.get("paths", {}), {"path_count": len(schema.get("paths", {}))}))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://release.test", timeout=10) as client:
            health = await client.get("/health")
            checks.append(check("http_health_double", health.status_code == 200 and health.json().get("status") == "ok", response_summary(health)))
            checks.append(check("http_chat_unicode", (await client.post("/chat", json={"message": "Mụn viêm đỏ, đau và sưng nên xử lý thế nào?", "bypass_cache": True})).status_code == 200, {}))
            checks.append(check("http_503_structured", structured_error_ok(await client.post("/chat", json={"message": "__release_readiness_503__"}), 503, "provider_unavailable"), {}))
            checks.append(check("http_504_structured", structured_error_ok(await client.post("/chat", json={"message": "__release_readiness_504__"}), 504, "agent_timeout"), {}))
            checks.append(check("http_safe_fallback_not_cacheable", safe_fallback_ok(await client.post("/chat", json={"message": "__release_readiness_safe_fallback__"})), {}))
            invalid = await client.post("/chat", json={"message": "   "})
            checks.append(check("http_invalid_request", invalid.status_code == 400 and invalid.json().get("detail", {}).get("code") == "empty_message", response_summary(invalid)))
    finally:
        restore_env("RELEASE_READINESS_TEST_MODE", old_mode)
        restore_env("RELEASE_READINESS_HEALTH_DOUBLE", old_health)
    return checks


def inspect_frontend_contract() -> dict[str, Any]:
    api_client = PROJECT_ROOT / "src" / "frontend" / "src" / "api" / "chatApi.js"
    api_config = PROJECT_ROOT / "src" / "frontend" / "src" / "config" / "api.js"
    package_json = PROJECT_ROOT / "src" / "frontend" / "package.json"
    lock_file = PROJECT_ROOT / "src" / "frontend" / "package-lock.json"
    source = api_client.read_text(encoding="utf-8") if api_client.exists() else ""
    config_source = api_config.read_text(encoding="utf-8") if api_config.exists() else ""
    combined_source = f"{source}\n{config_source}"
    package = json.loads(package_json.read_text(encoding="utf-8")) if package_json.exists() else {}
    scripts = package.get("scripts", {})
    details = {
        "package_manager": "npm" if lock_file.exists() else "unknown",
        "lock_file": lock_file.exists(),
        "build_script": "build" in scripts,
        "lint_script": "lint" in scripts,
        "uses_vite_api_url": "VITE_API_URL" in combined_source,
        "chat_endpoint": "/chat" in source,
        "health_endpoint": "/health" in source,
        "structured_error_parser": "parseApiError" in source and "error.status" in source and "error.code" in source,
        "no_sensitive_markers": not any(marker in source for marker in SECRET_PATTERNS),
    }
    details["passed"] = all(details.values()) if all(isinstance(v, bool) for v in details.values()) else (
        details["package_manager"] == "npm"
        and details["lock_file"]
        and details["build_script"]
        and details["uses_vite_api_url"]
        and details["chat_endpoint"]
        and details["health_endpoint"]
        and details["structured_error_parser"]
        and details["no_sensitive_markers"]
    )
    return details


def tracked_secret_scan() -> dict[str, Any]:
    patterns = [
        "AI" + "za",
        "GOOGLE" + "_API_KEY=",
        "LLAMA" + "_CLOUD_API_KEY=",
        "POSTGRES" + "_PASSWORD=",
        "NEO4J" + "_PASSWORD=",
    ]
    findings: list[str] = []
    tracked = run_command(["git", "ls-files"], timeout=30)
    for rel_path in tracked["stdout"].splitlines():
        path = PROJECT_ROOT / rel_path
        if not path.is_file() or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".ico", ".pdf"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in patterns:
            if pattern in text and rel_path not in {".env.example"}:
                findings.append(f"{rel_path}:{pattern}")
    return {"passed": not findings, "findings": findings[:20]}


def run_json_script(args: list[str], *, timeout: float = 120.0) -> dict[str, Any]:
    command = [sys.executable, *args]
    result = run_command(command, timeout=timeout)
    summary: dict[str, Any] = {}
    if result["stdout"]:
        try:
            summary = json.loads(result["stdout"])
        except json.JSONDecodeError:
            summary = {"stdout_excerpt": result["stdout"][:1000]}
    return {"passed": result["passed"] and bool(summary.get("passed", True)), "summary": summary, "returncode": result["returncode"]}


def extract_data_counts(summary: dict[str, Any]) -> dict[str, Any]:
    counts = {
        "acne_knowledge": None,
        "acne_entities_v1": None,
        "neo4j_nodes": None,
        "neo4j_relationships": None,
    }
    for item in summary.get("phase1_state_checks", []):
        name = item.get("name")
        details = item.get("details", {})
        if name == "qdrant_chunk_schema_and_points":
            counts["acne_knowledge"] = details.get("points_count")
        elif name == "qdrant_entity_schema_and_points":
            counts["acne_entities_v1"] = details.get("points_count")
        elif name == "neo4j_deterministic_graph":
            counts["neo4j_nodes"] = details.get("nodes")
            counts["neo4j_relationships"] = details.get("relationships")
    counts["passed"] = all(
        counts[key] in expected if isinstance(expected, set) else counts[key] == expected
        for key, expected in EXPECTED_COUNTS.items()
    )
    counts["expected"] = {
        key: sorted(value) if isinstance(value, set) else value
        for key, value in EXPECTED_COUNTS.items()
    }
    return counts


async def check_ollama(*, generate: bool = False) -> dict[str, Any]:
    from src.agent.llm.ollama_client import OLLAMA_BASE_URL, list_ollama_models

    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    started = time.perf_counter()
    try:
        models = await list_ollama_models(timeout_seconds=5)
        if model not in models:
            return {"passed": False, "blocked": True, "model": model, "available": False, "latency_ms": elapsed_ms(started)}
        details: dict[str, Any] = {"passed": True, "model": model, "available": True, "latency_ms": elapsed_ms(started)}
        if generate:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Trả lời đúng một từ tiếng Việt: sẵn sàng"}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 16},
            }
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            response.raise_for_status()
            text = response.json().get("message", {}).get("content", "")
            details.update({"passed": bool(text.strip()), "response_non_empty": bool(text.strip()), "latency_ms": elapsed_ms(started)})
        return details
    except Exception as exc:
        return {"passed": False, "blocked": True, "model": model, "error": safe_error(exc), "latency_ms": elapsed_ms(started)}


async def check_gemini_live() -> dict[str, Any]:
    key_configured = bool(os.getenv("GOOGLE_API_KEY"))
    model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    if not key_configured:
        return {"passed": False, "blocked": True, "model": model, "key_configured": False, "request_count": 0}

    from src.integrations.google_genai import generate_text_async

    started = time.perf_counter()
    try:
        text = await generate_text_async(
            prompt="Trả lời đúng một từ tiếng Việt: sẵn sàng",
            system_prompt="Bạn trả lời cực ngắn.",
            model_name=model,
            temperature=0.0,
            request_timeout=30,
        )
        return {
            "passed": bool(text.strip()),
            "blocked": False,
            "model": model,
            "key_configured": True,
            "request_count": 1,
            "response_non_empty": bool(text.strip()),
            "latency_ms": elapsed_ms(started),
        }
    except Exception as exc:
        return {
            "passed": False,
            "blocked": True,
            "model": model,
            "key_configured": True,
            "request_count": 1,
            "error": safe_error(exc),
            "latency_ms": elapsed_ms(started),
        }


@asynccontextmanager
async def api_server(*, test_mode: bool, health_double: bool):
    port = find_free_port()
    env = os.environ.copy()
    if test_mode:
        env["RELEASE_READINESS_TEST_MODE"] = "http_double"
    if health_double:
        env["RELEASE_READINESS_HEALTH_DOUBLE"] = "true"
    env.setdefault("PHASE2_DEBUG_METADATA", "false")
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        await wait_for_health(base_url)
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


async def wait_for_health(base_url: str, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: str | None = None
    async with httpx.AsyncClient(timeout=5) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return
                last_error = f"status={response.status_code}"
            except Exception as exc:
                last_error = safe_error(exc)
            await asyncio.sleep(0.5)
    raise TimeoutError(f"API did not become healthy: {last_error}")


async def http_boundary_checks(base_url: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    health = await http_get(base_url, "/health")
    checks.append(check("http_health", health["status_code"] == 200 and health["json"].get("status") == "ok", health))
    openapi = await http_get(base_url, "/openapi.json")
    paths = openapi["json"].get("paths", {}) if isinstance(openapi.get("json"), dict) else {}
    checks.append(check("http_openapi", openapi["status_code"] == 200 and "/chat" in paths, {"path_count": len(paths)}))
    chat = await http_chat(base_url, "Benzoyl peroxide có phải kháng sinh không?")
    answer = chat["json"].get("answer", "")
    checks.append(check("http_chat_unicode", chat["status_code"] == 200 and "Benzoyl peroxide" in answer and "kháng sinh" in answer, chat))
    invalid = await http_chat(base_url, "   ")
    checks.append(check("http_invalid_request", invalid["status_code"] == 400, invalid))
    err503 = await http_chat(base_url, "__release_readiness_503__")
    checks.append(check("http_503_structured", structured_error_payload_ok(err503, 503, "provider_unavailable"), err503))
    err504 = await http_chat(base_url, "__release_readiness_504__")
    checks.append(check("http_504_structured", structured_error_payload_ok(err504, 504, "agent_timeout"), err504))
    safe = await http_chat(base_url, "__release_readiness_safe_fallback__")
    checks.append(check("http_safe_fallback_not_cacheable", safe_fallback_payload_ok(safe), safe))
    return checks


async def http_get(base_url: str, path: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(f"{base_url}{path}")
    return response_summary(response)


async def http_chat(base_url: str, message: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(
            f"{base_url}/chat",
            json={"message": message, "bypass_cache": True, "session_id": f"release-{abs(hash(message)) % 100000}"},
        )
    return response_summary(response)


def response_summary(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except Exception:
        payload = {}
    return {
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type"),
        "json": sanitize(payload),
    }


def structured_error_ok(response: httpx.Response, status_code: int, code: str) -> bool:
    return structured_error_payload_ok(response_summary(response), status_code, code)


def structured_error_payload_ok(summary: dict[str, Any], status_code: int, code: str) -> bool:
    detail = summary.get("json", {}).get("detail", {})
    return (
        summary.get("status_code") == status_code
        and detail.get("code") == code
        and isinstance(detail.get("message"), str)
        and detail.get("retryable") is True
        and "traceback" not in json.dumps(detail, ensure_ascii=False).lower()
    )


def safe_fallback_ok(response: httpx.Response) -> bool:
    return safe_fallback_payload_ok(response_summary(response))


def safe_fallback_payload_ok(summary: dict[str, Any]) -> bool:
    metadata = summary.get("json", {}).get("metadata", {})
    cache = metadata.get("cache", {})
    return (
        summary.get("status_code") == 200
        and metadata.get("fallback_applied") is True
        and metadata.get("fallback_cache_eligible") is False
        and cache.get("hit") is False
    )


def failed_names(report: dict[str, Any]) -> list[str]:
    return [item["name"] for item in report.get("checks", []) if not item.get("passed")]


def restore_env(name: str, old_value: str | None) -> None:
    if old_value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old_value


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def safe_error(exc: Exception) -> str:
    return sanitize(f"{exc.__class__.__name__}: {exc}")[:500]


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="Check end-to-end release readiness.")
    parser.add_argument("--mode", choices=["offline", "local-services", "live"], default="offline")
    parser.add_argument("--skip-pip-check", action="store_true", help="Skip pip check for fast unit tests.")
    args = parser.parse_args()

    if args.mode == "offline":
        report = await run_offline(run_pip_check=not args.skip_pip_check)
    elif args.mode == "local-services":
        report = await run_local_services()
    else:
        report = await run_live()

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"END_TO_END_RELEASE_READINESS {args.mode}: {report['status']}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    _configure_utf8_stdio()
    raise SystemExit(asyncio.run(main_async()))
