"""Run a local self-hosted Langfuse smoke and privacy inspection.

The script creates ephemeral bootstrap credentials and an isolated Compose
project, removes only that project's temporary volumes, and never prints
credentials or synthetic prohibited-content sentinels.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from httpx import ASGITransport, AsyncClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.observability.yml"
BASE_URL = "http://127.0.0.1:3001"
REQUEST_ID = "11111111-2222-4333-8444-555555555555"
PROJECT_NAME = f"acne-advisor-observability-smoke-{os.getpid()}"


def _bootstrap_environment() -> dict[str, str]:
    env = os.environ.copy()

    def value(bytes_count: int = 32) -> str:
        return secrets.token_hex(bytes_count)

    env.update(
        {
            "LANGFUSE_POSTGRES_PASSWORD": value(),
            "LANGFUSE_CLICKHOUSE_PASSWORD": value(),
            "LANGFUSE_REDIS_PASSWORD": value(),
            "LANGFUSE_MINIO_PASSWORD": value(),
            "LANGFUSE_SALT": value(),
            "LANGFUSE_ENCRYPTION_KEY": value(),
            "LANGFUSE_NEXTAUTH_SECRET": value(),
            "LANGFUSE_INIT_ORG_ID": "acne-advisor-local",
            "LANGFUSE_INIT_ORG_NAME": "Acne Advisor Local",
            "LANGFUSE_INIT_PROJECT_ID": "acne-advisor-observability",
            "LANGFUSE_INIT_PROJECT_NAME": "Acne Advisor Observability",
            "LANGFUSE_INIT_PROJECT_PUBLIC_KEY": f"pk-lf-{value(16)}",
            "LANGFUSE_INIT_PROJECT_SECRET_KEY": f"sk-lf-{value()}",
            "LANGFUSE_INIT_USER_EMAIL": "observability-smoke@example.invalid",
            "LANGFUSE_INIT_USER_NAME": "Observability Smoke",
            "LANGFUSE_INIT_USER_PASSWORD": value(),
            "LANGFUSE_PUBLIC_URL": "http://localhost:3001",
            "LANGFUSE_MINIO_PUBLIC_URL": "http://localhost:9090",
            "LANGFUSE_PORT": "3001",
            "LANGFUSE_MINIO_PORT": "9090",
        }
    )
    return env


def _compose(env: dict[str, str], *args: str, quiet: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "docker",
            "compose",
            "--project-name",
            PROJECT_NAME,
            "-f",
            str(COMPOSE_FILE),
            *args,
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=quiet,
    )


def _wait_for_health(timeout_seconds: float = 300.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"{BASE_URL}/api/public/health?failIfDatabaseUnavailable=true"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=4) as response:  # noqa: S310
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(3)
    raise TimeoutError("Langfuse readiness timeout")


def _assert_containers_healthy(env: dict[str, str]) -> list[str]:
    expected = {
        "langfuse-web",
        "langfuse-worker",
        "langfuse-clickhouse",
        "langfuse-minio",
        "langfuse-redis",
        "langfuse-postgres",
    }
    deadline = time.monotonic() + 120.0
    last_status: dict[str, str] = {}
    while time.monotonic() < deadline:
        completed = _compose(env, "ps", "--format", "json", quiet=True)
        entries = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
        last_status = {
            str(entry.get("Service")): str(entry.get("Health") or "")
            for entry in entries
        }
        if set(last_status) == expected and all(
            status == "healthy" for status in last_status.values()
        ):
            return sorted(last_status)
        time.sleep(2)
    raise RuntimeError(f"Langfuse containers not healthy: {last_status}")


def _configure_application(env: dict[str, str]) -> None:
    values = {
        "OBSERVABILITY_ENABLED": "true",
        "LANGFUSE_ENABLED": "true",
        "LANGFUSE_BASE_URL": BASE_URL,
        "LANGFUSE_PUBLIC_KEY": env["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"],
        "LANGFUSE_SECRET_KEY": env["LANGFUSE_INIT_PROJECT_SECRET_KEY"],
        "LANGFUSE_TRACING_ENVIRONMENT": "smoke",
        "RELEASE_READINESS_TEST_MODE": "deterministic",
    }
    os.environ.update(values)


def _make_event() -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
    from src.observability.trace_exporter import build_observability_event

    raw_question = "SYNTHETIC_PRIVATE_QUESTION_DO_NOT_EXPORT"
    raw_history = "SYNTHETIC_PRIVATE_HISTORY_DO_NOT_EXPORT"
    raw_prompt = "SYNTHETIC_PRIVATE_PROMPT_DO_NOT_EXPORT"
    raw_answer = "SYNTHETIC_PRIVATE_ANSWER_DO_NOT_EXPORT"
    raw_secret = "SYNTHETIC_PRIVATE_SECRET_DO_NOT_EXPORT"
    raw_exception = "SYNTHETIC_PRIVATE_EXCEPTION_DO_NOT_EXPORT"
    chain_of_thought = "SYNTHETIC_CHAIN_OF_THOUGHT_DO_NOT_EXPORT"
    synthetic_session = "synthetic-person@example.invalid"
    result = {
        "answer": raw_answer,
        "agent_decision": {
            "action": "generate",
            "reason_code": "evidence_sufficient",
        },
        "agent_decision_history": [
            {
                "action": "retrieve",
                "reason_code": "needs_evidence",
                "retrieval_query": raw_question,
                "missing_evidence": raw_history,
            },
            {
                "action": "generate",
                "reason_code": "evidence_sufficient",
                "private_reasoning": chain_of_thought,
            },
        ],
        "retrieval_status": "ok",
        "retrieval_attempt": 1,
        "retrieval_trace": {
            "channels": {
                "dense": {"count": 2, "error": None},
                "bm25": {"count": 1, "error": None},
            },
            "candidate_trace": {
                "dense": [
                    {"candidate_id": "dense-safe-1"},
                    {"candidate_id": "dense-safe-2"},
                ],
                "bm25": [{"candidate_id": "bm25-safe-1"}],
                "fused": [
                    {"candidate_id": "dense-safe-1"},
                    {"candidate_id": "bm25-safe-1"},
                ],
            },
            "fused_candidate_count": 2,
            "eligible_candidate_count": 2,
            "selected_ids": ["dense-safe-1", "bm25-safe-1"],
            "retry_evidence": {
                "retained_candidate_ids": ["dense-safe-1"],
                "duplicate_candidate_ids": ["dense-safe-1"],
            },
            "reranker": {
                "enabled": True,
                "model": "BAAI/bge-reranker-v2-m3",
                "elapsed_ms": 3.5,
            },
        },
        "generation_invoked": True,
        "generation_provider": "synthetic-provider",
        "generation_model": "synthetic-model",
        "llm_fallback_used": False,
        "pipeline_manifest": {
            "phase": "production",
            "kb_version": "synthetic-kb-build",
        },
        "pipeline_fingerprint": "synthetic-pipeline-fingerprint",
        "performance_timings": {
            "agent_decision_1": 1.0,
            "retrieval_total": 7.0,
            "llm_generation": 11.0,
            "total_request": 25.0,
        },
    }
    event = build_observability_event(
        query=raw_question,
        request_id=REQUEST_ID,
        session_id=synthetic_session,
        result=result,
        complete=True,
        error=RuntimeError(raw_exception),
        error_stage="api",
        safe_payload={
            "history": raw_history,
            "prompt": raw_prompt,
            "authorization": raw_secret,
        },
    )
    prohibited = (
        raw_question,
        raw_history,
        raw_prompt,
        raw_answer,
        raw_secret,
        raw_exception,
        chain_of_thought,
        synthetic_session,
    )
    required = (
        REQUEST_ID,
        "dense-safe-1",
        "synthetic-provider",
        "synthetic-model",
        "synthetic-pipeline-fingerprint",
        "synthetic-kb-build",
        "RuntimeError",
        "retained_candidate_count",
        "duplicate_candidate_count",
    )
    return event, prohibited, required


def _export_and_inspect_trace() -> dict[str, Any]:
    from src.observability import langfuse_sink
    from src.observability.langfuse_sink import (
        begin_langfuse_request,
        check_langfuse_connectivity,
        export_event_to_langfuse,
        flush_langfuse,
        submit_evaluation_score,
    )

    if not check_langfuse_connectivity():
        raise RuntimeError("Langfuse connectivity probe failed")
    handle = begin_langfuse_request(REQUEST_ID)
    if handle is None:
        raise RuntimeError("Langfuse request handle was not created")
    event, prohibited, required = _make_event()
    if not export_event_to_langfuse(event, handle=handle):
        raise RuntimeError("Langfuse trace enqueue failed")
    if not submit_evaluation_score(REQUEST_ID, "Faithfulness", 0.73):
        raise RuntimeError("Langfuse score enqueue failed")
    if not flush_langfuse():
        raise RuntimeError("Langfuse flush failed")

    sink = langfuse_sink._SINK
    if sink is None:
        raise RuntimeError("Langfuse sink disappeared during inspection")
    trace_id = sink.client.create_trace_id(seed=REQUEST_ID)
    observations: list[dict[str, Any]] = []
    for _ in range(30):
        response = sink.client.api.observations.get_many(
            trace_id=trace_id,
            limit=100,
            fields="core,basic,time,io,metadata,model,usage,prompt,trace_context",
        )
        observations = [item.model_dump(mode="json") for item in response.data]
        if len(observations) >= 8:
            break
        time.sleep(2)
    if not observations:
        raise RuntimeError("Trace was not queryable after flush")
    serialized = json.dumps(observations, ensure_ascii=False, sort_keys=True)
    if any(value in serialized for value in prohibited):
        raise AssertionError("Prohibited content was present in the actual trace")
    missing = [value for value in required if value not in serialized]
    if missing:
        raise AssertionError(f"Approved fields missing from actual trace: {len(missing)}")
    trace_id_matches_request = all(
        (item.get("trace_id") or item.get("traceId")) == trace_id
        for item in observations
    )
    if not trace_id_matches_request:
        raise AssertionError("Actual observations did not use the deterministic request trace ID")
    by_name = {item["name"]: item for item in observations}
    required_names = {
        "chat-request",
        "agent-decision",
        "retrieve",
        "dense",
        "bm25",
        "rrf",
        "rerank",
        "pack",
        "generate",
    }
    if not required_names <= set(by_name):
        raise AssertionError("Actual trace hierarchy is missing required observations")
    retrieve_id = by_name["retrieve"]["id"]
    for name in ("dense", "bm25", "rrf", "rerank", "pack"):
        parent_id = by_name[name].get("parent_observation_id") or by_name[name].get(
            "parentObservationId"
        )
        if parent_id != retrieve_id:
            raise AssertionError(f"{name} was not nested below retrieve")

    scores: list[dict[str, Any]] = []
    for _ in range(15):
        score_response = sink.client.api.scores_v3.get_many_v3(
            trace_id=trace_id,
            name="Faithfulness",
            limit=10,
        )
        scores = [item.model_dump(mode="json") for item in score_response.data]
        if scores:
            break
        time.sleep(2)
    if not scores or not any(float(item["value"]) == 0.73 for item in scores):
        raise AssertionError("The real evaluation score was not queryable on the trace")
    return {
        "connectivity": "passed",
        "trace_id_matches_request": trace_id_matches_request,
        "observation_count": len(observations),
        "observation_names": sorted(item["name"] for item in observations),
        "privacy_inspection": "passed",
        "approved_fields_present": "passed",
        "score_enqueue": "passed",
        "score_query": "passed",
    }


async def _request_while_langfuse_is_stopped() -> dict[str, Any]:
    from src.api.app import app

    started = time.perf_counter()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/chat", json={"message": "synthetic outage fixture"})
    elapsed_ms = (time.perf_counter() - started) * 1000
    if response.status_code != 200:
        raise AssertionError(f"Business request failed with HTTP {response.status_code}")
    return {
        "outage_chat_status": response.status_code,
        "outage_elapsed_ms": round(elapsed_ms, 2),
        "business_response_preserved": True,
    }


def run() -> dict[str, Any]:
    env = _bootstrap_environment()
    _configure_application(env)
    started = False
    result: dict[str, Any] = {}
    try:
        _compose(env, "config", "--quiet", quiet=True)
        started = True
        _compose(env, "up", "-d", quiet=True)
        _wait_for_health()
        result["healthy_services"] = _assert_containers_healthy(env)
        result.update(_export_and_inspect_trace())

        _compose(env, "stop", "langfuse-web", quiet=True)
        result.update(asyncio.run(_request_while_langfuse_is_stopped()))
        _compose(env, "start", "langfuse-web", quiet=True)
        _wait_for_health(120.0)
        result["recovery"] = "passed"
        print(json.dumps({"smoke_result": result}, sort_keys=True), flush=True)
        return result
    finally:
        if started:
            from src.observability.langfuse_sink import shutdown_langfuse

            shutdown_langfuse()
            _compose(env, "down", "--volumes", quiet=True)


def main() -> int:
    result = run()
    result["final_stack_state"] = "stopped"
    result["volumes"] = "ephemeral smoke volumes removed"
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
