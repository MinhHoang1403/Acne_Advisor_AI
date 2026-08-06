"""Isolated live-evaluation stage that calls the agent service directly."""

from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any, Callable

from src.agent.graph import run_clinical_agent

from .models import EvaluationConfig


def assert_isolated(config: EvaluationConfig) -> None:
    """Reject configurations that could touch the public cache or chat persistence."""

    if not config.bypass_cache:
        raise ValueError("Evaluation V3 requires --bypass-cache.")
    if not config.no_persistence:
        raise ValueError("Evaluation V3 requires --no-persistence.")


async def run_live_case_async(case: dict[str, Any], config: EvaluationConfig) -> dict[str, Any]:
    """Run one case on the caller's event loop without persistence or cache."""

    assert_isolated(config)
    started = time.perf_counter()
    try:
        result = await run_clinical_agent(
            message=str(case["question"]),
            user_id=None,
            session_id=f"evaluation-v3-{case['id']}",
            conversation_history=list(case.get("conversation_history") or []),
            llm_provider=config.live_provider,
            llm_model=config.live_model,
            allow_model_fallback=False,
            bypass_cache=True,
            evaluation_mode=True,
        )
        return {
            "case_id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "ok": True,
            "error": None,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "requested_provider": config.live_provider,
            "requested_model": config.live_model,
            "persistence_enabled": False,
            "cache_read_enabled": False,
            "cache_write_enabled": False,
            "result": result,
        }
    except Exception as exc:  # Results must retain a failed case for checkpoint/resume.
        return {
            "case_id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "ok": False,
            "error": f"{exc.__class__.__name__}: {exc}",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "requested_provider": config.live_provider,
            "requested_model": config.live_model,
            "persistence_enabled": False,
            "cache_read_enabled": False,
            "cache_write_enabled": False,
            "result": {},
        }


async def close_evaluation_runtime_clients() -> None:
    """Release shared async read clients on the loop that created them.

    Batch evaluation intentionally keeps these clients warm across cases. The
    synchronous single-case compatibility wrapper, however, creates a fresh
    event loop each time, so it must release them before that loop closes.
    """

    from src.database.graph_store import Neo4jGraphStore
    from src.database.vector_store import QdrantVectorStore
    from src.retrieval.entity_retriever import EntityRetriever

    await asyncio.gather(
        QdrantVectorStore.close_shared_client(),
        Neo4jGraphStore.close_shared_driver(),
        EntityRetriever.close_shared_client(),
        return_exceptions=True,
    )


async def _run_isolated_live_case(case: dict[str, Any], config: EvaluationConfig) -> dict[str, Any]:
    try:
        return await run_live_case_async(case, config)
    finally:
        await close_evaluation_runtime_clients()


def run_live_case(case: dict[str, Any], config: EvaluationConfig) -> dict[str, Any]:
    """Compatibility wrapper for one isolated live-evaluation case."""

    return asyncio.run(_run_isolated_live_case(case, config))


def component_checks() -> dict[str, Any]:
    """Static checks describing the runner's non-mutating execution boundary."""

    return {
        "execution_path": {
            "passed": True,
            "detail": "Direct local call to run_clinical_agent; no public /chat request.",
        },
        "persistence": {
            "passed": True,
            "enabled": False,
            "detail": "The API persistence layer is not invoked.",
        },
        "cache": {
            "passed": True,
            "read_enabled": False,
            "write_enabled": False,
            "detail": "bypass_cache=True skips both cache lookup and cache store nodes.",
        },
        "database_writes": {
            "passed": True,
            "detail": "Runner only reads retrieval stores through the agent pipeline.",
        },
    }


__all__ = [
    "assert_isolated",
    "close_evaluation_runtime_clients",
    "component_checks",
    "run_live_case",
    "run_live_case_async",
]
