"""
Phase 2 preflight checks for API/runtime retrieval dependencies.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "acne_knowledge")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
PREFLIGHT_CHECK_TIMEOUT_SECONDS = float(os.getenv("PREFLIGHT_CHECK_TIMEOUT_SECONDS", "4.0"))
REQUIRED_CORE_CHECKS = ("postgres", "qdrant", "neo4j", "redis")


@dataclass
class CheckResult:
    status: str
    detail: str = ""
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"status": self.status}
        if self.detail:
            data["detail"] = self.detail
        if self.extra:
            data.update(self.extra)
        return data


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _normalize_provider(value: str | None) -> str:
    provider = (value or "gemini").strip().lower()
    return "ollama" if provider == "local" else provider


def get_runtime_provider_requirements() -> dict[str, Any]:
    """Describe the configured generation runtime without probing paid providers.

    Gemini is the default runtime provider. Its opportunistic Ollama fallback is
    selected only when a local model is available, so it does not make Ollama a
    core health dependency. An explicitly configured Ollama primary provider,
    or an explicitly required local fallback, does make it required.
    """

    provider = _normalize_provider(os.getenv("LLM_PROVIDER", "gemini"))
    fallback_enabled = _env_bool("LLM_PROVIDER_FALLBACK_ENABLED")
    fallback_provider = _normalize_provider(os.getenv("LLM_FALLBACK_PROVIDER", "ollama"))
    ollama_required = provider == "ollama" or (
        fallback_enabled and fallback_provider == "ollama"
    )
    if provider == "ollama":
        reason = "configured primary provider is ollama"
    elif fallback_enabled and fallback_provider == "ollama":
        reason = "configured fallback explicitly requires ollama"
    else:
        reason = "gemini runtime uses Ollama only as an opportunistic fallback"
    return {
        "provider": provider,
        "fallback_enabled": fallback_enabled,
        "fallback_provider": fallback_provider if fallback_enabled else None,
        "ollama_required": ollama_required,
        "ollama_requirement_reason": reason,
    }


def check_generation_provider(
    requirements: dict[str, Any],
    ollama: CheckResult,
) -> CheckResult:
    """Validate the configured primary provider without sending a chat request."""

    provider = requirements["provider"]
    if provider == "gemini":
        if os.getenv("GOOGLE_API_KEY", "").strip():
            return CheckResult(
                "ok",
                "Gemini is configured; external provider connectivity is not probed by health.",
                {"provider": "gemini", "connectivity_probed": False},
            )
        return CheckResult(
            "unavailable",
            "GOOGLE_API_KEY is not configured for the default Gemini runtime.",
            {"provider": "gemini", "connectivity_probed": False},
        )
    if provider == "ollama":
        if ollama.status == "ok":
            return CheckResult("ok", extra={"provider": "ollama", "connectivity_probed": True})
        return CheckResult(
            "unavailable",
            f"Configured Ollama runtime is unavailable: {ollama.detail or ollama.status}",
            {"provider": "ollama", "connectivity_probed": True},
        )
    return CheckResult(
        "unavailable",
        f"Unsupported configured runtime provider: {provider}",
        {"provider": provider, "connectivity_probed": False},
    )


async def check_postgres() -> CheckResult:
    try:
        from src.database.connection import engine

        async with engine.connect() as conn:
            await conn.exec_driver_sql("SELECT 1")
        return CheckResult("ok")
    except Exception as exc:
        return CheckResult("unavailable", str(exc))


async def check_redis() -> CheckResult:
    try:
        from src.cache.redis_cache import ping_redis

        return CheckResult("ok" if await ping_redis() else "unavailable")
    except Exception as exc:
        return CheckResult("unavailable", str(exc))


def _get_named_config(config: Any, name: str) -> Any | None:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    if hasattr(config, "get"):
        return config.get(name)
    return None


async def check_qdrant() -> CheckResult:
    try:
        from qdrant_client import AsyncQdrantClient
        from src.database.vector_store import qdrant_client_kwargs

        client = AsyncQdrantClient(**qdrant_client_kwargs())
        try:
            info = await client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
            params = info.config.params
            if isinstance(params, dict):
                vectors_config = params.get("vectors")
                sparse_vectors_config = params.get("sparse_vectors")
            else:
                vectors_config = getattr(params, "vectors", None)
                sparse_vectors_config = getattr(params, "sparse_vectors", None)

            dense_config = _get_named_config(vectors_config, "dense")
            bm25_config = _get_named_config(sparse_vectors_config, "bm25")
            errors: list[str] = []

            if dense_config is None:
                errors.append("missing named vector dense")
                dense_size = None
            else:
                dense_size = (
                    dense_config.get("size")
                    if isinstance(dense_config, dict)
                    else getattr(dense_config, "size", None)
                )
                if dense_size != EMBEDDING_DIMENSIONS:
                    errors.append(f"dense dim {dense_size} != {EMBEDDING_DIMENSIONS}")

            if bm25_config is None:
                errors.append("missing sparse vector bm25")

            points_count = getattr(info, "points_count", None)
            extra = {
                "collection": QDRANT_COLLECTION_NAME,
                "dense_dim": dense_size,
                "has_bm25": bm25_config is not None,
                "points_count": points_count,
            }
            if errors:
                return CheckResult("schema_mismatch", "; ".join(errors), extra)
            return CheckResult("ok", extra=extra)
        finally:
            await client.close()
    except Exception as exc:
        return CheckResult(
            "unavailable",
            f"Cannot connect/authenticate to Qdrant at {QDRANT_URL}. "
            f"Check QDRANT_URL and QDRANT_API_KEY. Error: {exc}",
        )


async def check_neo4j() -> CheckResult:
    try:
        from neo4j import AsyncGraphDatabase

        driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        )
        try:
            await driver.verify_connectivity()
            async with driver.session() as session:
                result = await session.run(
                    "MATCH (n) RETURN count(n) AS nodes LIMIT 1"
                )
                record = await result.single()
            return CheckResult("ok", extra={"nodes": record["nodes"] if record else None})
        finally:
            await driver.close()
    except Exception as exc:
        return CheckResult("unavailable", str(exc))


def _http_get_json(url: str, timeout: float = 5.0) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


async def check_ollama() -> CheckResult:
    try:
        data = await asyncio.to_thread(
            _http_get_json,
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags",
        )
        models = data.get("models", []) if isinstance(data, dict) else []
        names = {
            str(model.get("name", ""))
            for model in models
            if isinstance(model, dict)
        }
        short_names = {name.split(":")[0] for name in names}
        if OLLAMA_MODEL not in names and OLLAMA_MODEL not in short_names:
            return CheckResult(
                "model_missing",
                f"{OLLAMA_MODEL} not found",
                {"available_models": sorted(names)},
            )
        return CheckResult("ok", extra={"model": OLLAMA_MODEL})
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return CheckResult("unavailable", str(exc))


async def _bounded_check(name: str, check_coro: Any) -> CheckResult:
    try:
        return await asyncio.wait_for(check_coro, timeout=PREFLIGHT_CHECK_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return CheckResult("timeout", f"{name} health check exceeded {PREFLIGHT_CHECK_TIMEOUT_SECONDS:.1f}s")


async def run_phase2_preflight() -> dict[str, Any]:
    postgres, qdrant, neo4j, redis, ollama = await asyncio.gather(
        _bounded_check("postgres", check_postgres()),
        _bounded_check("qdrant", check_qdrant()),
        _bounded_check("neo4j", check_neo4j()),
        _bounded_check("redis", check_redis()),
        _bounded_check("ollama", check_ollama()),
    )
    requirements = get_runtime_provider_requirements()
    generation = check_generation_provider(requirements, ollama)
    ollama_data = ollama.to_dict()
    ollama_data.update(
        {
            "required": requirements["ollama_required"],
            "optional": not requirements["ollama_required"],
            "requirement_reason": requirements["ollama_requirement_reason"],
        }
    )
    checks = {
        "postgres": postgres.to_dict(),
        "qdrant": qdrant.to_dict(),
        "neo4j": neo4j.to_dict(),
        "redis": redis.to_dict(),
        "ollama": ollama_data,
        "generation": generation.to_dict(),
    }
    check_results = {
        "postgres": postgres,
        "qdrant": qdrant,
        "neo4j": neo4j,
        "redis": redis,
    }
    required = [check_results[name] for name in REQUIRED_CORE_CHECKS] + [generation]
    if requirements["ollama_required"]:
        required.append(ollama)
    overall = "ok" if all(check.status == "ok" for check in required) else "degraded"
    return {"status": overall, "checks": checks}
