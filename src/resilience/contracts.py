"""Contracts and settings for runtime resilience."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeResilienceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_total_timeout_seconds: float = Field(default=210.0, gt=0)
    retrieval_timeout_seconds: float = Field(default=20.0, gt=0)
    gemini_timeout_seconds: float = Field(default=45.0, gt=0)
    ollama_timeout_seconds: float = Field(default=160.0, gt=0)

    llm_max_retries: int = Field(default=1, ge=0)
    llm_retry_base_delay_seconds: float = Field(default=1.0, ge=0)
    llm_retry_max_delay_seconds: float = Field(default=4.0, ge=0)

    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = Field(default=3, ge=1)
    circuit_breaker_recovery_seconds: float = Field(default=60.0, gt=0)
    circuit_breaker_half_open_max_calls: int = Field(default=1, ge=1)

    llm_provider_fallback_enabled: bool = False
    llm_fallback_provider: str = "ollama"

    @model_validator(mode="after")
    def _validate_retry_delays(self) -> "RuntimeResilienceSettings":
        if self.llm_retry_max_delay_seconds < self.llm_retry_base_delay_seconds:
            raise ValueError("LLM_RETRY_MAX_DELAY_SECONDS must be >= LLM_RETRY_BASE_DELAY_SECONDS")
        return self


def runtime_resilience_settings_from_env(env: dict[str, str] | None = None) -> RuntimeResilienceSettings:
    source = env or os.environ

    def get_float(name: str, default: float) -> float:
        value = source.get(name)
        return default if value in {None, ""} else float(str(value))

    def get_int(name: str, default: int) -> int:
        value = source.get(name)
        return default if value in {None, ""} else int(str(value))

    def get_bool(name: str, default: bool) -> bool:
        value = source.get(name)
        if value in {None, ""}:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    data: dict[str, Any] = {
        "agent_total_timeout_seconds": get_float("AGENT_TOTAL_TIMEOUT_SECONDS", 210.0),
        "retrieval_timeout_seconds": get_float("RETRIEVAL_TIMEOUT_SECONDS", 20.0),
        "gemini_timeout_seconds": get_float("GEMINI_TIMEOUT_SECONDS", 45.0),
        "ollama_timeout_seconds": get_float("OLLAMA_TIMEOUT_SECONDS", 160.0),
        "llm_max_retries": get_int("LLM_MAX_RETRIES", 1),
        "llm_retry_base_delay_seconds": get_float("LLM_RETRY_BASE_DELAY_SECONDS", 1.0),
        "llm_retry_max_delay_seconds": get_float("LLM_RETRY_MAX_DELAY_SECONDS", 4.0),
        "circuit_breaker_enabled": get_bool("CIRCUIT_BREAKER_ENABLED", True),
        "circuit_breaker_failure_threshold": get_int("CIRCUIT_BREAKER_FAILURE_THRESHOLD", 3),
        "circuit_breaker_recovery_seconds": get_float("CIRCUIT_BREAKER_RECOVERY_SECONDS", 60.0),
        "circuit_breaker_half_open_max_calls": get_int("CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS", 1),
        "llm_provider_fallback_enabled": get_bool("LLM_PROVIDER_FALLBACK_ENABLED", False),
        "llm_fallback_provider": source.get("LLM_FALLBACK_PROVIDER", "ollama") or "ollama",
    }
    return RuntimeResilienceSettings.model_validate(data)


__all__ = ["RuntimeResilienceSettings", "runtime_resilience_settings_from_env"]
