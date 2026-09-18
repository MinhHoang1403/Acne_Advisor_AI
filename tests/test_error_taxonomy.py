from __future__ import annotations

import pytest

from src.observability.error_taxonomy import ERROR_FAMILY_OWNERS, classify_error
from src.resilience.exceptions import (
    AgentTimeoutError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


@pytest.mark.parametrize(
    ("error", "stage", "expected_family", "expected_owner"),
    [
        (AgentTimeoutError("timeout"), "agent", "agent_timeout", "agent"),
        (ProviderTimeoutError("timeout"), "decide", "decision_provider", "agent/decision"),
        (ProviderTimeoutError("timeout"), "generate", "generation_timeout", "generation"),
        ("TimeoutError", "dense", "dense_channel", "retrieval"),
        ("ConnectionError", "dense_store", "dense_store", "retrieval/qdrant"),
        (ProviderUnavailableError("unavailable"), "bm25", "bm25_store", "retrieval/bm25"),
        ("UnexpectedResponse", "bm25", "bm25_store", "retrieval/bm25"),
        ("ConnectionError", "persistence", "persistence", "database"),
        ("OSError", "observability", "observability", "observability"),
    ],
)
def test_error_taxonomy_uses_existing_codes_and_explicit_stage_owner(
    error,
    stage,
    expected_family,
    expected_owner,
):
    assert classify_error(error, stage) == (expected_family, expected_owner)


def test_every_error_family_has_a_component_owner():
    assert all(owner and owner != "unknown" for family, owner in ERROR_FAMILY_OWNERS.items() if family != "unknown")
