"""LangGraph node for optional Phase 2 observability export."""

from __future__ import annotations

import logging
import os
from typing import Any

from src.agent.state import ClinicalState
from src.observability.trace_exporter import build_observability_event, export_observability_event

logger = logging.getLogger(__name__)


async def observability_export_node(state: ClinicalState) -> dict[str, Any]:
    """Export a sanitized trace when observability is explicitly enabled."""

    enabled = os.getenv("OBSERVABILITY_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return {"observability_exported": False}

    try:
        query = state.get("standalone_question") or state.get("user_question", "")
        event = build_observability_event(
            query=query,
            state=dict(state),
            session_id=state.get("session_id"),
            pipeline_manifest=state.get("pipeline_manifest"),
            pipeline_fingerprint=state.get("pipeline_fingerprint"),
        )
        exported = export_observability_event(
            event,
            output_dir=os.getenv("OBSERVABILITY_TRACE_DIR", "logs/phase2_traces"),
            enabled=True,
        )
        return {"observability_exported": exported}
    except Exception as exc:
        logger.warning("Observability export failed safely: %s", exc)
        return {"observability_exported": False}


__all__ = ["observability_export_node"]
