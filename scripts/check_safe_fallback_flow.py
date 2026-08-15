#!/usr/bin/env python3
"""Offline deterministic safety and infrastructure-fallback check."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.nodes.fallback import generation_fallback_decision_node, safe_fallback_node  # noqa: E402
from src.agent.nodes.workflow import abstain_node  # noqa: E402
from src.agent.safety_policy import evaluate_safety  # noqa: E402
from src.quality.safe_fallback import (  # noqa: E402
    SAFE_FALLBACK_FLOW_VERSION,
    build_safe_fallback_answer,
    decide_generation_fallback,
)


def _case(case_id: str, passed: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"id": case_id, "passed": bool(passed), "details": details or {}}


async def run_check() -> dict[str, Any]:
    no_evidence = await abstain_node({"retrieval_status": "no_evidence"})
    retrieval_error = await abstain_node({"retrieval_error": "backend failed"})
    safety = evaluate_safety("Sau thuốc tôi khó thở, sưng môi và nổi mề đay")
    decision = await generation_fallback_decision_node({"draft_answer": ""})
    fallback = await safe_fallback_node(decision)
    cases = [
        _case("empty_query", "chưa nhận được câu hỏi" in build_safe_fallback_answer("empty_query")),
        _case("no_source_evidence", no_evidence["fallback_type"] == "no_retrieval_evidence"),
        _case("retrieval_error", retrieval_error["fallback_type"] == "retrieval_error"),
        _case("empty_generation", decide_generation_fallback(" ").fallback_type == "empty_generation"),
        _case("invalid_generation", decide_generation_fallback(None).fallback_type == "invalid_generation"),
        _case(
            "source_mapped_safety",
            bool(safety and safety.rule_id == "anaphylaxis_like_emergency" and safety.source_ids),
        ),
        _case(
            "fallback_not_cacheable",
            decision["fallback_cache_eligible"] is False
            and fallback["fallback_cache_eligible"] is False
            and fallback["actual_provider"] == "system",
        ),
    ]
    passed_cases = sum(1 for case in cases if case["passed"])
    return {
        "name": "SAFE_FALLBACK_FLOW",
        "version": SAFE_FALLBACK_FLOW_VERSION,
        "passed": passed_cases == len(cases),
        "total_cases": len(cases),
        "passed_cases": passed_cases,
        "failed_cases": len(cases) - passed_cases,
        "cases": cases,
    }


def main() -> int:
    report = asyncio.run(run_check())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("SAFE_FALLBACK_FLOW: PASS" if report["passed"] else "SAFE_FALLBACK_FLOW: FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
