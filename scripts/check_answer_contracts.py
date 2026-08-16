#!/usr/bin/env python3
"""Kiểm implementation offline tại evidence-grounded answer boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.answer_formatting import ANSWER_FORMATTING_CONTRACT_VERSION  # noqa: E402
from src.agent.prompts.medical_answer import MEDICAL_RAG_SYSTEM_PROMPT  # noqa: E402
from src.observability.versioning import build_pipeline_version_manifest  # noqa: E402


def run_answer_contract_checks() -> dict[str, Any]:
    """Kiểm architecture markers; đây không phải đánh giá chất lượng lâm sàng."""

    formatter_source = (PROJECT_ROOT / "src/agent/answer_formatting.py").read_text(encoding="utf-8")
    verifier_source = (PROJECT_ROOT / "src/quality/answer_verifier.py").read_text(encoding="utf-8")
    manifest = build_pipeline_version_manifest()
    checks = {
        "formatting_contract_v14": ANSWER_FORMATTING_CONTRACT_VERSION == "answer_formatting_contract_v14",
        "evidence_grounding_contract": manifest.get("evidence_grounding_version") == "evidence_grounded_runtime_v1",
        "system_policy_requires_evidence": "evidence" in MEDICAL_RAG_SYSTEM_PROMPT.casefold(),
        "no_taxonomy_in_formatter": "DrugEntityNormalizer" not in formatter_source,
        "no_medical_semantic_verifier": "medical_semantic_verification\": False" in verifier_source,
        "proposition_engine_removed": not (PROJECT_ROOT / "src/quality/proposition_detector.py").exists(),
    }
    return {
        "passed": all(checks.values()),
        "scope": "implementation_contracts_not_clinical_quality",
        "checks": checks,
    }


def main() -> int:
    report = run_answer_contract_checks()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
