from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess
import sys

from scripts.answer_quality_diagnostics import (
    _agent_observation,
    _channel_trace,
    _fused_candidate_trace,
    assess_retrieval_coverage,
    load_diagnostic_cases,
    run_live_agent_diagnostic,
    run_live_retrieval_diagnostic,
)
from src.retrieval.context_packer import pack_context
from src.retrieval.contracts import NormalizedQuery, RetrievedCandidate


CASES_PATH = Path(__file__).parent / "fixtures" / "answer_quality_diagnostic_cases.json"


def test_diagnostic_cases_have_traceable_evidence_and_required_case_groups() -> None:
    cases = load_diagnostic_cases(CASES_PATH)
    by_id = {case["case_id"]: case for case in cases}

    assert {
        "bp_antibiotic",
        "adapalene_class",
        "clindamycin_monotherapy",
        "adapalene_bp_comparison",
        "retinoid_pregnancy",
        "dermatologist_referral",
        "isotretinoin_pregnancy",
        "historical_02",
        "historical_03",
        "historical_04",
        "paraphrase_bp_classification",
        "paraphrase_adapalene_class",
        "paraphrase_comedone_difference",
        "paraphrase_inflammatory_lesions",
        "paraphrase_referral",
        "historical_16",
        "conversation_adapalene_pregnancy",
        "conversation_entity_reference",
        "safety_cross_subject",
        "safety_same_subject",
        "safety_resolved_history",
    } <= by_id.keys()

    for case in cases:
        assert case["expected_behavior"]
        assert case["assessment_mode"]
        if case["assessment_mode"] == "retrieval_evidence":
            assert case["evidence_groups"]
            assert all(group["any_of"] for group in case["evidence_groups"])

    historical_mask = by_id["historical_16"]
    assert historical_mask["assessment_mode"] == "source_scope_review"
    assert "not masks or face coverings" in historical_mask["review_notes"]

    assert by_id["bp_antibiotic"]["evidence_groups"][0]["any_of"] == [
        "db4632c5-44ed-5dea-80c2-addb0a2534b3"
    ]
    assert by_id["adapalene_class"]["evidence_groups"][0]["any_of"] == [
        "460222d6-c4ba-56f9-a26c-bf255b6afb39",
        "850609c9-471a-54cc-934f-478ad48d826d",
    ]
    assert by_id["clindamycin_monotherapy"]["evidence_groups"][0]["any_of"] == [
        "f675512f-93f5-572a-b705-35132a1445b7"
    ]
    assert by_id["paraphrase_referral"]["evidence_groups"][0]["any_of"] == [
        "2326dc1b-cf4b-520b-8798-f9617fafc0b0",
        "0564b723-4643-5aa0-ae71-c322c9871dd7",
    ]


def test_retrieval_coverage_classifies_candidate_and_context_loss_without_semantic_claims() -> None:
    cases = {case["case_id"]: case for case in load_diagnostic_cases(CASES_PATH)}
    retinoid_case = cases["retinoid_pregnancy"]
    comparison_case = cases["adapalene_bp_comparison"]

    retrieval_miss = assess_retrieval_coverage(
        retinoid_case,
        candidate_ids=["unrelated-chunk"],
        packed_ids=["unrelated-chunk"],
    )
    assert retrieval_miss["classification"] == "retrieval_miss"
    assert retrieval_miss["semantic_truth_checked"] is False

    context_loss = assess_retrieval_coverage(
        comparison_case,
        candidate_ids=["460222d6-c4ba-56f9-a26c-bf255b6afb39", "db4632c5-44ed-5dea-80c2-addb0a2534b3", "66d47093-ac23-5331-8955-8867287370cb"],
        packed_ids=["460222d6-c4ba-56f9-a26c-bf255b6afb39", "66d47093-ac23-5331-8955-8867287370cb"],
    )
    assert context_loss["classification"] == "context_missing_required_fact"

    evidence_packed = assess_retrieval_coverage(
        comparison_case,
        candidate_ids=["460222d6-c4ba-56f9-a26c-bf255b6afb39", "db4632c5-44ed-5dea-80c2-addb0a2534b3", "66d47093-ac23-5331-8955-8867287370cb"],
        packed_ids=["460222d6-c4ba-56f9-a26c-bf255b6afb39", "db4632c5-44ed-5dea-80c2-addb0a2534b3", "66d47093-ac23-5331-8955-8867287370cb"],
    )
    assert evidence_packed["classification"] == "evidence_packed"


def test_fused_candidate_trace_reports_packer_budget_without_reselecting() -> None:
    candidates = [
        RetrievedCandidate(
            candidate_id="selected",
            collection="knowledge",
            text="Selected evidence.",
            payload={"source_id": "source-a"},
            rank=1,
            debug={"dense_rank": 1, "bm25_rank": None},
        ),
        RetrievedCandidate(
            candidate_id="dropped",
            collection="knowledge",
            text="Later evidence.",
            payload={"source_id": "source-b"},
            rank=2,
            debug={"dense_rank": 2, "bm25_rank": None},
        ),
    ]
    packed = pack_context(
        NormalizedQuery(original_query="question", normalized_text="question"),
        candidates,
        max_items=1,
        max_chars=1000,
    )

    trace = _fused_candidate_trace(candidates, packed)

    assert trace[0]["packed"] is True
    assert trace[0]["drop_reason"] is None
    assert trace[1]["packed"] is False
    assert trace[1]["drop_reason"] is None
    assert trace[1]["cumulative_chars_before"] > 0
    assert trace[1]["remaining_chars_before"] < 1000


def test_channel_trace_preserves_rank_and_native_score() -> None:
    trace = _channel_trace(
        [
            {"id": "first", "score": 0.9},
            {"chunk_id": "second", "score": 0.5},
        ]
    )

    assert trace == [
        {"candidate_id": "first", "rank": 1, "native_score": 0.9},
        {"candidate_id": "second", "rank": 2, "native_score": 0.5},
    ]


def test_source_scope_review_does_not_invoke_retrieval(monkeypatch) -> None:
    class _NoRetrieval:
        called = False

        async def close(self) -> None:
            return None

    cases = {case["case_id"]: case for case in load_diagnostic_cases(CASES_PATH)}
    retriever = _NoRetrieval()
    monkeypatch.setattr(
        "scripts.answer_quality_diagnostics.EvidenceRetriever",
        lambda: retriever,
    )

    observations = asyncio.run(run_live_retrieval_diagnostic([cases["historical_16"]]))

    assert retriever.called is False
    assert observations == [
        {
            "case_id": "historical_16",
            "classification": "not_run",
            "reason": "source_scope_review does not treat analogous evidence as direct retrieval support.",
            "semantic_truth_checked": False,
        }
    ]


def test_agent_diagnostic_preserves_attempt_order_and_provider_trace(monkeypatch) -> None:
    cases = {case["case_id"]: case for case in load_diagnostic_cases(CASES_PATH)}

    async def fake_run_clinical_agent(*_args, **kwargs):
        assert kwargs["bypass_cache"] is True
        assert kwargs["allow_model_fallback"] is True
        assert kwargs["include_generation_diagnostics"] is True
        return {
            "standalone_question": "Adapalene thuộc nhóm thuốc gì?",
            "retrieval_attempt": 2,
            "agent_decision_history": [{"action": "retrieve"}, {"action": "retry"}],
            "agent_decision_evidence_traces": [
                {
                    "decision_index": 3,
                    "retrieval_attempts_used": 2,
                    "packed_evidence_ids": ["460222d6-c4ba-56f9-a26c-bf255b6afb39"],
                    "decision_visible_evidence_ids": [
                        "460222d6-c4ba-56f9-a26c-bf255b6afb39"
                    ],
                    "decision_visible_items": [
                        {
                            "item_id": "460222d6-c4ba-56f9-a26c-bf255b6afb39",
                            "source_id": "source",
                            "position_in_packed_context": 1,
                            "original_text_length": 900,
                            "decision_visible_text_length": 900,
                            "truncated_for_decision": False,
                        }
                    ],
                }
            ],
            "retrieval_attempt_traces": [
                {
                    "attempt_index": 1,
                    "candidate_trace": {"fused": [{"candidate_id": "unrelated"}]},
                    "packed_evidence": [],
                },
                {
                    "attempt_index": 2,
                    "candidate_trace": {"fused": [{"candidate_id": "460222d6-c4ba-56f9-a26c-bf255b6afb39"}]},
                    "packed_evidence": [
                        {"item_id": "460222d6-c4ba-56f9-a26c-bf255b6afb39", "source_id": "source"}
                    ],
                },
            ],
            "generation_evidence_trace": {
                "conversation_history_messages": 0,
                "packed_evidence": [
                    {"item_id": "460222d6-c4ba-56f9-a26c-bf255b6afb39", "source_id": "source"}
                ],
            },
            "requested_provider": "gemini",
            "requested_model": "configured",
            "actual_provider": "gemini",
            "actual_model": "fallback",
            "llm_fallback_used": True,
            "fallback_provider": "gemini",
            "fallback_model": "fallback",
            "fallback_chain": [{"status": "failed"}, {"status": "success"}],
            "generation_diagnostics": {
                "raw_generated_answer": "Raw answer.",
                "pre_verifier_answer": "Presented answer.",
            },
            "answer_quality_report": {"passed": True, "issues": []},
            "answer": "Câu trả lời cần review.",
            "fallback_applied": False,
            "safety_decision": None,
        }

    monkeypatch.setattr("scripts.answer_quality_diagnostics.run_clinical_agent", fake_run_clinical_agent)
    observations = asyncio.run(run_live_agent_diagnostic([cases["adapalene_class"]]))

    observation = observations[0]
    assert observation["classification"] == "requires_review"
    assert [item["attempt_index"] for item in observation["retrieval_attempts"]] == [1, 2]
    assert observation["attempt_coverage"][0]["classification"] == "retrieval_miss"
    assert observation["generation_coverage"]["classification"] == "evidence_packed"
    assert observation["evidence_path"] == "evidence_recovered_by_retry"
    assert observation["provider_execution"]["actual_model"] == "fallback"
    assert observation["raw_generated_answer"] == "Raw answer."
    assert observation["pre_verifier_answer"] == "Presented answer."
    assert observation["verifier_outcome"]["passed"] is True
    assert observation["decision_evidence"][0]["decision_index"] == 3
    assert "text" not in observation["decision_evidence"][0]["decision_visible_items"][0]
    assert observation["semantic_truth_checked"] is False
    assert observation["routing_classification"] == "REQUIRES_REVIEW"


def test_agent_diagnostic_identifies_decision_view_loss_false_abstention() -> None:
    cases = {case["case_id"]: case for case in load_diagnostic_cases(CASES_PATH)}
    gold_id = "db4632c5-44ed-5dea-80c2-addb0a2534b3"
    result = {
        "retrieval_attempt": 2,
        "agent_decision_history": [{"action": "abstain", "reason_code": "evidence_gap"}],
        "agent_decision_evidence_traces": [
            {
                "decision_visible_evidence_ids": ["unrelated-visible"],
                "decision_visible_items": [],
            }
        ],
        "retrieval_attempt_traces": [
            {
                "candidate_trace": {"fused": [{"candidate_id": gold_id}]},
                "packed_evidence": [{"item_id": gold_id}],
            }
        ],
        "fallback_applied": True,
        "fallback_reason_code": "insufficient_evidence",
        "fallback_type": "no_retrieval_evidence",
        "actual_provider": "system",
        "safety_decision": None,
    }

    observation = _agent_observation(cases["bp_antibiotic"], result)

    assert observation["routing_classification"] == "DECISION_VIEW_LOSS_FALSE_ABSTENTION"
    assert observation["decision_coverage"][-1]["packed_complete"] is False
    assert observation["attempt_coverage"][-1]["packed_complete"] is True


def test_agent_diagnostic_distinguishes_provider_fallback_from_evidence_gap() -> None:
    cases = {case["case_id"]: case for case in load_diagnostic_cases(CASES_PATH)}
    result = {
        "retrieval_attempt": 0,
        "agent_decision_history": [{"action": "abstain"}],
        "agent_decision_evidence_traces": [],
        "retrieval_attempt_traces": [],
        "fallback_applied": True,
        "fallback_reason_code": "provider_unavailable",
        "fallback_type": "provider_error",
        "actual_provider": "system",
        "safety_decision": None,
    }

    observation = _agent_observation(cases["bp_antibiotic"], result)

    assert observation["routing_classification"] == "PROVIDER_SAFE_FALLBACK"


def test_agent_diagnostic_keeps_source_scope_case_non_executing(monkeypatch) -> None:
    cases = {case["case_id"]: case for case in load_diagnostic_cases(CASES_PATH)}

    async def fail_if_called(*_args, **_kwargs):
        raise AssertionError("source scope review must not invoke the Agent")

    monkeypatch.setattr("scripts.answer_quality_diagnostics.run_clinical_agent", fail_if_called)
    observations = asyncio.run(run_live_agent_diagnostic([cases["historical_16"]]))

    assert observations == [
        {
            "case_id": "historical_16",
            "classification": "source_scope_review",
            "reason": "Analogical evidence remains outside direct retrieval gold coverage.",
            "semantic_truth_checked": False,
        }
    ]


def test_agent_diagnostic_passes_prior_turns_as_bounded_history(monkeypatch) -> None:
    cases = {case["case_id"]: case for case in load_diagnostic_cases(CASES_PATH)}
    histories: list[list[dict[str, str]]] = []

    async def fake_run_clinical_agent(question, **kwargs):
        history = list(kwargs["conversation_history"])
        histories.append(history)
        return {
            "standalone_question": question,
            "retrieval_attempt": 0,
            "agent_decision_history": [],
            "retrieval_attempt_traces": [],
            "generation_evidence_trace": {
                "conversation_history_messages": len(history),
                "packed_evidence": [],
            },
            "requested_provider": "mock",
            "requested_model": "mock-model",
            "actual_provider": "mock",
            "actual_model": "mock-model",
            "llm_fallback_used": False,
            "fallback_provider": None,
            "fallback_model": None,
            "fallback_chain": [],
            "answer": f"Answer for {question}",
            "fallback_applied": False,
            "safety_decision": None,
        }

    monkeypatch.setattr("scripts.answer_quality_diagnostics.run_clinical_agent", fake_run_clinical_agent)
    observations = asyncio.run(run_live_agent_diagnostic([cases["conversation_adapalene_pregnancy"]]))

    turns = observations[0]["turns"]
    assert histories[0] == []
    assert histories[1][0]["role"] == "user"
    assert histories[1][1]["role"] == "assistant"
    assert turns[1]["conversation_history_messages"] == 2
    assert turns[1]["semantic_truth_checked"] is False


def test_diagnostic_cli_runs_directly_from_repository_root() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/answer_quality_diagnostics.py", "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Read-only source-grounded retrieval diagnostic" in completed.stdout
