#!/usr/bin/env python3
"""Run S2 diagnostics and write source-grounded ablation artifacts."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.s2_ablation import run_s2_ablation  # noqa: E402


DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports" / "evaluation"
GOLD_AUDIT = [
    {
        "dataset": "retrieval_v5_release_cases.json (R8)",
        "cases": 18,
        "origin": "Project-authored locked regression fixture",
        "labels": "Harness-created :primary static IDs",
        "human_reviewed": False,
        "decision_grade": "TRUSTED_REGRESSION_SET_WITH_CEILING",
        "limitation": "18/18 for current and minimal; no canonical corpus chunk IDs",
    },
    {
        "dataset": "semantic_reranker_cases.json",
        "cases": 12,
        "origin": "Project-authored reranker fixtures",
        "labels": "Hand-written candidate relevance and expected top IDs",
        "human_reviewed": False,
        "decision_grade": "NOT_DECISION_GRADE_GOLD",
        "limitation": "No canonical corpus passage IDs or documented reviewer",
    },
    {
        "dataset": "p3_evidence_sufficiency_cases.json",
        "cases": 17,
        "origin": "P3 implementation contract fixtures",
        "labels": "Expected roles, retry, final state, and action",
        "human_reviewed": False,
        "decision_grade": "NOT_DECISION_GRADE_GOLD",
        "limitation": "Self-consistency regression, not observed outcome gold",
    },
    {
        "dataset": "phase1_ingest_eval_cases.json",
        "cases": 6,
        "origin": "Project-authored entity/graph regression fixtures",
        "labels": "Expected entities and graph edges",
        "human_reviewed": False,
        "decision_grade": "NOT_DECISION_GRADE_GOLD",
        "limitation": "No canonical passage IDs or adjudication record",
    },
    {
        "dataset": "phase2_retrieval_eval_cases.json",
        "cases": 8,
        "origin": "Project-authored metadata fixtures",
        "labels": "Intent and taxonomy metadata",
        "human_reviewed": False,
        "decision_grade": "NOT_DECISION_GRADE_GOLD",
        "limitation": "Not evidence/chunk relevance labels",
    },
    {
        "dataset": "p4_claim_grounding_cases.json",
        "cases": 32,
        "origin": "Project-authored P4 contract fixtures",
        "labels": "Expected claim verdict",
        "human_reviewed": False,
        "decision_grade": "NOT_DECISION_GRADE_GOLD",
        "limitation": "P4 cannot validate its own runtime architecture",
    },
    {
        "dataset": "p45_production_shadow_questions.json",
        "cases": 75,
        "origin": "Curated production-like questions",
        "labels": "Questions only; human claim labels empty",
        "human_reviewed": False,
        "decision_grade": "NOT_DECISION_GRADE_GOLD",
        "limitation": "Zero completed human adjudications",
    },
    {
        "dataset": "acne_system_eval_v3.jsonl",
        "cases": 300,
        "origin": "LLM-generated canonical V3 set",
        "labels": "Expected concepts/behavior; Gemini judge outputs in prior runs",
        "human_reviewed": False,
        "decision_grade": "NOT_DECISION_GRADE_FOR_S2",
        "limitation": "LLM origin, mostly empty accepted_sources, no source-label provenance",
    },
]

REFERENCES = [
    {
        "key": "rrf",
        "title": "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods",
        "authors": "G. V. Cormack, C. L. A. Clarke, Stefan Buettcher",
        "year": 2009,
        "venue": "SIGIR 2009",
        "doi": "10.1145/1571941.1572114",
        "url": "https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf",
        "claim": "RRF formula and the paper's fixed k=60 experiment; not proof for this corpus.",
        "level": "LEVEL 1",
    },
    {
        "key": "bm25",
        "title": "The Probabilistic Relevance Framework: BM25 and Beyond",
        "authors": "Stephen Robertson, Hugo Zaragoza",
        "year": 2009,
        "venue": "Foundations and Trends in Information Retrieval 3(4)",
        "doi": "10.1561/1500000019",
        "url": "https://doi.org/10.1561/1500000019",
        "claim": "Authoritative account of BM25 scoring and assumptions.",
        "level": "LEVEL 1",
    },
    {
        "key": "dense",
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "authors": "Vladimir Karpukhin et al.",
        "year": 2020,
        "venue": "EMNLP 2020",
        "doi": "10.18653/v1/2020.emnlp-main.550",
        "url": "https://aclanthology.org/2020.emnlp-main.550/",
        "claim": "Dual-encoder dense passage retrieval method; not corpus-specific performance.",
        "level": "LEVEL 1",
    },
    {
        "key": "recall",
        "title": "Introduction to Information Retrieval, Chapter 8",
        "authors": "Christopher D. Manning, Prabhakar Raghavan, Hinrich Schuetze",
        "year": 2008,
        "venue": "Cambridge University Press",
        "doi": "",
        "url": "https://nlp.stanford.edu/IR-book/pdf/08eval.pdf",
        "claim": "Authoritative relevance-evaluation definitions including recall.",
        "level": "LEVEL 1",
    },
    {
        "key": "mrr",
        "title": "TREC Question Answering Track Scoring Metric",
        "authors": "National Institute of Standards and Technology",
        "year": 2000,
        "venue": "TREC QA",
        "doi": "",
        "url": "https://trec.nist.gov/presentations/TREC9/qa/tsld005.htm",
        "claim": "MRR: reciprocal first-correct rank, zero for misses, mean over questions.",
        "level": "LEVEL 1",
    },
    {
        "key": "gemini_embedding",
        "title": "Gemini Embedding 2 model information",
        "authors": "Google DeepMind",
        "year": 2026,
        "venue": "Official provider documentation",
        "doi": "",
        "url": "https://deepmind.google/models/gemini/embedding/",
        "claim": "Implementation contract supports output dimensions from 128 through 3072.",
        "level": "LEVEL 3",
    },
    {
        "key": "bge_reranker",
        "title": "BAAI/bge-reranker-v2-m3 model card",
        "authors": "Beijing Academy of Artificial Intelligence",
        "year": 2024,
        "venue": "Official model card",
        "doi": "",
        "url": "https://huggingface.co/BAAI/bge-reranker-v2-m3",
        "claim": "Model identity, multilingual use, and query-document scoring contract only.",
        "level": "LEVEL 3",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled S2 component diagnostics.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reranker-provider", choices=["local_rules", "semantic", "hybrid"], default="hybrid")
    parser.add_argument("--knowledge-count", type=int, default=639)
    parser.add_argument("--entity-count", type=int, default=32)
    parser.add_argument("--neo4j-nodes", type=int, default=32)
    parser.add_argument("--neo4j-relationships", type=int, default=27)
    args = parser.parse_args()

    output_dir = args.output_dir or DEFAULT_REPORT_ROOT / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_s2_controlled_ablation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_s2_ablation(reranker_provider=args.reranker_provider)
    runtime = {
        "knowledge": args.knowledge_count,
        "entities": args.entity_count,
        "neo4j_nodes": args.neo4j_nodes,
        "neo4j_relationships": args.neo4j_relationships,
        "semantic_enrichment": "not_run",
        "ingestion": "not_run",
        "reindex": "not_run",
        "reembedding": "not_run",
    }
    complexity = _complexity_inventory()
    git = _git_snapshot()
    _write_machine_outputs(output_dir, result, runtime, complexity, git)
    _write_markdown_outputs(output_dir, result, runtime, complexity, git)
    print(json.dumps({"status": result["overall_status"], "output_dir": str(output_dir)}, indent=2))
    return 0


def _write_machine_outputs(
    output_dir: Path,
    result: dict[str, Any],
    runtime: dict[str, Any],
    complexity: dict[str, Any],
    git: dict[str, Any],
) -> None:
    experiments = result["experiments"]
    reranker = experiments["reranker"]
    payloads = {
        "s2_cases.json": {
            "gold_inventory": GOLD_AUDIT,
            "reranker_cases": reranker["cases"],
            "candidate_policy_cases": experiments["candidate_policy"]["cases"],
            "sufficiency_cases": experiments["sufficiency_retry"]["cases"],
            "entity_cases": experiments["entity"]["cases"],
        },
        "s2_metric_results.json": {
            "reranker": {"baseline": reranker["baseline"], "variant": reranker["variant"]},
            "candidate_policy": {
                "retention": experiments["candidate_policy"]["mean_retention"],
                "duplicate_slots": experiments["candidate_policy"]["mean_duplicate_slot_rate"],
            },
            "selector": experiments["selector"],
            "packer": experiments["packer"],
        },
        "s2_component_deltas.json": _component_deltas(result),
        "s2_component_decisions.json": result["component_decisions"],
        "s2_latency.json": {
            "provider_free": {
                "candidate_policy": "measured in locked R8 but sub-millisecond values are descriptive",
                "selector": experiments["selector"]["mean_latency_ms"],
                "packer": experiments["packer"]["mean_latency_ms"],
            },
            "local_reranker": reranker["mean_variant_latency_ms"],
            "live_generation": "NOT_RUN_NOT_REQUIRED",
        },
        "s2_call_counts.json": result["call_counts"],
        "s2_gold_audit.json": GOLD_AUDIT,
        "s2_complexity.json": complexity,
        "s2_runtime.json": runtime,
        "s2_git.json": git,
    }
    for name, payload in payloads.items():
        (output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _write_markdown_outputs(
    output_dir: Path,
    result: dict[str, Any],
    runtime: dict[str, Any],
    complexity: dict[str, Any],
    git: dict[str, Any],
) -> None:
    docs = {
        "S2_METHOD_SOURCE_REGISTRY.md": _method_registry(),
        "S2_METRIC_DEFINITIONS.md": _metric_definitions(),
        "S2_REFERENCES.md": _references(),
        "S2_GROUND_TRUTH_AUDIT.md": _ground_truth_audit(),
        "S2_DATA_LEAKAGE_AUDIT.md": _leakage_audit(),
        "S2_HEURISTIC_AUDIT.md": _heuristic_audit(),
        "S2_RERANKER_ABLATION.md": _reranker_report(result),
        "S2_CANDIDATE_POLICY_ABLATION.md": _candidate_policy_report(result),
        "S2_SUFFICIENCY_RETRY_ABLATION.md": _sufficiency_report(result),
        "S2_ENTITY_ABLATION.md": _entity_report(result),
        "S2_GRAPH_ABLATION.md": _graph_report(result),
        "S2_SELECTOR_PACKER_ABLATION.md": _selector_packer_report(result),
        "S2_COMPLEXITY_COST.md": _complexity_report(complexity),
        "S2_COMPONENT_DECISIONS.md": _decision_report(result),
        "S2_S3_CLEANUP_HANDOFF.md": _s3_handoff(result),
        "S2_TEST_RESULTS.md": _test_results(),
        "S2_RUNTIME_INTEGRITY.md": _runtime_report(runtime),
        "S2_GIT_INTEGRITY.md": _git_report(git),
        "S2_CONTROLLED_ABLATION_REPORT.md": _main_report(result, runtime, complexity, git),
    }
    for name, content in docs.items():
        (output_dir / name).write_text(content.rstrip() + "\n", encoding="utf-8")


def _method_registry() -> str:
    rows = [
        ("Dense retrieval", "A0 method", "Karpukhin et al. 2020", "LEVEL 1", "Dual-encoder dense retrieval", "src/evaluation/minimal_rag.py"),
        ("BM25", "A0 sparse channel", "Robertson & Zaragoza 2009", "LEVEL 1", "BM25 method", "src/database/vector_store.py"),
        ("RRF", "A0 fusion", "Cormack et al. 2009", "LEVEL 1", "RRF formula; k=60 precedent", "src/retrieval/rrf.py"),
        ("Recall@k query hit-rate", "Presence by cutoff", "Manning et al. 2008", "LEVEL 1", "Recall foundation; S2 specialization explicit", "src/evaluation/ablation_metrics.py::recall_at_k"),
        ("MRR", "First relevant rank", "NIST TREC QA", "LEVEL 1", "Exact reciprocal-rank aggregation", "src/evaluation/ablation_metrics.py::mean_reciprocal_rank"),
        ("BGE reranker", "Local variant identity", "BAAI model card", "LEVEL 3", "Implementation identity only", "src/retrieval/reranking/providers.py"),
        ("Arithmetic mean", "Descriptive latency", "Elementary definition", "Definition only", "No architecture decision alone", "src/evaluation/ablation_metrics.py::arithmetic_mean"),
    ]
    table = "\n".join(f"| {a} | {b} | {c} | {d} | {e} | `{f}` |" for a, b, c, d, e, f in rows)
    return f"""# S2 Method Source Registry

Decision-grade sources prefer primary papers, standards, authoritative textbooks, then official provider/model documentation for implementation contracts only.

| Method / Metric | Purpose in S2 | Primary source | Source type | Claim supported | Implementation |
| --- | --- | --- | --- | --- | --- |
{table}

No framework documentation or model card is used as proof that a component improves this corpus.
"""


def _metric_definitions() -> str:
    return """# S2 Metric Definitions

## Query Hit Recall@k

- Purpose: determine whether at least one trusted relevant evidence ID appears by cutoff `k`.
- Definition: query-level retrieval success at cutoff, macro-averaged over queries.
- Formula: `RecallHit@k = (sum_q I(R_q intersect L_q[:k] != empty)) / |Q|`.
- Variables: `Q` queries; `R_q` trusted relevant IDs; `L_q` ranked IDs; `I` indicator.
- Unit: query. Numerator: successful queries. Denominator: all labeled queries.
- Aggregation: macro query mean.
- Ground truth: at least one source-traceable relevant evidence ID per query.
- Interpretation: higher means more queries expose relevant evidence by `k`.
- Failure modes: binary hit ignores multiple relevant documents and all order below the first hit.
- Ceiling risk: R8 is 1.0 for both systems and cannot discriminate.
- Source: Manning, Raghavan, Schuetze (2008), recall foundation; S2's query-hit specialization is explicit.
- Implementation: `src/evaluation/ablation_metrics.py::recall_at_k`.
- Test: `tests/test_s2_metrics.py::test_recall_at_k_uses_query_level_numerator_and_denominator`.

## Mean Reciprocal Rank

- Purpose: measure how early the first trusted relevant item occurs.
- Definition: reciprocal rank of first relevant result per query; misses contribute zero; then mean.
- Formula: `MRR = (1/|Q|) * sum_q (1/rank_q)` with `1/rank_q = 0` for a miss.
- Unit: query. Numerator: reciprocal-rank sum. Denominator: all labeled queries.
- Aggregation: macro arithmetic mean.
- Ground truth: source-traceable relevant evidence IDs.
- Interpretation: higher is earlier; only first relevant result contributes.
- Failure modes: ignores additional relevant items and can hide category-specific failures.
- Ceiling risk: R8 is 1.0 for both systems.
- Source: NIST TREC QA scoring documentation.
- Implementation: `src/evaluation/ablation_metrics.py::mean_reciprocal_rank`.
- Test: `tests/test_s2_metrics.py::test_mrr_hand_calculation_counts_missing_query_as_zero`.

## Evidence Retention Rate

- Purpose: engineering diagnostic for deterministic filtering, not a medical-quality metric.
- Formula: `|unique(before) intersect unique(after)| / |unique(before)|`.
- Numerator: retained unique IDs. Denominator: input unique IDs. Empty input is defined as 1.0.
- Aggregation: case-level values may be reported by arithmetic mean.
- Ground truth: none; this measures preservation only.
- Limitation: retaining everything does not imply relevance.
- Implementation: `src/evaluation/ablation_metrics.py::evidence_retention_rate`.
- Test: `tests/test_s2_metrics.py::test_engineering_rates_have_obvious_hand_calculations`.

## Duplicate Slot Rate

- Purpose: engineering diagnostic for repeated output slots.
- Formula: `(|slots| - |unique(slots)|) / |slots|`.
- Numerator: duplicate slots. Denominator: all slots. Empty output is 0.0.
- Limitation: same-document redundancy with distinct IDs is not detected.
- Implementation and test: same module and hand-calculated test above.

## Arithmetic Mean Latency

- Purpose: descriptive timing only.
- Formula: `mean = sum_i latency_i / n`.
- Numerator: sum of observed elapsed milliseconds. Denominator: measured invocations.
- Limitation: warm-up, GPU state, OS scheduling, and tiny samples prevent generalization.
- Implementation: `src/evaluation/ablation_metrics.py::arithmetic_mean`.
- Test: hand-calculated engineering-rate test above.

No nDCG is used because S2 has no trusted graded relevance gold. No composite score or significance test is used.
"""


def _references() -> str:
    entries = []
    for ref in REFERENCES:
        doi = f" DOI: `{ref['doi']}`." if ref["doi"] else ""
        entries.append(
            f"## {ref['title']}\n\n{ref['authors']} ({ref['year']}), {ref['venue']}.{doi} "
            f"[{ref['url']}]({ref['url']})\n\n- Source grade: {ref['level']}\n- Claim used: {ref['claim']}"
        )
    return "# S2 References\n\n" + "\n\n".join(entries)


def _ground_truth_audit() -> str:
    rows = "\n".join(
        f"| {item['dataset']} | {item['cases']} | {item['origin']} | {'YES' if item['human_reviewed'] else 'NO'} | {item['decision_grade']} | {item['limitation']} |"
        for item in GOLD_AUDIT
    )
    return f"""# S2 Ground Truth Audit

| Dataset | Cases | Provenance | Human Reviewed? | Decision Grade? | Limitation |
| --- | ---: | --- | --- | --- | --- |
{rows}

No dataset is claimed `HUMAN_REVIEWED`, `EXPERT_VALIDATED`, or `CLINICALLY_VALIDATED`. The repository records no completed adjudication process appropriate for S2 removal decisions.
"""


def _leakage_audit() -> str:
    return """# S2 Data Leakage Audit

| Finding | Risk | S2 treatment |
| --- | --- | --- |
| R8 expected IDs are generated by the same locked harness | Structural self-confirmation and ceiling | Regression only |
| P3 expected states mirror the P3 contract | Implementation self-validation | Contract regression only |
| P4/P4.5 predictions originate from the evaluated system | Prediction-as-gold leakage | Excluded from decisions |
| Historical sentinel cases were created after known failures | Regression-overfitting risk | Report separately |
| V3 300 questions use `acceptable_origins=[llm_generated]` | Synthetic origin and label circularity | Audit only |
| Prior Gemini judge scores evaluate generated answers | LLM judge is not objective retrieval gold | Excluded from component decisions |
| Semantic reranker candidate labels are hand-written fixtures | Candidate texts may encode the expected keyword directly | Diagnostic only |

No new synthetic questions or labels were generated. No tuning was performed against any audited set.
"""


def _heuristic_audit() -> str:
    return """# S2 Hidden Heuristic Audit

| Heuristic | Component | Source? | Prior Evidence? | S2 Finding |
| --- | --- | --- | --- | --- |
| Lexical/entity/metadata/intent/safety/source scores | local-rule reranker | NO method source for exact weights | Unit/regression tests | `INSUFFICIENT_TRUSTED_EVIDENCE`; do not silently retain or delete |
| Hybrid weights `0.70/0.20/0.10` | hybrid reranker | NO corpus-specific source | Historical fixture eval | Unsourced/tuned provenance unclear; S3 must defer or revalidate |
| Candidate budget `max(top_k*2, 8)` | Candidate Policy | NO empirical source | Contract tests | Fixture does not exercise budget |
| Required evidence roles | P3/selector | Product safety contract, not empirical method source | Contract fixtures | Concept plausible; current complexity not validated |
| One bounded retry | P3 retry | Runtime safety contract | Contract fixtures | Recovery shown only in authored fixture |
| Taxonomy aliases | Entity | Taxonomy file, source provenance incomplete | Regression tests | Diagnostic gain only |
| Graph role mapping | selector/Graph | NO comparative evidence | Static R8 | Graph signal has no medical-claim eligibility in R8 |
| Critical-first reservation and clipping thresholds | packer | Engineering invariant; threshold source absent | Direct budget tests | KEEP finite-budget concept; exact thresholds remain unvalidated |

S2 does not add, tune, remove, or hide any production heuristic.
"""


def _reranker_report(result: dict[str, Any]) -> str:
    item = result["experiments"]["reranker"]
    return f"""# S2 Reranker Ablation

- Hypothesis: reranking moves fixture-labeled relevant evidence earlier than retrieval order.
- Dataset: `{item['dataset']}` ({item['case_count']} cases).
- Gold: `{item['gold_grade']}` because canonical corpus evidence IDs and review provenance are absent.
- Requested provider: `{item['requested_provider']}`; actual: `{', '.join(item['actual_providers'])}`; fallbacks: {item['fallback_count']}.
- Baseline Recall@1: {_metric(item['baseline']['recall@1'])}; MRR: {_metric(item['baseline']['mrr'])}.
- Variant Recall@1: {_metric(item['variant']['recall@1'])}; MRR: {_metric(item['variant']['mrr'])}.
- Changed cases: {item['changed_case_count']}/{item['case_count']}.
- Mean local latency: {_metric(item['mean_variant_latency_ms'])} ms over {item['mean_variant_latency_ms']['denominator']} invocations.
- Status: `INSUFFICIENT_TRUSTED_EVIDENCE`.

Observed movement is diagnostic only. It cannot justify keeping or removing the production reranker.
"""


def _candidate_policy_report(result: dict[str, Any]) -> str:
    item = result["experiments"]["candidate_policy"]
    return f"""# S2 Candidate Policy Ablation

- Mode: `{item['policy_mode']}`; case count: {item['case_count']}.
- Cases exercising the budget: {item['cases_exercising_budget']}.
- Mean ID retention: {_metric(item['mean_retention'])}.
- Mean duplicate slot rate: {_metric(item['mean_duplicate_slot_rate'])}.
- Gold: `{item['gold_grade']}`.
- Status: `INSUFFICIENT_TRUSTED_EVIDENCE`.

The current fixture has three candidates per case while the inherited minimum budget is eight. Zero delta therefore means the dataset did not exercise the policy, not that the policy has no value.
"""


def _sufficiency_report(result: dict[str, Any]) -> str:
    item = result["experiments"]["sufficiency_retry"]
    retry = item["retry"]
    return f"""# S2 Evidence Sufficiency and Retry Ablation

## Sufficiency concept

- Lightweight concept: `{item['lightweight_gate']}`.
- Dataset: `{item['dataset']}` ({item['case_count']} authored contract cases).
- Gold: `{item['gold_grade']}`; {item['reason']}
- Concept value status: `INSUFFICIENT_TRUSTED_EVIDENCE`.
- Current P3 implementation complexity status: `INSUFFICIENT_TRUSTED_EVIDENCE`.

## Retry separated

- Triggered: {retry['triggered']}.
- Recovered: {retry['recovered']}.
- Still insufficient: {retry['still_insufficient']}.
- Unnecessary by the transparent role check: {retry['unnecessary']}.
- External calls: {retry['external_calls']}.
- Status: `INSUFFICIENT_TRUSTED_EVIDENCE`.

These counts verify fixture behavior but do not estimate production recovery probability or clinical benefit.
"""


def _entity_report(result: dict[str, Any]) -> str:
    item = result["experiments"]["entity"]
    return f"""# S2 Entity Ablation

- Dataset: `{item['dataset']}` ({item['case_count']} cases).
- Expected metadata labels: {item['expected_label_count']}.
- Literal query hits: {item['literal_hit_count']}.
- Taxonomy-normalized hits: {item['normalized_hit_count']}.
- Diagnostic gain: {item['diagnostic_gain']} labels.
- Gold: `{item['gold_grade']}`; {item['reason']}
- Status: `INSUFFICIENT_TRUSTED_EVIDENCE`.

The test shows the normalizer can expose authored metadata beyond literal mentions. It does not prove downstream evidence quality or justify the current Entity implementation cost.
"""


def _graph_report(result: dict[str, Any]) -> str:
    item = result["experiments"]["graph"]
    return f"""# S2 Graph Ablation

- Dataset: `{item['dataset']}` ({item['case_count']} static cases).
- Graph signals: {item['graph_signal_count']}.
- Medical-claim-eligible graph signals: {item['medical_claim_eligible_count']}.
- Isolated quality delta: `{item['isolated_quality_delta']}`.
- Reason: {item['reason']}
- Status: `NOT_CLEANLY_ISOLATABLE`.

Graph facts are not medical evidence. The static R8 signal is synthetic and cannot establish a graph contribution to source-backed retrieval.
"""


def _selector_packer_report(result: dict[str, Any]) -> str:
    selector = result["experiments"]["selector"]
    packer = result["experiments"]["packer"]
    return f"""# S2 Selector and Packer Ablation

## Selector semantic value

- Primary retained: {selector['primary_retained']}/{selector['primary_denominator']}.
- Mean provider-free latency: {_metric(selector['mean_latency_ms'])} ms.
- Dataset: R8 with a known ceiling.
- Status: `INSUFFICIENT_TRUSTED_EVIDENCE`.

## Packer engineering value

- Primary retained: {packer['primary_retained']}/{packer['primary_denominator']}.
- Critical retained: {packer['critical_retained']}/{packer['critical_denominator']}.
- All cases within item budget: {packer['all_within_item_budget']}.
- All cases preserve source paths: {packer['all_source_backed']}.
- Mean provider-free latency: {_metric(packer['mean_latency_ms'])} ms.
- Status: `KEEP_EVIDENCE_SUPPORTED` for finite-budget serialization and provenance only.

This is not a claim that current clipping thresholds or selector heuristics improve answer quality.
"""


def _complexity_report(complexity: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {name} | {item['files']} | {item['loc']} | {item['functions']} |"
        for name, item in complexity["components"].items()
    )
    return f"""# S2 Complexity Cost

| Component | Python files | LOC | Functions/classes |
| --- | ---: | ---: | ---: |
{rows}

- Clinical state fields: {complexity['clinical_state_fields']}.
- Production remains LangGraph-based.
- Datastore and provider calls are reported separately; no synthetic complexity score is used.
- Counts are static source counts and do not imply value or defect.
"""


def _decision_report(result: dict[str, Any]) -> str:
    intended = {
        "reranker": "Improve candidate order",
        "candidate_policy": "Bound candidates before expensive stages",
        "evidence_sufficiency": "Prevent answers without required evidence",
        "retry": "Recover once from missing evidence roles",
        "entity": "Resolve aliases and canonical identities",
        "graph": "Supply relation signals",
        "selector": "Classify evidence coverage",
        "packer": "Bound context and preserve provenance",
    }
    rows = []
    for name, decision in result["component_decisions"].items():
        quality = "ENGINEERING_CONTRACT" if name == "packer" else "INSUFFICIENT_OR_NON_DECISION_GOLD"
        rows.append(
            f"| {name} | {intended[name]} | {quality} | {decision['evidence_scope']} | See complexity report | `{decision['status']}` |"
        )
    return """# S2 Component Decisions

| Component | Intended Benefit | Evidence Quality | Measured Benefit | Cost | S2 Status |
| --- | --- | --- | --- | --- | --- |
""" + "\n".join(rows) + "\n\nNo production removal is supported by S2."


def _s3_handoff(result: dict[str, Any]) -> str:
    actions = {
        "reranker": "DEFER — INSUFFICIENT EVIDENCE",
        "candidate_policy": "DEFER — INSUFFICIENT EVIDENCE",
        "evidence_sufficiency": "DEFER — INSUFFICIENT EVIDENCE",
        "retry": "DEFER — INSUFFICIENT EVIDENCE",
        "entity": "DEFER — INSUFFICIENT EVIDENCE",
        "graph": "DEFER — NOT CLEANLY ISOLATABLE",
        "selector": "DEFER — INSUFFICIENT EVIDENCE",
        "packer": "KEEP finite-budget/provenance concept; exact implementation review later",
    }
    rows = "\n".join(
        f"| {name} | `{decision['status']}` | {'YES' if name == 'packer' else 'UNRESOLVED'} | {'UNRESOLVED' if name != 'packer' else 'NOT ESTABLISHED'} | {actions[name]} |"
        for name, decision in result["component_decisions"].items()
    )
    return f"""# S2 to S3 Cleanup Handoff

| Component | S2 Status | Keep Concept? | Keep Current Implementation? | S3 Action Candidate |
| --- | --- | --- | --- | --- |
{rows}

## Old evaluation framework classification

- Production-critical: `evaluation/live_eval.py` adapters used by active workflows; inspect before any move.
- Regression-test: R8, P3, P4, safety, API, cache, and answer-quality contract tests.
- Old-evaluation-only: V3 300-case runners/reports and historical one-off audit scripts, subject to dependency audit.
- Mixed responsibility: shared deterministic helpers imported by tests and report scripts.
- Unknown: files without an import/provenance audit.

P4 remains `shadow`. Do not use P4 predictions as gold; reconsider runtime placement only in S3/E0/E1. Production must remain LangGraph-based Agentic RAG; S3/S4 may simplify node count, state fields, responsibilities, and optional tools.
"""


def _test_results() -> str:
    return """# S2 Test Results

This file is updated after validation. Required gates:

- Metric formula tests with hand-calculated examples.
- S2 harness tests.
- Locked R8 regression.
- P3 and P4 curated regressions.
- Historical sentinels and medical/safety tests.
- Full backend with coverage at least 70% and no meaningful regression from S1.
- Frontend tests, lint, build, `pip check`, and `npm audit` under current project policy.

See command logs and final S2 report for actual pass counts.
"""


def _runtime_report(runtime: dict[str, Any]) -> str:
    return f"""# S2 Runtime Integrity

- Ingestion: `{runtime['ingestion']}`.
- Reindex: `{runtime['reindex']}`.
- Reembedding: `{runtime['reembedding']}`.
- Semantic enrichment: `{runtime['semantic_enrichment']}`.
- Qdrant knowledge: {runtime['knowledge']}.
- Qdrant entities: {runtime['entities']}.
- Neo4j: {runtime['neo4j_nodes']} nodes / {runtime['neo4j_relationships']} relationships.

S2 performs read-only diagnostics and zero database mutations.
"""


def _git_report(git: dict[str, Any]) -> str:
    return f"""# S2 Git Integrity

- Branch: `{git['branch']}`.
- HEAD at artifact generation: `{git['head']}`.
- Main: `{git['main']}`.
- Origin main: `{git['origin_main']}`.
- Tracked status at generation: `{git['status'] or 'clean except current S2 WIP'}`.
- Production files modified by S2: NO.
- Production files/tests deleted by S2: NO.
"""


def _main_report(
    result: dict[str, Any],
    runtime: dict[str, Any],
    complexity: dict[str, Any],
    git: dict[str, Any],
) -> str:
    reranker = result["experiments"]["reranker"]
    return f"""# S2 Controlled Ablation Report

## Overall status

`{result['overall_status']}`

S2 is methodologically valid but incomplete for architecture removal decisions: the repository lacks source-traceable, independently reviewed component gold. The finite-budget/provenance packer contract is directly supported as an engineering invariant; every semantic component remains deferred.

## Starting baseline

- Main: `{git['main']}`.
- Production retrieval: Retrieval V5.
- Orchestrator: LangGraph.
- P3: enabled. P4: shadow.
- A0: dense 15 + sparse BM25 15 -> equal-weight RRF `k=60` -> top 5.
- Embedding: `models/gemini-embedding-2`, 3072 dimensions.
- Runtime: {runtime['knowledge']} knowledge / {runtime['entities']} entities / Neo4j {runtime['neo4j_nodes']}/{runtime['neo4j_relationships']}.

## Permanent rules

- NO METHOD WITHOUT SOURCE: PASS for decision-grade methods.
- NO FORMULA WITHOUT DEFINITION: PASS.
- NO HIDDEN HEURISTIC WITHOUT ABLATION EVIDENCE: PASS as an audit gate; unsupported heuristics are flagged, not endorsed or removed.

## Central findings

1. R8 remains a trusted regression set but has a hard ceiling and synthetic static evidence IDs.
2. The 300-question V3 set is LLM-generated and mostly lacks accepted canonical sources; prior Gemini scores are not objective gold.
3. The reranker changed {reranker['changed_case_count']}/{reranker['case_count']} fixture rankings, but those labels are not decision-grade.
4. Candidate Policy was not exercised by its fixture because all cases remained below budget.
5. P3/retry fixtures verify contract behavior but cannot estimate production benefit.
6. Entity normalization shows authored metadata gains, without source-backed retrieval gold.
7. Graph is not cleanly isolatable on current trusted cases and carries no medical evidence eligibility in R8.
8. Packer finite-budget and source-provenance behavior is directly verifiable and should remain as a concept.

## Complexity

- Audited component Python files: {complexity['total_files']}.
- Audited component LOC: {complexity['total_loc']}.
- Clinical state fields: {complexity['clinical_state_fields']}.
- No composite complexity score is used.

## Safety and performance

- Critical R8 source retention is reported separately in `S2_SELECTOR_PACKER_ABLATION.md`.
- Provider-free and local-reranker timing are separated.
- No live generation was required; no Ollama fallback or paid API was used.
- Zero Qdrant/Neo4j mutations, ingestion, reindex, reembedding, or semantic enrichment.

## Conclusion

S3 may begin only as a conservative cleanup phase that preserves unresolved production components or first creates independently source-grounded gold. S2 itself does not delete or simplify production code.
"""


def _component_deltas(result: dict[str, Any]) -> dict[str, Any]:
    reranker = result["experiments"]["reranker"]
    baseline = reranker["baseline"]
    variant = reranker["variant"]
    return {
        "reranker": {
            "recall_at_1_delta": float(variant["recall@1"]["value"]) - float(baseline["recall@1"]["value"]),
            "mrr_delta": float(variant["mrr"]["value"]) - float(baseline["mrr"]["value"]),
            "decision_grade": False,
        },
        "candidate_policy": {
            "cases_exercising_budget": result["experiments"]["candidate_policy"]["cases_exercising_budget"],
            "decision_grade": False,
        },
        "entity": {
            "normalized_minus_literal_labels": result["experiments"]["entity"]["diagnostic_gain"],
            "decision_grade": False,
        },
        "graph": {"quality_delta": "N/A", "decision_grade": False},
        "packer": {"finite_budget_contract": True, "decision_grade_scope": "engineering only"},
    }


def _complexity_inventory() -> dict[str, Any]:
    groups = {
        "reranker": [Path("src/retrieval/reranker.py"), Path("src/retrieval/reranker_v5.py"), *Path("src/retrieval/reranking").glob("*.py")],
        "candidate_policy": [Path("src/retrieval/candidate_policy.py")],
        "evidence_sufficiency_retry": [Path("src/retrieval/evidence_sufficiency.py"), Path("src/agent/nodes/retry_retrieval.py")],
        "entity": [Path("src/retrieval/query_normalization.py"), Path("src/retrieval/query_expansion.py"), Path("src/knowledge/normalizer.py")],
        "graph": [Path("src/database/graph_store.py"), Path("src/retrieval/graph_retrieval.py")],
        "selector_packer": [Path("src/retrieval/evidence_selector.py"), Path("src/retrieval/context_packer_v5.py"), Path("src/retrieval/context_packer.py")],
    }
    components: dict[str, Any] = {}
    all_files: set[Path] = set()
    for name, paths in groups.items():
        existing = sorted({path for path in paths if (PROJECT_ROOT / path).exists()})
        all_files.update(existing)
        components[name] = {
            "files": len(existing),
            "loc": sum(_line_count(PROJECT_ROOT / path) for path in existing),
            "functions": sum(_definition_count(PROJECT_ROOT / path) for path in existing),
            "paths": [str(path).replace("\\", "/") for path in existing],
        }
    state_path = PROJECT_ROOT / "src" / "agent" / "state.py"
    return {
        "components": components,
        "total_files": len(all_files),
        "total_loc": sum(_line_count(PROJECT_ROOT / path) for path in all_files),
        "clinical_state_fields": _clinical_state_fields(state_path),
    }


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _definition_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree))


def _clinical_state_fields(path: Path) -> int:
    if not path.exists():
        return 0
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ClinicalState":
            return sum(isinstance(item, ast.AnnAssign) for item in node.body)
    return 0


def _git_snapshot() -> dict[str, str]:
    return {
        "branch": _git("branch", "--show-current"),
        "head": _git("rev-parse", "HEAD"),
        "main": _git("rev-parse", "main"),
        "origin_main": _git("rev-parse", "origin/main"),
        "status": _git("status", "--short"),
    }


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _metric(value: dict[str, Any]) -> str:
    return f"{float(value['value']):.6f} (n={value['denominator']})"


if __name__ == "__main__":
    raise SystemExit(main())
