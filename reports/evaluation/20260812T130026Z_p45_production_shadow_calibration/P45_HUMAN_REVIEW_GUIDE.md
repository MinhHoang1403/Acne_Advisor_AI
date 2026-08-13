# P4.5 Human Evidence-Support Review Guide

This is a **HUMAN EVIDENCE-SUPPORT REVIEW**, not clinical validation unless a qualified clinician performs it.

Review `p45_human_labels.jsonl`. Grade each claim only from the provided evidence, not from outside medical knowledge. The blind file intentionally hides P4 predictions.

## Verdicts

- `SUPPORTED`: evidence supports the claim as written.
- `PARTIALLY_SUPPORTED`: evidence supports only a weaker or partial version.
- `UNSUPPORTED`: supplied evidence does not establish the claim.
- `CONTRADICTED`: supplied evidence is inconsistent with the claim.
- `NO_EVIDENCE`: no appropriate source-backed evidence was supplied.
- `REVIEW_UNCERTAIN`: evidence is genuinely ambiguous; do not guess.

## Claim rows

For every `CLAIM_REVIEW` row, fill `human_verdict`, `human_criticality`, `mapping_correct`, `claim_extraction_correct`, `reviewer_confidence`, `reviewer_notes`, `reviewer_id`, and `reviewer_qualification`. Leave P4 fields hidden in this file.

## Case rows

For every `CASE_REVIEW` row, record omitted medically meaningful claims in `missing_claims`, omitted critical claims in `missing_critical_claims`, bad splitting in `incorrectly_split`, and non-claim fragments in `unrelated_fragments`.

Do not alter case IDs, claim IDs, questions, answers, evidence, provenance, or dataset hashes. Preserve uncertain labels for later adjudication. A second reviewer must not be fabricated.
