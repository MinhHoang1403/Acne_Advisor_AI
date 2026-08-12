# P3 Sufficiency Contract

## Position

P3 runs after one complete Retrieval V5 result. It consumes the existing Evidence Selector and Context Packer V2 outputs; it does not alter Dense, Sparse, RRF, Candidate Policy, reranking, selection, or packing.

## Typed state

`EvidenceSufficiencyAssessment` exposes:

- `status`: `SUFFICIENT`, `INSUFFICIENT`, or `CRITICAL_EVIDENCE_MISSING`.
- `stage`: `PRE_PACK` or `POST_PACK`.
- required, satisfied, missing, and critical-missing roles.
- source-backed evidence IDs and source IDs.
- deterministic reason codes and retry eligibility.
- bounded attempt index, trace ID, and evaluator latency.

## Deterministic rules

1. Every required role must be covered by a selected source-backed chunk.
2. Valid provenance requires `chunk_id` plus `source_path` or `document_id`.
3. At least one valid evidence ID and source ID must remain.
4. A critical query requires explicitly selected `critical` evidence, not merely a structural safety signal.
5. EntitySignalV5 and GraphSignalV5 may derive requirements and retry hints but never satisfy evidence requirements.
6. PRE_PACK validates Selector coverage. POST_PACK validates only IDs that survived Packer serialization.
7. `CRITICAL_EVIDENCE_OVERFLOW` is a non-retryable prompt-budget failure.
8. Provider failure, out-of-scope input, and untraceable evidence are non-retryable.

## Compatibility

P3 is active only when `RETRIEVAL_PIPELINE_VERSION=v5` and `P3_EVIDENCE_SUFFICIENCY_ENABLED=true`. Explicit V4 rollback bypasses P3 and retains the released V4 fallback behavior.
