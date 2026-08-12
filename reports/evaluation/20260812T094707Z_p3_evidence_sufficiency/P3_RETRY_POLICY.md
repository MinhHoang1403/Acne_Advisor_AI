# P3 Retry Policy

## Hard bound

- Total retrieval attempts: at most 2.
- Attempt 0: original V5 retrieval.
- Attempt 1: the only P3 retry.
- There is no recursive edge from attempt 1 to retry planning.
- Environment values above 2 are clamped to 2; values below 1 are clamped to 1.

## Eligibility

Retry is allowed only on attempt 0 when a required or critical source-backed role may plausibly be recovered by query representation refinement.

Retry is not allowed for:

- sufficient evidence;
- attempt 1;
- provider failure after existing V5 fallback handling;
- out-of-scope requests;
- source provenance failure;
- critical Packer overflow or other post-pack limitation.

## Strategy

The retry query deterministically combines the original standalone query with:

- missing Selector role hints;
- normalized entity IDs;
- existing EntitySignal aliases/canonical names;
- existing GraphSignal relation and target hints.

The strategy never injects an answer, changes a score, changes RRF weights, changes a model, or tunes channel depth. SHA-256 query hashes prove that attempt 1 differs from attempt 0.

## Isolation

Attempt 1 reruns the normal V5 pipeline and becomes the active result only as a fresh result. Scores from attempts 0 and 1 are never merged or sorted together. `retry_history` retains compact, separate selected/packed IDs, source IDs, statuses, and query hashes for both attempts.
