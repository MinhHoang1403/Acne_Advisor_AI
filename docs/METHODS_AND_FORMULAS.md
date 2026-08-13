# Methods and Formulas

## Dense Retrieval

Production chunks use `models/gemini-embedding-2`, dimension 3072 and cosine
distance. Provider task-type handling is centralized in the Google GenAI
integration and vector-store query path.

## Compatibility Sparse Representation

Canonical code: `src/ingestion/sparse_legacy.py`.

For token count `tf(t,d)` and maximum token count `max_tf(d)`, the stored value is:

```text
(1 + ln(tf(t,d))) / (1 + ln(max_tf(d)))
```

Tokens are lowercased and mapped to stable 31-bit indices from the first eight
hex digits of MD5. Hash collisions are summed. There is no IDF, document-length
normalization, `k1`, or `b`; therefore this is `CUSTOM_SPARSE_NOT_BM25`. The
Qdrant name `bm25` is a legacy storage contract until an explicit migration.

## Reciprocal Rank Fusion

Canonical code: `src/retrieval/rrf.py`.

Each ranked channel contributes a reciprocal term based on its rank and `k`.
The current caller default is `k=60`. Metadata remains an observation/tie-break
signal under Retrieval V5 and must not silently redefine source evidence.

## Hybrid Reranker

Current configured weights are semantic 0.70, rule 0.20 and retrieval 0.10.
These are project engineering parameters, not scientifically validated constants.

## Evidence Sufficiency

Canonical code: `src/retrieval/evidence_sufficiency.py`. P3 has a hard maximum
of two retrieval attempts. P4 claim grounding remains shadow-only.

## Status Labels

- `PROJECT_ENGINEERING_PARAMETER`: operational choice without scientific claim.
- `UNSOURCED_HEURISTIC`: deterministic behavior needing evidence review.
- `CUSTOM_SPARSE_NOT_BM25`: current compatibility formula.
- `S4A_REVIEW_REQUIRED`: Phase 1 methodology work deliberately deferred.
