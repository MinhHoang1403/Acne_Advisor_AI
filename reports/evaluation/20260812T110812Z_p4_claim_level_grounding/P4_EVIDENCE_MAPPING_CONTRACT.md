# P4 Evidence Mapping Contract

Primary evidence scope is `GENERATION_CONTEXT_EVIDENCE`: only `chunk` items in the actual `PackedContext` supplied to generation are eligible. A valid link requires `chunk_id` and either `source_path` or `document_id`.

`EntitySignal` and `GraphSignal` may help normalization upstream but are never accepted as standalone medical evidence. Entity items and invalid-provenance chunks are rejected by the mapper.

Mapping priority is direct source/citation marker, normalized entity/alias overlap, then lexical overlap. At most three evidence items map to one claim. Existing taxonomy-aware query normalization supports aliases; no new embedding model or corpus embedding is used.

`ClaimEvidenceLink` records claim/evidence/source IDs, generation scope, structural mapping reason, lexical overlap, entity overlap, provenance validity, and rank. `semantic_score` remains separate and unset. Retrieval RRF, reranker, lexical mapping, and entailment confidence are never merged.

A claim without plausible valid packed evidence receives `NO_EVIDENCE`; unrelated chunks are not forced into the verifier.
