# Frozen Phase 1 Knowledge Foundation

`PHASE1_FROZEN = true` as of 2026-08-14.

The authoritative machine record is `data/phase1_build_manifest.json`, build
`ec0a6de32d58ac181af6`. It binds the source manifest, taxonomy, parser,
normalization, chunking, filtering, provenance, embedding, BM25, EntityCard,
and graph contracts to the activated physical Qdrant collections.

## Frozen Contracts

- Four-source canonical corpus and its SHA-256 hashes.
- Content-addressed parser artifacts and conservative NFC normalization.
- Structure-first chunks capped at 2400 Unicode characters with no overlap.
- Proof-only artifact filtering and exact deduplication.
- Portable source, document, record, and chunk provenance.
- Gemini Embedding 2, 3072 dimensions, cosine distance, no `task_type`.
- Qdrant-native BM25 with collection IDF, `k=1.2`, `b=0.75`, average length
  256, word tokenizer, lowercase, language `none`, no stemming or stopwords.
- Source-backed taxonomy, 32 narrow EntityCards, and deterministic Neo4j graph
  with 32 nodes and 27 relationships.
- Logical Qdrant aliases `acne_knowledge` and `acne_entities`.
- Canonical operator interface `scripts/phase1.py` with `build`, `validate`,
  and `status` commands.

## Change Control

A later phase must not alter these contracts to improve a benchmark or try a
new technology. A new, explicit Phase 1 migration proposal is required for a
confirmed implementation defect, corrupt/lost datastore, materially updated
authoritative medical source, provider incompatibility/deprecation, or an
approved research revision. Every migration must preserve rollback evidence,
build a parallel candidate, and pass the layered validation contract before
cutover.

Semantic enrichment is resolved as **removed** from the canonical workflow.
No LLM-generated graph fact is part of the frozen foundation.
