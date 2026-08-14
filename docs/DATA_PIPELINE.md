# Frozen Phase 1 Data Pipeline

The canonical path is implemented under `src/ingestion/` and exposed only by
`scripts/phase1.py` (`build`, `validate`, `status`). `PHASE1_FROZEN = true`.
The machine record is `data/phase1_build_manifest.json`, build
`ec0a6de32d58ac181af6`: four sources, 512 knowledge chunks, 32 EntityCards, and
a Neo4j graph with 32 nodes and 27 relationships.

```text
data/sources/manifest.yaml
  -> content-addressed parse
  -> conservative normalization
  -> structure-first chunks (2400 Unicode chars, no overlap)
  -> proof-only artifact filtering and exact deduplication
  -> portable provenance
  -> Gemini Embedding 2 dense vectors + Qdrant-native BM25 documents
  -> immutable Qdrant knowledge/entity candidates
  -> deterministic source-backed Neo4j graph
  -> layered validation and content-derived build manifest
  -> rollback-guarded alias cutover
```

| Responsibility | Code |
|---|---|
| Source identity and verification | `src/ingestion/source_manifest.py` |
| Parsed artifact cache | `src/ingestion/parser.py` |
| Normalization | `src/ingestion/normalization.py` |
| Chunking | `src/ingestion/chunking.py` |
| Filtering | `src/ingestion/filtering.py` |
| Provenance identities | `src/ingestion/provenance.py` |
| Dense embedding/cache | `src/ingestion/embedding.py` |
| Native BM25 contract | `src/ingestion/bm25.py` |
| Candidate indexing/cutover | `src/ingestion/index.py`, `pipeline.py` |
| Taxonomy, EntityCards, graph | `src/knowledge/` |
| Layered validation/manifest | `src/ingestion/validation.py`, `manifest.py` |

Unchanged source parsing and embedding are reused by strict content/contract
identity. Structural output is deterministic; provider vectors are cached and
validated but floating-point provider output is not claimed to be bitwise
reproducible. LLM semantic graph extraction is removed from the canonical path.

The maintained sequence source is
[`docs/diagrams/phase1-frozen-foundation.mmd`](diagrams/phase1-frozen-foundation.mmd).

## Frozen Contracts

- four-source registry with SHA-256 content identity;
- conservative Unicode normalization;
- structure-first chunks capped at 2400 Unicode characters, zero overlap;
- proof-only filtering, exact deduplication and complete provenance;
- Gemini Embedding 2, 3072 dimensions, cosine distance, no task type;
- Qdrant-native BM25 with the parameters in
  [Methods and Formulas](METHODS_AND_FORMULAS.md);
- source-backed taxonomy, 32 EntityCards and deterministic 32/27 graph;
- logical aliases `acne_knowledge` and `acne_entities`.

## Change Control

Do not rebuild this foundation to tune a runtime benchmark. A new explicit
migration is required for a confirmed implementation defect, corrupt/lost
datastore, materially updated authoritative source, provider incompatibility,
or approved research revision. Build candidates in parallel, preserve native
Qdrant snapshots and a Neo4j cold backup, pass layered validation, then perform
the guarded alias cutover. Git history records prior implementation experiments;
they are not active alternatives.
