# Frozen Phase 1 Data Pipeline

The canonical path is implemented under `src/ingestion/` and exposed only by
`scripts/phase1.py` (`build`, `validate`, `status`).

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
