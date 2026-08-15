# Knowledge Preparation and Indexing

The canonical pipeline is implemented under `src/ingestion/` and exposed by
`scripts/knowledge_build.py` through the `build`, `validate`, and `status` commands. The
current validated build is recorded in `data/knowledge_build_manifest.json` as
`ec0a6de32d58ac181af6`: four sources, 512 knowledge chunks, 32 EntityCards, and
a Neo4j graph with 32 nodes and 27 relationships.

## NICE Source Provenance

The NICE-derived snapshot was acquired through a text-rendering transport. Its
represented date is 2026-08-03, while official NICE NG198 metadata reports
2026-04-30. A complete replacement from the official source was not available
through the supported acquisition routes. The snapshot remains part of the
project's research corpus, but the project does not claim that it is a fully
verified current official version.

This disclosure does not alter source identity, build identity, Qdrant, Neo4j,
or any indexed artifact. Replacing the snapshot requires a controlled knowledge
migration with new provenance and a new build identity.

```text
data/sources/manifest.yaml
  -> content-addressed parsing
  -> conservative normalization
  -> structure-aware chunks (2400 Unicode characters, no overlap)
  -> proof-based artifact filtering and exact deduplication
  -> portable provenance
  -> Gemini Embedding 2 Dense vectors + Qdrant-native BM25 documents
  -> versioned Qdrant knowledge/entity candidates
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
| Candidate indexing and cutover | `src/ingestion/index.py`, `pipeline.py` |
| Taxonomy, EntityCards, and graph | `src/knowledge/` |
| Layered validation and manifest | `src/ingestion/validation.py`, `manifest.py` |

Unchanged parsing and embedding outputs are reused through strict content and
contract identity. Structural output is deterministic. Provider vectors are
cached and validated, but floating-point provider output is not claimed to be
bitwise reproducible. LLM semantic graph extraction is not part of the current
pipeline.

The maintained sequence source is
[`docs/diagrams/knowledge-foundation.mmd`](diagrams/knowledge-foundation.mmd).
The diagram describes the current knowledge-build workflow.

## Build Contracts

- four-source registry with SHA-256 content identity;
- conservative Unicode normalization;
- structure-aware chunks capped at 2400 Unicode characters with zero overlap;
- proof-based filtering, exact deduplication, and complete provenance;
- Gemini Embedding 2, 3072 dimensions, cosine distance, and no task type;
- Qdrant-native BM25 using the parameters in
  [Methods and Formulas](METHODS_AND_FORMULAS.md);
- source-backed taxonomy, 32 EntityCards, and a deterministic 32/27 graph;
- logical aliases `acne_knowledge` and `acne_entities`.

## Change Control

Normal runtime development reuses the current indexed build. A new build is
appropriate for a confirmed implementation defect, corrupt or lost datastore,
materially updated authoritative source, provider incompatibility, or approved
research revision. Candidate indexes are built in parallel, native Qdrant
snapshots and a Neo4j cold backup are preserved, layered validation is required,
and alias cutover remains guarded.
