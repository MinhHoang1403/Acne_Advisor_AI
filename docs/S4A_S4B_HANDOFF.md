# S4A to S4B Handoff

S4A completed the Phase 1 knowledge foundation. S4B may improve Retrieval V5
or LangGraph orchestration around this foundation, but may not rebuild or
redesign ingestion.

## Immutable Inputs

- Build manifest: `data/phase1_build_manifest.json`
- Source manifest: `data/sources/manifest.yaml`
- Taxonomy: `data/taxonomy/drug_aliases.yaml`
- Knowledge alias: `acne_knowledge` (512 chunks)
- Entity alias: `acne_entities` (32 EntityCards)
- Neo4j: 32 deterministic nodes / 27 source-backed relationships
- Dense: `dense`, `models/gemini-embedding-2`, 3072, cosine
- Lexical: `bm25`, Qdrant-native BM25 `Document` inference with IDF
- Provenance: every knowledge chunk has portable source/document/record/chunk
  identity, content hashes, locator, section, source URL where available, and
  build ID

## Evidence Boundary

Only canonical knowledge chunks independently ground medical claims.
EntityCards and Neo4j facts are structural aids; neither may independently
ground a medical answer. S4B must preserve this distinction in evidence
selection, traces, citations, retries, and abstention.

## S4B Allowed Scope

S4B may change query understanding, retrieval orchestration, fusion,
reranking, context packing, LangGraph state/routing/tools, answer generation,
and observability. It must consume logical aliases and must not depend on the
physical build suffix.

S4B must not change sources, parser, normalization, chunking, filtering,
provenance schema, embedding dimensions/model, BM25 contract, taxonomy,
EntityCard construction, graph construction, or Phase 1 datastore schema.
