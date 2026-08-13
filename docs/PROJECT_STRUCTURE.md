# Project Structure

- `src/ingestion/`: frozen Phase 1 parsing, normalization, chunking, filtering,
  provenance, dense/BM25 indexing, manifest, validation and orchestration.
- `src/knowledge/`: canonical taxonomy, EntityCards and deterministic Neo4j graph.
- `scripts/phase1.py`: the only supported Phase 1 operator entrypoint.
- `data/sources/manifest.yaml`: portable canonical medical source registry.
- `data/taxonomy/drug_aliases.yaml`: active source-backed taxonomy.
- `data/phase1_method_sources.json`: machine-readable method/source registry.
- `src/database/`: Qdrant, Neo4j and PostgreSQL runtime adapters.
- `src/retrieval/`: Retrieval V5 contracts and evidence selection.
- `src/agent/`: LangGraph workflow, prompts, generation and presentation.
- `src/quality/`: deterministic medical safety and grounding checks.
- `src/api/`: FastAPI application and preflight.
- `src/frontend/`: maintained React/Vite client.
- `tests/`: retained production and methodological regression contracts.
- `docs/REFERENCES.md`: exact technical/medical claims and sources.

Runtime databases, licensed source files, parsed/embedding caches, local build
manifest, reports and rollback snapshots are intentionally ignored. Git tracks
the source registry, taxonomy and reproducible code contracts, never secrets.
