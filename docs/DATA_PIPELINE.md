# Data Pipeline

## Canonical Build

The supported full build entrypoint is `scripts/run_full_phase1.py`. The
lower-level `scripts/ingest_knowledge.py` remains the incremental ingestion CLI.
Both operate on the canonical `sample_data/` source directory by default.

Current responsibilities:

| Stage | Location |
|---|---|
| Source discovery and orchestration | `scripts/ingest_knowledge.py` |
| JSON source loading | `src/ingestion/json_loader.py` |
| Markdown cleanup | `src/ingestion/cleanup.py` |
| Fixed-width fallback splitting | `src/ingestion/chunking.py` |
| Noisy-chunk heuristics | `src/ingestion/filtering.py` |
| Domain metadata | `src/ingestion/domain_metadata.py` |
| Dense embedding integration | `src/integrations/google_genai.py` |
| Compatibility sparse vectors | `src/ingestion/sparse_legacy.py` |
| Taxonomy/entity/graph build | `src/knowledge/` |

The ingestion manifest records source hashes and completion state so unchanged
documents can be skipped. Core Phase 1 and optional semantic graph enrichment
are separate operations; the current production baseline records semantic
enrichment as `not_run`.

## Method Status

- Chunk size, overlap and noisy-chunk rules are existing engineering choices,
  not literature-validated parameters (`S4A_REVIEW_REQUIRED`).
- Sparse output is `CUSTOM_SPARSE_NOT_BM25`; the datastore key `bm25` is retained
  solely for compatibility.
- S3B changes module ownership only. It does not rebuild or migrate data.
