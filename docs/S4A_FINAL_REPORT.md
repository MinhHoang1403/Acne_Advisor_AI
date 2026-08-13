# S4A Final Phase 1 Report

Research cutoff: 2026-08-14 inclusive. Status: Phase 1 frozen.

## Baseline And Outcome

| Item | Before | Frozen build |
|---|---:|---:|
| Canonical sources | 4 | 4 |
| Knowledge chunks | 639 | 512 |
| EntityCards | 32 | 32 |
| Neo4j nodes | 32 | 32 |
| Neo4j relationships | 27 | 27 |
| Sparse method | normalized log-TF hashing | Qdrant-native BM25 |

The count change is intentional: structural chunking and proof-only filtering
replaced historical splitting/noise heuristics. Reconciliation found no
unexplained source loss; five exact parser/TOC/page artifacts were filtered.
Short medical and safety statements remain eligible.

## Frozen Method

```text
canonical sources -> content-addressed parse -> NFC normalization
  -> structure-first chunking (2400 chars, no overlap)
  -> proof-only filtering + exact dedupe -> complete provenance
  -> Gemini Embedding 2 dense + Qdrant-native BM25
  -> immutable candidate collections -> logical aliases
  -> source-backed EntityCards -> deterministic Neo4j graph
  -> layered validation -> frozen build manifest
```

BM25 follows Robertson and Zaragoza's probabilistic relevance framework. The
provider implementation is Qdrant `qdrant/bm25`, collection modifier IDF,
`k=1.2`, `b=0.75`, average length 256, word tokenizer, language `none`,
lowercase, no ASCII folding, stemming, or stopword removal. Documents and
queries use the same constructor in `src/ingestion/bm25.py`.

Dense embeddings use the official Gemini Embedding 2 contract at 3072
dimensions with cosine distance and no task type. Dense retrieval carries the
cross-lingual semantic role; BM25 provides exact lexical evidence and is not
claimed to solve Vietnamese-English vocabulary mismatch alone.

## Data Contracts

- Source identity is curated, path-independent, and bound to SHA-256 content.
- Document/chunk IDs are deterministic and content-bound.
- Provenance completeness is 100% across all 512 chunks.
- Taxonomy entries and aliases have source references; unsupported ambiguous
  aliases were excluded.
- EntityCards organize canonical identity and query expansion but are not
  independent medical evidence.
- Graph relationships are deterministic and source-backed; graph facts alone
  cannot ground medical claims.
- Semantic enrichment is removed, so canonical Phase 1 has no Ollama or
  LLM-generated graph dependency.

## Reproducibility And Migration

Two compiles over the same parsed artifacts produced equal build IDs,
structural hashes, chunk IDs/content/provenance, taxonomy, graph, and BM25
inputs. The activated build is `ec0a6de32d58ac181af6`; embedding caches yielded
512 knowledge and 32 entity hits with zero provider calls during final cutover.

Before mutation, two readable Qdrant snapshots and a non-empty Neo4j cold
backup were stored under the ignored rollback root. Parallel candidates passed
offline, Qdrant, EntityCard, graph, Retrieval V5, P3, P4 shadow, safety, backend,
frontend, and readiness checks before alias activation. The historical entity
collection was deleted only after post-cutover checks passed; snapshots remain.

## Code Map

| Method | Source/contract | Code |
|---|---|---|
| Sources | clinical authority registry | `data/sources/manifest.yaml` |
| Parser/cache | LlamaParse/direct UTF-8 | `src/ingestion/parser.py` |
| Normalization | engineering invariant | `src/ingestion/normalization.py` |
| Chunking | empirical project parameter | `src/ingestion/chunking.py` |
| Filtering | proof-only invariant | `src/ingestion/filtering.py` |
| Provenance | engineering invariant | `src/ingestion/provenance.py` |
| Dense | official Gemini contract | `src/ingestion/embedding.py` |
| BM25 | Robertson-Zaragoza + Qdrant | `src/ingestion/bm25.py` |
| Candidate/index | migration invariant | `src/ingestion/index.py` |
| Manifest | reproducibility contract | `src/ingestion/manifest.py` |
| Validation | layered gates | `src/ingestion/validation.py` |
| Orchestration | canonical business logic | `src/ingestion/pipeline.py` |
| Operator CLI | thin adapter | `scripts/phase1.py` |

There is no known Phase 1 methodological debt as of the cutoff. Changes after
this point require the migration procedure in `docs/PHASE1_FROZEN.md`.
