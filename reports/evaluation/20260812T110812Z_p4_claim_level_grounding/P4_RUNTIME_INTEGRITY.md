# P4 Runtime Integrity

Read-only post-validation checks:

| Component | Expected | Observed | Result |
| --- | --- | --- | --- |
| Qdrant `acne_knowledge` | 639, dense 3072, sparse bm25 | 639, dense 3072, sparse bm25 | PASS |
| Qdrant `acne_entities_v1` | 32 | 32 | PASS |
| Neo4j nodes | 32 | 32 | PASS |
| Neo4j relationships | 27 | 27 | PASS |
| Core Phase 1 | completed_validated | completed_validated | PASS |
| Semantic enrichment | not_run | not_run | PASS |

Ingestion: NO. Reindex: NO. Reembedding: NO. Taxonomy mutation: NO. Semantic enrichment: NO. Database/collection reset: NO.
