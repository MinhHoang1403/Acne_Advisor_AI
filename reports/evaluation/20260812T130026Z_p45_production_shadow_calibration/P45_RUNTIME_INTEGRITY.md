# P4.5 Runtime Integrity

Post-generation read-only validation:

| Component | State |
| --- | --- |
| Qdrant `acne_knowledge` | 639 points, dense 3072 + sparse BM25 |
| Qdrant `acne_entities_v1` | 32 points, dense 3072 + sparse BM25 |
| Neo4j | 32 nodes / 27 relationships |
| Ingestion manifest | 4 completed records, 639 point IDs |
| Core Phase 1 | `completed_validated` |
| Semantic enrichment | `not_run` |
| Retrieval pipeline | `v5` |
| P4 requested/effective mode | `shadow` / `shadow` |
| Critical/all-claim enforcement readiness | `false` / `false` |
| Cache answer version | `v5` |
| Pipeline fingerprint | `af95aafe84645cf9a7987b91` |

No ingestion, reindex, re-embedding, taxonomy mutation, semantic enrichment, cache flush, or database reset occurred.
