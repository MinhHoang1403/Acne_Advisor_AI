# Ma Tran Danh Gia Toan Dien

Tai lieu nay dinh nghia pham vi cua benchmark canonical `acne_rag_eval_comprehensive_v1.jsonl`. Khong co dong nao duoc ghi nhan la "da danh gia" neu chua co cach kiem tra ro rang o cot component test hoac runtime check.

| Thanh phan | Cach kiem tra | Dataset 300 cau | Component test | Runtime check | Metric | Report section |
|---|---|---:|---:|---:|---|---|
| PDF/JSON ingestion, manifest va traceability | Read-only snapshot payload Qdrant va manifest | Co | `test_phase1_ingest_eval.py` | `validate_phase1_complete.py` | source traceability validity | 5 |
| Chunking, embeddings dense/sparse va vector dimension | Kiem tra collection/config, khong upsert | Gian tiep | `test_hybrid_retriever.py` | `validate_kb_collections.py` | vector snapshot | 5 |
| Taxonomy, aliases va entity cards | Ground truth entity tu taxonomy v2 | Co | `test_taxonomy_*.py` | `eval_taxonomy_expansion.py` | entity hit, alias accuracy | 5 |
| Neo4j graph va entity relations | Case entity/relationship, snapshot read-only | Co | `test_graph_index.py` | `validate_neo4j_schema.py` | entity relation coverage | 5 |
| Query normalization va conversation rewrite | Cac case typo, phu dinh va multi-turn | Co | `test_query_*.py` | `/chat` live | multi-turn context accuracy | 6 |
| Domain guardrail va severity | OOD, emergency, pregnancy, mild adverse | Co | `test_severity_guard.py` | `eval_severity_aware_guard.py` | route/safety metrics | 7 |
| Semantic cache va fingerprint | Cache bypass va metadata provenance | Co | `test_cache_versioning.py` | `inspect_cache_versions.py` | cache bypass rate | 8 |
| Hybrid/entity retrieval, graph enrichment, candidate merge | Accepted document-level source va entity truth | Co | `test_phase2_retrieval_eval.py` | `eval_phase2_retrieval.py` | source hit, entity hit | 5 |
| Semantic reranker va context packing | Retention/traces khi runtime cung cap | Gian tiep | `test_reranker*.py` | `eval_phase2_reranking.py` | reranker retention or N/A | 5 |
| Provider routing, answer generation va verifier | Requested/actual provider metadata | Co | `test_provider*.py`, `test_answer_verifier.py` | live `/chat` | provenance, non-empty answer | 6 |
| Safe fallback, timeout, retry, circuit breaker | Route-aware deterministic/judge va fault tests | Co | `test_safe_fallback_flow.py`, resilience tests | `eval_runtime_resilience.py` | fallback appropriateness, retry/timeout | 8 |
| Emergency, pregnancy, antibiotic stewardship | Critical cases va hard gates | Co | guard/safety tests | live `/chat` | critical safety recall | 7 |
| Exact format va presentation | Table, bullet, exact count contracts | Co | presentation tests | live `/chat` | format/instruction pass | 6 |
| API schema, health va persistence | FastAPI contract va dedicated eval sessions | Co | API tests | `/health`, `/chat` | success/error/provenance | 8 |
| Frontend va release readiness | Unit/build/lint/audit, offline readiness | Khong ap dung | frontend tests | `check_release_readiness.py` | runtime readiness | 4, 8 |

## Nguyen tac dien giai

- Dataset 300 cau chi danh gia end-to-end va cac metric co ground truth trong dataset; no khong thay the test don vi, readiness hay audit data foundation.
- Metric retrieval chi duoc tinh tren case co `accepted_sources` hoac `expected_entities`. `nDCG@k` duoc bao la khong ap dung khi dataset khong co graded relevance.
- Cac snapshot Qdrant, Neo4j va manifest la read-only. Ma tran nay khong yeu cau ingestion, rebuild hay ghi vao database.
