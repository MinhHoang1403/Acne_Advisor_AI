# Architecture

Acne Advisor AI has two bounded layers:

1. Frozen Phase 1 builds a provenance-complete medical knowledge foundation.
2. Phase 2 runs the FastAPI/LangGraph Retrieval V5 application over that
   foundation.

| Responsibility | Canonical location |
|---|---|
| Phase 1 methods and orchestration | `src/ingestion/` |
| Single Phase 1 operator CLI | `scripts/phase1.py` |
| Taxonomy, EntityCards, graph | `src/knowledge/` |
| Qdrant, Neo4j, PostgreSQL adapters | `src/database/` |
| Retrieval V5 | `src/retrieval/` |
| LangGraph and generation | `src/agent/` |
| Safety/answer verification | `src/quality/` |
| Cache and resilience | `src/cache/`, `src/resilience/` |
| API and frontend | `src/api/`, `src/frontend/` |

Qdrant logical aliases are `acne_knowledge` and `acne_entities`; immutable
physical collections include the content-derived build ID. Knowledge points
contain named `dense` (3072/cosine) and native `bm25` vectors. Neo4j is a
deterministic taxonomy graph with relationship provenance. Entity/graph data
cannot independently ground medical claims.

S4B may change retrieval/agent orchestration but must not redesign frozen Phase
1 contracts without an explicit migration proposal.
