# Sơ Đồ Luồng Hoạt Động

Các sơ đồ ở đây phản ánh kiến trúc đang hoạt động sau S4A. Phase 1 chỉ có một
giao diện vận hành là `scripts/phase1.py`; semantic graph extraction bằng LLM
không thuộc nền tảng chuẩn.

## Phase 1 Frozen Knowledge Foundation

Nguồn Mermaid: [phase1-frozen-foundation.mmd](diagrams/phase1-frozen-foundation.mmd).

```mermaid
sequenceDiagram
    autonumber
    actor Operator
    participant CLI as scripts/phase1.py
    participant Sources as Canonical source manifest
    participant Build as src/ingestion
    participant Gemini as Gemini Embedding 2
    participant Qdrant
    participant Neo4j

    Operator->>CLI: phase1.py build
    CLI->>Sources: Verify portable IDs and SHA-256 hashes
    Sources-->>Build: Four canonical source records
    Build->>Build: Reuse content-addressed parsed artifacts
    Build->>Build: NFC normalize, structure-first chunk, proof-only filter
    Build->>Build: Attach complete provenance and deterministic IDs
    Build->>Gemini: Embed cache misses only (3072 dimensions)
    Gemini-->>Build: Dense vectors
    Build->>Qdrant: Candidate chunks: dense + native BM25 Document inputs
    Build->>Qdrant: Candidate source-backed EntityCards
    Build->>Build: Compile deterministic taxonomy graph
    Build->>CLI: Candidate manifest and layered validation
    CLI-->>Operator: Candidate build ID and validation result

    opt Explicit guarded activation
        Operator->>CLI: phase1.py build --activate --rollback-root ...
        CLI->>CLI: Verify Qdrant snapshots and Neo4j cold backup
        CLI->>Qdrant: Switch logical aliases to validated physical build
        CLI->>Neo4j: Reconcile deterministic source-backed graph
        CLI->>CLI: Run post-cutover validation
        CLI-->>Operator: Activated build manifest
    end
```

## Phase 2 Online Chat/RAG

Phase 2 consumes the frozen contracts. It may orchestrate dense retrieval,
Qdrant-native BM25, EntityCards, and Neo4j facts, but it does not rebuild them.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant API as FastAPI
    participant Agent as LangGraph
    participant Cache as Redis
    participant Retriever as Retrieval V5
    participant Qdrant
    participant Neo4j
    participant LLM as Gemini/Ollama

    User->>API: POST /chat
    API->>Agent: Validated state and history
    Agent->>Agent: Guardrail and query preparation
    Agent->>Cache: Versioned semantic cache lookup
    alt Valid cache hit
        Cache-->>Agent: Cached grounded answer
    else Cache miss
        Agent->>Retriever: Retrieve evidence
        Retriever->>Qdrant: Dense query
        Retriever->>Qdrant: Native BM25 Document query
        Retriever->>Neo4j: Optional deterministic structural facts
        Retriever-->>Agent: Ranked source chunks and structural context
        Agent->>Agent: Evidence sufficiency and grounding controls
        Agent->>LLM: Source-grounded prompt
        LLM-->>Agent: Draft answer
        Agent->>Agent: Safety, formatting, and fallback policies
        Agent->>Cache: Store only eligible answer
    end
    Agent-->>API: Answer, source metadata, trace metadata
    API-->>User: UTF-8 JSON response
```

## Render

```powershell
& "$env:APPDATA\npm\mmdc.cmd" -i .\docs\diagrams\phase1-frozen-foundation.mmd -o .\docs\diagrams\phase1-frozen-foundation.svg -p .\docs\puppeteer-config.json
```
