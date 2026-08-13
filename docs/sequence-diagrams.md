# Sơ Đồ Luồng Hoạt Động

Tài liệu này chứa bản preview Mermaid cho hai luồng chính của Acne Advisor AI. Bản được ưu tiên là **sơ đồ gộp đầy đủ** để xem tổng quan toàn hệ thống. Các sơ đồ tách nhỏ vẫn được giữ ở cuối tài liệu như bản phụ để xem từng đoạn nhỏ hơn.

## Bản Gộp Full Ưu Tiên

| File | Nội dung |
|---|---|
| [docs/diagrams/phase1-full-ingestion.mmd](diagrams/phase1-full-ingestion.mmd) | Toàn bộ Phase 1 từ PDF/DOCX đến Qdrant/Neo4j |
| [docs/diagrams/phase2-full-chat-rag.mmd](diagrams/phase2-full-chat-rag.mmd) | Toàn bộ Phase 2 từ User chat đến response và lưu history |

Export SVG bản full:

```powershell
& "$env:APPDATA\npm\mmdc.cmd" -i .\docs\diagrams\phase1-full-ingestion.mmd -o .\docs\diagrams\phase1-full-ingestion.svg -p .\docs\puppeteer-config.json
& "$env:APPDATA\npm\mmdc.cmd" -i .\docs\diagrams\phase2-full-chat-rag.mmd -o .\docs\diagrams\phase2-full-chat-rag.svg -p .\docs\puppeteer-config.json
```

## Sơ Đồ Phụ/Tách Nhỏ

| File | Nội dung |
|---|---|
| [phase1a-parse-cache.mmd](diagrams/phase1a-parse-cache.mmd) | Discover PDF/DOCX, LlamaParse, Markdown cache, chunking |
| [phase1b-graph-neo4j.mmd](diagrams/phase1b-graph-neo4j.mmd) | Graph cache, Ollama graph extraction, Neo4j upsert |
| [phase1c-qdrant-upsert.mmd](diagrams/phase1c-qdrant-upsert.mmd) | Gemini embedding, custom sparse TF, Qdrant schema/upsert |
| [phase2a-guardrail.mmd](diagrams/phase2a-guardrail.mmd) | `/chat`, validation, history loading, guardrails |
| [phase2b-retrieval.mmd](diagrams/phase2b-retrieval.mmd) | Redis cache, Qdrant dense/sparse retrieval, Neo4j enrichment |
| [phase2c-answer-history.mmd](diagrams/phase2c-answer-history.mmd) | LLM generation/fallback, formatter, cache store, chat persistence |

## Export SVG Sơ Đồ Phụ

Ví dụ cho một file:

```powershell
& "$env:APPDATA\npm\mmdc.cmd" -i .\docs\diagrams\phase1a-parse-cache.mmd -o .\docs\diagrams\phase1a-parse-cache.svg -p .\docs\puppeteer-config.json
```

Export toàn bộ:

```powershell
& "$env:APPDATA\npm\mmdc.cmd" -i .\docs\diagrams\phase1a-parse-cache.mmd -o .\docs\diagrams\phase1a-parse-cache.svg -p .\docs\puppeteer-config.json
& "$env:APPDATA\npm\mmdc.cmd" -i .\docs\diagrams\phase1b-graph-neo4j.mmd -o .\docs\diagrams\phase1b-graph-neo4j.svg -p .\docs\puppeteer-config.json
& "$env:APPDATA\npm\mmdc.cmd" -i .\docs\diagrams\phase1c-qdrant-upsert.mmd -o .\docs\diagrams\phase1c-qdrant-upsert.svg -p .\docs\puppeteer-config.json
& "$env:APPDATA\npm\mmdc.cmd" -i .\docs\diagrams\phase2a-guardrail.mmd -o .\docs\diagrams\phase2a-guardrail.svg -p .\docs\puppeteer-config.json
& "$env:APPDATA\npm\mmdc.cmd" -i .\docs\diagrams\phase2b-retrieval.mmd -o .\docs\diagrams\phase2b-retrieval.svg -p .\docs\puppeteer-config.json
& "$env:APPDATA\npm\mmdc.cmd" -i .\docs\diagrams\phase2c-answer-history.mmd -o .\docs\diagrams\phase2c-answer-history.svg -p .\docs\puppeteer-config.json
```

## Diagram 1: Phase 1 Offline Ingestion

```mermaid
sequenceDiagram
    autonumber
    actor Operator
    participant CLI as ingest_knowledge.py
    participant Source as Source Docs
    participant LlamaParse
    participant MdCache as Markdown Cache
    participant Chunker
    participant GraphCache as Graph Cache
    participant Ollama
    participant Neo4j
    participant Gemini as Gemini Embedding
    participant Qdrant

    Operator->>CLI: Chạy ingestion command
    CLI->>CLI: Load .env và CLI flags
    CLI->>Source: Discover PDF/DOCX

    alt Không có PDF/DOCX
        CLI-->>Operator: Báo lỗi preflight và dừng
    else Có tài liệu nguồn
        CLI->>CLI: Preflight source, LlamaParse key, Ollama, Qdrant, Neo4j, Gemini key
    end

    loop Mỗi tài liệu nguồn
        CLI->>MdCache: Kiểm tra Markdown cache theo file fingerprint
        alt Markdown cache hit và không --refresh-markdown
            MdCache-->>CLI: Trả Markdown đã cache
        else Cache miss hoặc refresh
            CLI->>LlamaParse: Parse PDF/DOCX thành Markdown
            LlamaParse-->>CLI: Markdown
            CLI->>MdCache: Ghi Markdown cache an toàn
        end

        CLI->>Chunker: Chunk Markdown theo header và CHUNK_SIZE
        Chunker-->>CLI: SemanticChunk list với metadata, chunk_id, content_hash
    end

    opt --limit-files hoặc --limit-chunks
        CLI->>CLI: Giới hạn số tài liệu/chunks để test nhanh
    end

    loop Theo graph extraction batches
        CLI->>GraphCache: Kiểm tra graph cache key(version, prompt schema, model, source, index, chunk hash)
        alt Graph cache hit hợp lệ
            GraphCache-->>CLI: Nodes và edges đã normalize
        else Cache miss/invalid hoặc --refresh-graph-cache
            alt --skip-graph-extraction
                CLI-->>CLI: Tạo extraction_error payload, không gọi Ollama
            else Gọi Ollama
                CLI->>Ollama: Prompt theo OLLAMA_MODEL trích xuất graph JSON
                Ollama-->>CLI: Raw JSON/text
                CLI->>CLI: Parse, normalize, validate nodes/edges
                alt Payload hợp lệ
                    CLI->>GraphCache: Ghi .tmp rồi atomic replace sang JSON cache
                else Payload lỗi
                    CLI-->>CLI: Tăng error count, không ghi cache hỏng
                end
            end
        end

        alt --skip-neo4j
            CLI-->>CLI: Bỏ qua Neo4j upsert
        else Upsert Neo4j
            CLI->>Neo4j: MERGE nodes và relationships
            Neo4j-->>CLI: Upsert hoàn tất
        end
    end

    alt --skip-qdrant
        CLI-->>Operator: Kết thúc không upsert vector
    else Embed và upsert Qdrant
        CLI->>Qdrant: Ensure collection acne_knowledge có dense=3072 và bm25
        alt Qdrant schema mismatch
            Qdrant-->>CLI: Báo thiếu/sai dense hoặc bm25
            CLI-->>Operator: Raise lỗi rõ ràng, không upsert âm thầm
        else Schema hợp lệ
            loop Theo clean chunk batches
                CLI->>CLI: Lọc noisy chunks
                CLI->>Gemini: Embed batch với task retrieval_document
                Gemini-->>CLI: Dense vectors
                CLI->>CLI: Compute hashed normalized log-TF sparse vectors
                CLI->>Qdrant: Upsert points gồm dense, bm25, payload, graph_nodes
                Qdrant-->>CLI: Batch upsert thành công
            end
        end
    end

    CLI-->>Operator: In summary, cache stats, upsert stats
```

## Diagram 2: Phase 2 Online Chat/RAG

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Frontend
    participant FastAPI
    participant PostgreSQL
    participant LangGraph
    participant Guardrails
    participant Redis as Redis Cache
    participant Retriever
    participant Qdrant
    participant Neo4j
    participant LLM as Gemini/Ollama

    User->>Frontend: Nhập câu hỏi
    Frontend->>FastAPI: POST /chat {message, session_id, history, model, bypass_cache}
    FastAPI->>FastAPI: Validate input và repair UTF-8/mojibake

    opt Follow-up có session_id nhưng frontend không gửi history
        FastAPI->>PostgreSQL: Load recent chat history
        PostgreSQL-->>FastAPI: Conversation history
    end

    FastAPI->>LangGraph: run_clinical_agent(state)
    LangGraph->>LangGraph: Normalize question
    LangGraph->>LangGraph: Rewrite follow-up thành standalone question nếu cần
    LangGraph->>Guardrails: Domain/safety guardrail

    alt Out-of-domain
        Guardrails-->>LangGraph: is_in_domain=false, guardrail=out_of_domain
        LangGraph->>LangGraph: Formatter tạo câu trả lời ngắn, skipped retrieval
        LangGraph-->>FastAPI: Safe out-of-domain response
    else Emergency/triệu chứng nghiêm trọng
        Guardrails-->>LangGraph: guardrail emergency/urgent
        LangGraph->>LangGraph: Formatter tạo cảnh báo đi khám/cấp cứu
        LangGraph-->>FastAPI: Urgent response, skipped retrieval
    else Prompt injection hoặc yêu cầu kê đơn
        Guardrails-->>LangGraph: unsafe_prescription_request
        LangGraph->>LangGraph: Từ chối kê đơn/chọn liều
        LangGraph-->>FastAPI: Safe refusal, skipped retrieval
    else Câu hỏi trong phạm vi mụn/da liễu
        LangGraph->>Redis: cache_lookup_node
        alt bypass_cache=true
            Redis-->>LangGraph: Bỏ qua lookup, reason=bypassed
        else Có history hoặc context rủi ro cao
            Redis-->>LangGraph: Cache skipped
        else Cache hit hợp lệ
            Redis-->>LangGraph: Cached answer, sources, metadata
            LangGraph->>LangGraph: Finalize cached response
            LangGraph-->>FastAPI: Cached response
        else Cache miss
            Redis-->>LangGraph: Cache miss
            LangGraph->>LangGraph: Extract symptoms/profile cơ bản
            LangGraph->>Retriever: Retrieve context
            Retriever->>Qdrant: Gemini query embedding + dense search using dense
            Qdrant-->>Retriever: Dense candidates
            Retriever->>Qdrant: Custom sparse TF search using legacy key bm25
            Qdrant-->>Retriever: Sparse candidates
            Retriever->>Retriever: RRF fusion, metadata boost, ưu tiên non-References
            Retriever->>Neo4j: Query graph facts từ graph_nodes hoặc keyword fallback
            Neo4j-->>Retriever: Graph facts
            Retriever-->>LangGraph: vector_contexts, graph_facts, sources
            LangGraph->>LangGraph: Safety check
            LangGraph->>LLM: Prompt y khoa với context, graph facts, safety flags

            alt Model chính lỗi và allow_model_fallback=true
                LLM-->>LangGraph: Lỗi primary model
                LangGraph->>LLM: Gọi fallback Gemini/Ollama
            else Model chính OK
                LLM-->>LangGraph: Draft answer
            end

            LLM-->>LangGraph: Draft answer
            LangGraph->>LangGraph: Formatter/post-process, nguồn, disclaimer
            LangGraph->>Redis: cache_store_node nếu eligible và quality gate pass

            opt Cacheable answer
                Redis-->>LangGraph: Store cache:answer:* với TTL
            end

            LangGraph-->>FastAPI: Final answer, sources, graph_facts, metadata
        end
    end

    FastAPI->>PostgreSQL: Persist chat session và messages
    alt Lưu DB thành công
        PostgreSQL-->>FastAPI: Commit
    else DB lỗi non-fatal
        PostgreSQL-->>FastAPI: Log warning, vẫn trả response
    end

    FastAPI-->>Frontend: ChatResponse
    Frontend-->>User: Hiển thị answer, sources, metadata, history
```
