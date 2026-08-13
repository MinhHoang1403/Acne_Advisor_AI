# Agent Workflow

The production entrypoint is `build_clinical_graph()` in `src/agent/graph.py`.
It compiles a LangGraph `StateGraph` over `ClinicalState`.

The main flow is:

```text
normalize -> rewrite -> severity -> guard -> cache lookup
  -> extract -> retrieve -> evidence sufficiency
  -> bounded retry or abstention
  -> safety -> generate -> provider fallback decision
  -> claim grounding -> finalize -> answer quality
  -> cache store -> observability export
```

Conditional edges preserve early guard responses, valid cache hits, bounded P3
retry, deterministic abstention, and generation fallback. The API path continues
to call this graph; no minimal baseline or plain sequential chain replaces it.

The current `ClinicalState` is intentionally retained. A major field reduction,
node merge, or tool-oriented redesign belongs to S4B and must be supported by
behavioral evidence before implementation.
