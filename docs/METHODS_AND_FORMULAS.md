# Methods and Formulas

## Dense Retrieval

The frozen knowledge collection uses `models/gemini-embedding-2`, 3072
dimensions, and cosine distance. Runtime query embeddings use the same model
contract. S4B does not rebuild or alter stored vectors.

## Native BM25

The sparse channel is Qdrant-native BM25 with collection-side IDF and the frozen
Phase 1 configuration: `k1=1.2`, `b=0.75`, `avg_len=256`, tokenizer `word`,
lowercase enabled, language `none`, no stemming, and no stopword list.

For term `t`, document `d`, corpus size `N`, document frequency `n(t)`, document
length `|d|`, and average length `avgdl`:

```text
IDF(t) = ln((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
score(t,d) = IDF(t) * tf(t,d) * (k1 + 1)
             / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

## Reciprocal Rank Fusion

Dense and BM25 results are fused only by rank in `src/retrieval/rrf.py`:

```text
RRF(d) = sum_r w_r / (k + rank_r(d))
```

S4B uses `k=60`, `w_dense=1.0`, and `w_bm25=1.0`. Equal weights are an explicit
engineering policy, not a clinical constant. No metadata boost, reranker,
candidate policy, selector, Entity score, or Graph score changes relevance.

## Context and Evidence Contracts

`src/retrieval/context_packer.py` preserves fused order, deduplicates only by
stable item identity, retains provenance, and enforces finite item/character
budgets. Evidence is sufficient only when at least one packed item has both
medical text and a source identifier. This is a deterministic provenance gate;
semantic interpretation remains with the LLM.

Retrieval is bounded by `RETRIEVAL_TIMEOUT_SECONDS` and two attempts. These are
engineering safety/latency policies. Failure to obtain provenance-complete
evidence results in explicit abstention.

## Frozen Phase 1

Structure-first chunking remains capped at 2400 Unicode characters with zero
overlap. Phase 1 filtering, provenance, embedding, BM25, taxonomy, EntityCards,
and Neo4j contracts are unchanged by S4B.
