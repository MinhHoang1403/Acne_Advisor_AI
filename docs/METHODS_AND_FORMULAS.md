# Methods and Formulas

## Dense Retrieval

The frozen knowledge collection uses `models/gemini-embedding-2`, 3072
dimensions, and cosine distance. Runtime query embeddings use the same model
contract. Runtime retrieval does not rebuild or alter stored vectors.

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

The runtime uses `k=60`, `w_dense=1.0`, and `w_bm25=1.0`. Equal weights are an explicit
engineering policy, not a clinical constant. No post-fusion relevance
adjustment or structural-store score changes the fused ranking.

## Context and Evidence Contracts

`src/retrieval/context_packer.py` preserves fused order, deduplicates only by
stable item identity, retains provenance, and enforces finite item/character
budgets. Evidence is marked usable only when at least one packed item has both
text and a source identifier. This is a deterministic presence/provenance gate,
not a semantic sufficiency or entailment claim. The exact bounded
`PackedContext.context_text` is the evidence sent to generation; full candidate
text cannot bypass the configured character limit.

Retrieval is bounded by `RETRIEVAL_TIMEOUT_SECONDS` and two attempts. These are
engineering safety/latency policies. Failure to obtain provenance-complete
evidence results in explicit abstention. One failed Dense or BM25 channel is
reported as `degraded_dense` or `degraded_bm25`; ranking policy and RRF weights
remain unchanged.

## Frozen Phase 1

Structure-first chunking remains capped at 2400 Unicode characters with zero
overlap. Phase 1 filtering, provenance, embedding, BM25, taxonomy, EntityCards,
and Neo4j contracts are frozen.

## Parameter Classification

| Parameter | Value | Classification |
|---|---:|---|
| chunk maximum | 2400 Unicode characters | frozen project engineering contract |
| chunk overlap | 0 | frozen project engineering contract |
| Dense dimensions | 3072 | provider/index compatibility contract |
| BM25 `k1`, `b`, `avg_len` | 1.2, 0.75, 256 | frozen provider configuration |
| RRF `k` | 60 | runtime engineering policy |
| channel weights | 1.0 / 1.0 | runtime engineering policy |
| context items/chars | env-bounded defaults 8 / 6000 | runtime resource policy |
| retrieval attempts | maximum 2 | bounded safety/latency policy |

These values are not clinical constants. Their definitions and method sources
are mapped in `data/phase1_method_sources.json` and [References](REFERENCES.md).
