# Methods and Formulas

## Dense Retrieval

Gemini `models/gemini-embedding-2`, 3072 dimensions and cosine distance. The
model does not receive a task type. Cache identity binds provider, model,
dimension, distance, task configuration and exact normalized text. Code:
`src/ingestion/embedding.py`.

## BM25

The sparse channel is Qdrant-native BM25 with collection IDF and shared
document/query preprocessing (`src/ingestion/bm25.py`). For term `t`, document
`d`, corpus size `N`, document frequency `n(t)`, length `|d|`, and average
length `avgdl`:

```text
IDF(t) = ln((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
score(t,d) = IDF(t) * tf(t,d) * (k1 + 1)
             / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

Frozen Qdrant configuration: `k1=1.2`, `b=0.75`, `avg_len=256`, tokenizer
`word`, lowercase enabled, language `none`, no stemming, stopword list or ASCII
folding. These are explicit implementation/project parameters, not claimed
clinical constants. Tests compare hand calculations and provider schema.

## Chunking and Filtering

Structure-first Markdown chunking uses a 2400 Unicode-character cap and zero
overlap. It was selected from three actual-corpus candidates using chunk length,
boundary, duplication and safety-text retention diagnostics. Filtering removes
only empty/page-number/TOC/parser artifacts and exact duplicates; short medical
content is retained.

## Retrieval V5 and P3/P4

Reciprocal Rank Fusion remains in `src/retrieval/rrf.py`; retrieval orchestration
is Phase 2 and is not part of the frozen Phase 1 build formula. P3 allows at
most two retrieval attempts. P4 remains shadow-only. EntityCards and graph
signals are discovery/structure aids and cannot independently ground medical
answers.
