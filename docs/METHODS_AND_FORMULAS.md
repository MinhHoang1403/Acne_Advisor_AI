# Methods and Formulas

## Dense Retrieval

The current indexed knowledge collection uses `models/gemini-embedding-2`, 3072
dimensions, and cosine distance. Runtime query embeddings use the same model
contract. Runtime retrieval does not rebuild or alter stored vectors.

Google owns vector generation; the project sends text and validates output count
and dimension; Qdrant owns cosine search over stored vectors. Official Gemini
Embedding 2 documentation does not accept `task_type` and recommends embedding
task instructions as text prefixes for text-only retrieval. The current build and
runtime query path use unprefixed text. Whether query/document prefixes improve
Vietnamese acne retrieval is an unmeasured evaluation question that would require
a controlled re-embedding experiment; this documentation does not change the
current vectors or index.

## Native BM25

The sparse channel is Qdrant-native BM25 with collection-side IDF and the current
build configuration: `k1=1.2`, `b=0.75`, `avg_len=256`, tokenizer `word`, lowercase
enabled, language `none`, no stemming, and no stopword list. Here `avg_len=256` is
a configured provider/index baseline. It is not asserted to be the measured true
average of the current corpus. Neither the BM25 literature nor Qdrant provider
documentation proves these values optimal for Acne Advisor AI.

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

The runtime uses `k=60`, `w_dense=1.0`, and `w_bm25=1.0`. The RRF paper supports
rank-based reciprocal fusion; it does not establish those project constants as
optimal. Equal weights are an explicit engineering policy, not a clinical
constant. No post-fusion relevance adjustment or structural-store score changes
the fused ranking.

## Context and Evidence Contracts

`src/retrieval/context_packer.py` preserves fused order, deduplicates only by
stable item identity, retains provenance, and enforces finite item/character
budgets. The defaults are 8 packed items and 6000 characters from an initial
candidate pool of 16. These are resource policies, not relevance thresholds.
Evidence is marked usable only when at least one packed item has both
text and a source identifier. This is a deterministic presence/provenance check,
not a semantic sufficiency or entailment claim. The exact bounded
`PackedContext.context_text` is the evidence sent to generation; full candidate
text cannot bypass the configured character limit.

Each Dense and BM25 channel is independently bounded by
`RETRIEVAL_TIMEOUT_SECONDS`, and the agent remains bounded to two retrieval
executions. These are engineering safety/latency policies. A timed-out channel
does not discard evidence already returned by the other channel. Failure to
obtain provenance-complete evidence results in explicit abstention. One failed
Dense or BM25 channel is reported as `degraded_dense` or `degraded_bm25`;
ranking policy and RRF weights remain unchanged.

## Bounded Agent Decision

`src/agent/action_decision.py` asks the configured generation model for one
strict `AgentDecision` JSON object. The action space is `retrieve`, `retry`,
`generate`, or `abstain`; the object may also carry a bounded retrieval query
and a reason code, but no medical answer or free-form chain of thought. This is
an implemented agent method related to ReAct, Active RAG, and Adaptive-RAG, not
a reproduction of any one paper.

Python enforces the transition contract and `MAX_RETRIEVAL_ATTEMPTS = 2`.
Schema-invalid output, a model-selected impossible transition, a repeated
non-recoverable query, or an exhausted attempt budget becomes abstention. The
model owns semantic action choice; deterministic code owns finite execution and
fail-closed behavior. This is an engineering policy: `retrieve` means the first
evidence acquisition, `retry` means the later acquisition, and the retrieval
tool can execute no more than two times. The model cannot bypass the budget by
selecting `retrieve` after the first execution.

## Exact Cache Identity

`src/cache/exact_cache.py` normalizes a question by Unicode-aware case folding,
replacing `?!.,:;` with spaces, collapsing whitespace, and then requiring an
exact normalized match. The SHA-256 cache-key payload is:

```text
cache_schema_version | answer_cache_version | pipeline_fingerprint
| normalized_question | provider | model
```

This is an engineering cache policy. It does not compute embedding similarity
and makes no semantic-equivalence claim. SHA-256 follows NIST FIPS 180-4, but
that standard does not validate the chosen fields or normalization policy.

The pipeline fingerprint uses the first 24 hexadecimal characters of SHA-256
over a canonical, secret-free JSON manifest. It is a deterministic cache/version
partition, not an authentication signature. EntityCard Qdrant point IDs use
UUIDv5 from RFC 9562 with `uuid.NAMESPACE_URL` and the canonical project entity
identity. The standard defines deterministic UUID construction, not the
project's canonical-name semantics or provenance quality.

## Runtime Resilience

Provider retries use bounded exponential delay with positive jitter:

```text
base(i)  = base_delay * 2^(max(0, i - 1))
capped   = min(base(i), max_delay)
delay(i) = min(capped + U(0, capped * jitter_ratio), max_delay)
```

The default is one retry, a 1-second base, a 4-second cap, and a 0.1 jitter
ratio, always constrained by the request deadline. HTTP 429/500/502/503/504,
timeouts, and transport failures are transient; HTTP 400/401/403/404/422 and
validation/configuration failures are permanent. These classifications are an
engineering resilience policy, not a medical or research result.

All stages share one finite deadline. For configured stage timeout `T` and
remaining request budget `R`, the effective timeout is:

```text
effective_timeout = min(T, R)
```

Retry and provider fallback consume the same remaining budget. AWS engineering
guidance supports finite timeout, retry, exponential backoff, and jitter as a
pattern; it does not prescribe this project's exact retry count or timing values.

The Agent decision model sees at most the first 5 evidence items and the first
1200 characters of each item. This bounded visibility is an engineering policy,
not evidence sufficiency or a scientific optimum. The generation path can see
more packed evidence through its separate limit of 8 items and 6000 characters;
that asymmetry remains an evaluation question.

When Ollama reports truncated generation, the client performs at most one compact
retry with an instruction capped at 160 words. This engineering policy can change
output length, latency, provider-call count, completeness, and style; it is not
claimed optimal.

## Safety and Verification

`src/agent/safety_policy.py` owns nine narrow deterministic overrides mapped to
clinical/public-health sources or an explicit no-prescription engineering
policy. It is not a general medical reasoner. Ordinary answer meaning remains
`source evidence -> LLM synthesis`.

`src/quality/answer_verifier.py` checks answer structure, source allowlisting,
and provenance-related contracts. It does not prove medical truth, semantic
entailment, or evidence completeness.

## Knowledge Build Contracts

Structure-aware chunking is capped at 2400 Unicode characters with zero overlap.
Filtering, provenance, embedding, BM25, taxonomy, EntityCard, and Neo4j settings
are versioned parts of the current validated build.

Chunk boundaries prefer headings, paragraphs, and sentences. Segmentation
literature supports the claim that segmentation can affect RAG, but the project
does not implement PIC and has not established the 2400-character cap or zero
overlap as optimal.

Domain metadata retains a compatibility field named `confidence`, calculated
only as a coverage heuristic over `n` populated metadata groups:

```text
metadata_coverage(n) = 0                              if n = 0
metadata_coverage(n) = min(0.3 + 0.1 * n, 1.0)       otherwise
```

This value does not enter retrieval scoring and is not a calibrated probability,
medical confidence, source reliability score, or scientific formula.

## Parameter Classification

| Parameter | Value | Classification |
|---|---:|---|
| chunk maximum | 2400 Unicode characters | versioned build engineering contract |
| chunk overlap | 0 | versioned build engineering contract |
| Dense dimensions | 3072 | provider/index compatibility contract |
| BM25 `k1`, `b`, `avg_len` | 1.2, 0.75, 256 | indexed provider configuration |
| RRF `k` | 60 | runtime engineering policy |
| channel weights | 1.0 / 1.0 | runtime engineering policy |
| retrieval candidates | default 16 | runtime resource policy |
| context items/chars | defaults 8 / 6000 | runtime resource policy |
| retrieval attempts | maximum 2 | bounded safety/latency policy |
| decision evidence visibility | first 5 items; 1200 chars/item | runtime engineering policy |
| agent action schema | 4 semantic actions | implemented research method + engineering contract |
| exact cache namespace | v10 | engineering invalidation policy |
| cache question/TTL | 600 chars / 86400 seconds | engineering resource policy |
| request/history bounds | 500 chars; 10 messages x 1000 chars | engineering resource policy |
| total/retrieval deadlines | 210 / 20 seconds | engineering resilience policy |
| Gemini/Ollama call deadlines | 45 / 160 seconds | engineering resilience policy |
| LLM retries | 1 retry, 1-second base, 4-second cap, 0.1 jitter | engineering resilience policy |
| Ollama truncation retry | 1 compact retry; 160-word instruction bound | provider-specific engineering policy |
| safety overrides | 9 source-mapped rules | clinical safety sources + engineering policy |
| answer shape parsing | requested table/column/item/style only | structural engineering policy |
| source allowlist | evidence source IDs only | provenance engineering contract |
| core preflight | Qdrant, query embedding, generation provider | architecture dependency contract |

These values are not clinical constants. Their definitions, code owners, sources,
adaptations, and limitations are mapped in
[Method Traceability](METHOD_TRACEABILITY.md),
`data/method_sources.json`, and [References](REFERENCES.md).

## Source Classification

Project claims use these labels:

| Classification | Meaning |
|---|---|
| `IMPLEMENTED_RESEARCH_METHOD` | a cited method is present in code, with project-specific adaptation stated |
| `OFFICIAL_PROVIDER_CONTRACT` | behavior required by an external SDK/service contract |
| `OFFICIAL_FRAMEWORK_CONTRACT` | orchestration or API behavior required by framework documentation |
| `CLINICAL_SAFETY_SOURCE` | an external source supports a narrow deterministic safety action |
| `RELATED_LITERATURE` | relevant research that is not claimed as implemented |
| `ENGINEERING_POLICY` | a bounded project decision without scientific-universality claim |
| `EMPIRICAL_PROJECT_DECISION` | a value selected from project measurements and valid only for this system |
