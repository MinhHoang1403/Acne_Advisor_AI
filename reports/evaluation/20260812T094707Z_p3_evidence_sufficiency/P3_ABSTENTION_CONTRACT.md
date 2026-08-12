# P3 Abstention Contract

| Abstention type | Trigger | Retry allowed? | Final behavior | Safety rationale |
| --- | --- | --- | --- | --- |
| `INSUFFICIENT_EVIDENCE` | Required source-backed roles still missing | Only once on attempt 0 | Deterministic evidence-limited response | Avoid unsupported conclusions |
| `CRITICAL_EVIDENCE_MISSING` | Pregnancy, contraindication, emergency, or other existing critical role lacks source-backed evidence | Only when the attempt-0 PRE_PACK miss is recoverable | No unsupported critical recommendation; severity guard still applies | Preserve P0 medical gates |
| `OUT_OF_SCOPE` | Domain contract rejects the request | No | Existing domain-safe response path | Do not search unrelated material |
| `SOURCE_PROVENANCE_FAILURE` | Evidence lacks `chunk_id` and a document/source identity | No | State that traceable evidence is unavailable | Untraceable claims cannot support medical advice |
| `RETRIEVAL_PROVIDER_FAILURE` | V5 retrieval returns a recoverable provider error | No additional P3 retry | Existing temporary retrieval-error response | P3 does not replace provider resilience |

The retrieval layer emits structured `EvidenceAbstention`; prose is produced only by the existing safe-fallback/finalization layer. Every abstention is non-cacheable and passes through the severity-aware answer guard.
