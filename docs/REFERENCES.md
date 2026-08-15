# Method, Provider, and Medical Sources

Verified through 2026-08-14. Machine-readable claim mapping lives in
`data/phase1_method_sources.json`; canonical corpus metadata lives in
`data/sources/manifest.yaml`. Scientific evidence, provider contracts, and
clinical sources are deliberately separated.

## Retrieval Methods

- Robertson, S. E., and Zaragoza, H. (2009). *The Probabilistic Relevance
  Framework: BM25 and Beyond*. Foundations and Trends in Information
  Retrieval, 3(4), 333-389. DOI: 10.1561/1500000019. Supports the BM25 family,
  term-frequency saturation, inverse document frequency, and length
  normalization. It does not specify Qdrant implementation details or prove the
  project's `k1`, `b`, `avg_len`, tokenizer, or language settings optimal.
- Cormack, G. V., Clarke, C. L. A., and Buettcher, S. (2009). *Reciprocal Rank
  Fusion Outperforms Condorcet and Individual Rank Learning Methods*. SIGIR.
  DOI: 10.1145/1571941.1572114. Supports rank-only fusion at the runtime
  boundary; it is not part of knowledge indexing. It does not prove `k=60` or
  equal Dense/BM25 weights optimal for this corpus.
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks*. NeurIPS. Establishes retrieval-grounded
  generation as a distinct model/evidence architecture.
- Yao, S. et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language
  Models*. ICLR. Supports explicit interleaving of model decisions and bounded
  actions/tools. Acne Advisor AI uses a strict bounded action object and does not
  reproduce the paper's prompt setup or expose free-form chain of thought.
- Jiang, Z. et al. (2023). *Active Retrieval Augmented Generation*. EMNLP.
  Supports retrieval decisions driven by evidence need rather than an
  unconditionally fixed sequence. The project does not implement FLARE's
  token-level mechanism, and the source does not validate the two-attempt limit.
- Jeong, S. et al. (2024). *Adaptive-RAG: Learning to Adapt Retrieval-Augmented
  Large Language Models through Question Complexity*. NAACL. Supports bounded
  adaptation of retrieval behavior to the request. The project does not
  implement the paper's complexity classifier or strategy set.

## Framework Contracts

- LangGraph, *Graph API* and `StateGraph` reference, verified 2026-08-14.
  https://docs.langchain.com/oss/python/langgraph/graph-api
  https://reference.langchain.com/python/langgraph/graph/state/StateGraph
- LangChain, *Tools* and LangGraph *Agentic RAG*, verified 2026-08-14.
  https://docs.langchain.com/oss/python/langchain/tools
  https://docs.langchain.com/oss/python/langgraph/agentic-rag
- Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain
  Question Answering*. EMNLP. DOI: 10.18653/v1/2020.emnlp-main.550. Supports
  dense retrieval as a semantic channel. It does not validate Gemini Embedding
  2, Vietnamese cross-lingual retrieval, or this medical corpus.
- Wang, Z. et al. (2025). *Document Segmentation Matters for
  Retrieval-Augmented Generation*. Findings of the Association for
  Computational Linguistics: ACL 2025, 8063-8075.
  DOI: 10.18653/v1/2025.findings-acl.422. This is related literature on PIC and
  segmentation effects. The project does not implement PIC, and the paper does
  not validate the project's 2400-character cap.

## Provider Contracts

- Qdrant, *Full-text search* and *BM25 inference*. Verified 2026-08-14.
  https://qdrant.tech/documentation/search/text-search/full-text-search/
  https://qdrant.tech/documentation/inference/inference-bm25/
  These pages define provider behavior/configuration, not retrieval quality or
  parameter optimality for Acne Advisor AI.
- Qdrant, *Collections and distance metrics*. Verified 2026-08-14.
  https://qdrant.tech/documentation/manage-data/collections/
  This defines the cosine-distance provider contract; it does not validate the
  semantic quality of vectors or ranked results.
- Google AI for Developers, *Gemini Embedding 2* and *Embeddings*. Verified
  2026-08-14. https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2
  https://ai.google.dev/gemini-api/docs/embeddings
  The documentation supports the model/dimension contract, notes that Gemini
  Embedding 2 does not accept `task_type`, and recommends text-prefix retrieval
  instructions for text-only use cases. It does not validate Vietnamese acne
  retrieval quality or whether such prefixes improve this corpus. The current
  index remains unprefixed pending a controlled evaluation.
- LlamaIndex, `llama-parse` 0.6.94 package contract. Verified 2026-08-14.
  https://pypi.org/project/llama-parse/0.6.94/

## Technical Standards

- National Institute of Standards and Technology. *Secure Hash Standard (SHS),
  FIPS PUB 180-4*. Updated August 2015. DOI: 10.6028/NIST.FIPS.180-4.
  https://csrc.nist.gov/pubs/fips/180-4/upd1/final
  This standard defines SHA-256. It does not validate project field selection,
  input normalization, digest truncation, cache semantics, or authenticity.
- Davis, K., Peabody, B., and Leach, P. *Universally Unique IDentifiers
  (UUIDs), RFC 9562*. May 2024. DOI: 10.17487/RFC9562.
  https://www.rfc-editor.org/info/rfc9562/
  This standard defines UUIDv5 construction. It does not validate project entity
  canonicalization, source provenance, or identity semantics.

## Resilience Engineering

- Brooker, M.; Amazon Web Services. *Timeouts, retries, and backoff with jitter*.
  Amazon Builders' Library, 2019.
  https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/
  This supports finite timeout, bounded retry, exponential backoff, and jitter as
  engineering patterns. It does not prescribe the project's exact timeout,
  retry count, delay cap, or jitter ratio.

## Canonical Clinical Corpus

- National Institute for Health and Care Excellence. *Acne vulgaris:
  management (NG198)*. Official metadata reports an update on 2026-04-30.
  https://www.nice.org.uk/guidance/ng198
- Reynolds, R. V. et al. (2024). *Guidelines of care for the management of acne
  vulgaris*. Journal of the American Academy of Dermatology.
  DOI: 10.1016/j.jaad.2023.12.017.
- Bo Y te Viet Nam. Quyet dinh 4416/QD-BYT ngay 06/12/2023, *Huong dan chan
  doan va dieu tri cac benh da lieu*; the local corpus contains only the acne
  excerpt (pages 433-442).
- American Academy of Dermatology public acne education pages, source snapshot
  retrieved 2026-07-03. https://www.aad.org/public/diseases/acne

### NICE Source Provenance

The NICE-derived project snapshot was acquired through a text-rendering transport
and represents 2026-08-03, which differs from the official NICE metadata date
above. A complete official replacement was unavailable through the supported
acquisition routes. The snapshot remains part of the research corpus, but the
project does not claim it is a fully verified current official version. This
provenance note does not change the source bytes, source ID, build identity,
Qdrant index, or Neo4j graph.

## Current Safety Cross-Checks

These sources validate currency and safety review; they are not silently
inserted into the canonical retrieval corpus.

- UK MHRA isotretinoin safety communications, verified 2026-08-14.
  https://www.gov.uk/drug-safety-update/isotretinoin-roaccutanev-new-safety-measures-to-be-introduced-in-the-coming-months-including-additional-oversight-on-initiation-of-treatment-for-patients-under-18-years
  https://www.gov.uk/drug-safety-update/oral-retinoids-pregnancy-prevention-reminder-of-measures-to-minimise-teratogenic-risk
- European Medicines Agency, retinoid pregnancy-prevention measures, verified
  2026-08-14. https://www.ema.europa.eu/en/medicines/human/referrals/retinoid-containing-medicinal-products
- NHS, *Anaphylaxis* and *Chest pain*, verified 2026-08-14.
  https://www.nhs.uk/conditions/anaphylaxis/
  https://www.nhs.uk/symptoms/chest-pain/
- World Health Organization, *Suicide: questions and answers*, verified
  2026-08-14. https://www.who.int/news-room/questions-and-answers/item/suicide
- DailyMed, isotretinoin medication guide, verified 2026-08-14.
  https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=72867c88-070f-4608-bfef-cc5225ebce6d
