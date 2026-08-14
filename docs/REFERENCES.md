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
  normalization.
- Cormack, G. V., Clarke, C. L. A., and Buettcher, S. (2009). *Reciprocal Rank
  Fusion Outperforms Condorcet and Individual Rank Learning Methods*. SIGIR.
  DOI: 10.1145/1571941.1572114. Supports rank-only fusion at the runtime
  boundary; it is not part of Phase 1 indexing.
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks*. NeurIPS. Establishes retrieval-grounded
  generation as a distinct model/evidence architecture.
- Yao, S. et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language
  Models*. ICLR. Supports explicit interleaving of model decisions and bounded
  actions/tools.
- Jiang, Z. et al. (2023). *Active Retrieval Augmented Generation*. EMNLP.
  Supports retrieval decisions driven by evidence need rather than an
  unconditionally fixed sequence.
- Jeong, S. et al. (2024). *Adaptive-RAG: Learning to Adapt Retrieval-Augmented
  Large Language Models through Question Complexity*. NAACL. Supports bounded
  adaptation of retrieval behavior to the request.

## Framework Contracts

- LangGraph, *Graph API* and `StateGraph` reference, verified 2026-08-14.
  https://docs.langchain.com/oss/python/langgraph/graph-api
  https://reference.langchain.com/python/langgraph/graph/state/StateGraph
- LangChain, *Tools* and LangGraph *Agentic RAG*, verified 2026-08-14.
  https://docs.langchain.com/oss/python/langchain/tools
  https://docs.langchain.com/oss/python/langgraph/agentic-rag
- Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain
  Question Answering*. EMNLP. DOI: 10.18653/v1/2020.emnlp-main.550. Supports
  dense retrieval as a semantic channel.
- Qu, C. et al. (2025). *Document Segmentation Matters for Retrieval-Augmented
  Generation*. Findings of ACL. DOI: 10.18653/v1/2025.findings-acl.422.
  Supports evaluating segmentation on the actual corpus instead of assuming a
  universal chunk size.

## Provider Contracts

- Qdrant, *Full-text search* and *BM25 inference*. Verified 2026-08-14.
  https://qdrant.tech/documentation/search/text-search/full-text-search/
  https://qdrant.tech/documentation/inference/inference-bm25/
- Google AI for Developers, *Gemini Embedding 2* and *Embeddings*. Verified
  2026-08-14. https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2
  https://ai.google.dev/gemini-api/docs/embeddings
- LlamaIndex, `llama-parse` 0.6.94 package contract. Verified 2026-08-14.
  https://pypi.org/project/llama-parse/0.6.94/

## Canonical Clinical Corpus

- National Institute for Health and Care Excellence. *Acne vulgaris:
  management (NG198)*. Updated 2026-08-03.
  https://www.nice.org.uk/guidance/ng198
- Reynolds, R. V. et al. (2024). *Guidelines of care for the management of acne
  vulgaris*. Journal of the American Academy of Dermatology.
  DOI: 10.1016/j.jaad.2023.12.017.
- Bo Y te Viet Nam. Quyet dinh 4416/QD-BYT ngay 06/12/2023, *Huong dan chan
  doan va dieu tri cac benh da lieu*; the local corpus contains only the acne
  excerpt (pages 433-442).
- American Academy of Dermatology public acne education pages, frozen crawl
  retrieved 2026-07-03. https://www.aad.org/public/diseases/acne

## Current Safety Cross-Checks

These sources validate currency and safety review; they are not silently
inserted into the canonical retrieval corpus.

- UK MHRA isotretinoin safety communications, verified 2026-08-14.
  https://www.gov.uk/drug-safety-update/isotretinoin-roaccutanev-new-safety-measures-to-be-introduced-in-the-coming-months-including-additional-oversight-on-initiation-of-treatment-for-patients-under-18-years
- European Medicines Agency, retinoid pregnancy-prevention measures, verified
  2026-08-14. https://www.ema.europa.eu/en/medicines/human/referrals/retinoid-containing-medicinal-products
