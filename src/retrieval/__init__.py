"""Public source-evidence retrieval API."""

from src.retrieval.contracts import (
    NormalizedQuery,
    PackedContext,
    RetrievedCandidate,
)
from src.retrieval.context_packer import pack_context
from src.retrieval.reranker import CandidateReranker, RerankerSettings, rerank_candidates
from src.retrieval.service import EvidenceRetriever, RetrievalResult, retrieve_evidence

__all__ = [
    "NormalizedQuery",
    "PackedContext",
    "RetrievedCandidate",
    "CandidateReranker",
    "EvidenceRetriever",
    "RetrievalResult",
    "pack_context",
    "rerank_candidates",
    "RerankerSettings",
    "retrieve_evidence",
]
