"""Public source-evidence retrieval API."""

from src.retrieval.contracts import (
    NormalizedQuery,
    PackedContext,
    RetrievedCandidate,
)
from src.retrieval.context_packer import pack_context
from src.retrieval.service import EvidenceRetriever, RetrievalResult, retrieve_evidence

__all__ = [
    "NormalizedQuery",
    "PackedContext",
    "RetrievedCandidate",
    "EvidenceRetriever",
    "RetrievalResult",
    "pack_context",
    "retrieve_evidence",
]
