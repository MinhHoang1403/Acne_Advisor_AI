"""Compatibility import for the canonical Phase 2 evidence retriever.

The implementation lives in :mod:`src.retrieval.service`; this module remains
as a stable import path for API clients written before S4B.
"""

from src.retrieval.service import EvidenceRetriever, RetrievalResult

HybridRetriever = EvidenceRetriever

__all__ = ["EvidenceRetriever", "HybridRetriever", "RetrievalResult"]
