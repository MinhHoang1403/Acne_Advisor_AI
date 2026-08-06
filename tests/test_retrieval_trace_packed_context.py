from src.retrieval.context_packer import pack_context
from src.retrieval.contracts import QueryExpansion, RetrievalTrace
from src.retrieval.query_normalization import normalize_query
from src.database.retriever import _sources_for_contexts


def test_retrieval_trace_accepts_packed_context_without_breaking_old_fields():
    normalized = normalize_query("Mụn đầu đen là gì?")
    expansion = QueryExpansion(
        original_query=normalized.original_query,
        normalized_query=normalized,
        expanded_terms=[normalized.original_query],
    )
    packed = pack_context(normalized, [], max_items=3)
    trace = RetrievalTrace(
        original_query=normalized.original_query,
        normalized_query=normalized,
        expansion=expansion,
        packed_context=packed,
    )

    data = trace.model_dump(mode="json")

    assert data["selected_context"] == []
    assert data["packed_context"]["intent"] == "acne_type"
    assert "warnings" in data["packed_context"]


def test_sources_include_every_document_used_in_the_packed_prompt_context():
    sources = _sources_for_contexts(
        [
            {"source_file": "entity:active_ingredient", "retrieval_source": "entity"},
            {"source_file": "guideline.pdf", "retrieval_source": "chunk"},
            {"source_file": "dataset.json", "retrieval_source": "chunk"},
            {"source_file": "guideline.pdf", "retrieval_source": "chunk"},
        ]
    )

    assert sources == ["guideline.pdf", "dataset.json"]
