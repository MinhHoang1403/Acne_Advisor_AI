from __future__ import annotations

import random

from src.retrieval.context_packer import pack_context
from src.retrieval.contracts import NormalizedQuery, RetrievedCandidate
from src.retrieval.rrf import reciprocal_rank_fusion


def test_duplicate_channel_document_does_not_manufacture_rrf_support():
    a, b = {"id": "a", "text": "A"}, {"id": "b", "text": "B"}
    expected = reciprocal_rank_fusion([a, b], [b, a])
    assert reciprocal_rank_fusion([a, a, b], [b, a, a]) == expected


def test_sampled_whole_chunk_packing_invariants_and_source_identity():
    rng = random.Random(20261025)
    query = NormalizedQuery(original_query="Question", normalized_text="Question")
    for _ in range(80):
        candidates = [RetrievedCandidate(
            candidate_id=str(i), collection="test", text=("Không được cắt mất phủ định. " * rng.randint(1, 180)).strip(),
            payload={"source_id": f"source-{i}", "chunk_id": str(i)},
        ) for i in range(rng.randint(1, 25))]
        candidates += rng.sample(candidates, min(3, len(candidates)))
        rng.shuffle(candidates)
        budget, items = rng.randint(256, 7000), rng.randint(1, 9)
        packed = pack_context(query, candidates, max_items=items, max_chars=budget)
        original = {c.candidate_id: c for c in candidates}
        assert len(packed.context_text) <= budget
        assert len(packed.items) <= items
        assert len({item.item_id for item in packed.items}) == len(packed.items)
        for item in packed.items:
            assert item.text == original[item.item_id].text
            assert item.payload["source_id"] == original[item.item_id].payload["source_id"]
