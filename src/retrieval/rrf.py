"""Hợp nhất thứ hạng Dense và BM25 bằng Reciprocal Rank Fusion (RRF).

Dense cosine score và BM25 score có scale khác nhau, vì vậy module này không
cộng raw score. RRF dùng vị trí one-indexed trong từng channel theo công thức
``score(d) = sum_c w_c / (k + rank_c(d))``. Candidate xuất hiện ở cả hai
channel nhận cả hai contribution; channel không chứa candidate thì không đóng
góp cho candidate đó.

Output đã sắp theo ``rrf_score`` được chuyển tiếp tới context packer. Module này
không truy vấn Qdrant, không rerank theo nội dung và không đánh giá medical truth.
Muốn đổi công thức fusion bắt đầu tại ``reciprocal_rank_fusion()``; các default
runtime hiện được truyền từ ``src/retrieval/service.py``.
"""

from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    *,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    k: int = 60,
) -> list[dict[str, Any]]:
    """Hợp nhất hai ranked list theo point ID và rank one-indexed.

    ``weight`` là ``w_c``, ``rank`` là ``rank_c(d)`` và ``k`` là hằng số trong
    mẫu số. ``k`` làm mềm ảnh hưởng của vị trí tuyệt đối; nó không phải relevance
    threshold. Hai weight bằng nhau biểu diễn equal-weight fusion.
    """

    scores: dict[str, float] = {}
    documents: dict[str, dict[str, Any]] = {}

    # Raw score vẫn được giữ trong metadata để quan sát, nhưng chỉ contribution
    # ``weight / (k + rank)`` tham gia thứ tự fused cuối cùng.
    for channel, weight, rank_key, score_key in (
        (dense_results, dense_weight, "dense_rank", "dense_score"),
        (sparse_results, sparse_weight, "sparse_rank", "sparse_score"),
    ):
        for rank, document in enumerate(channel, start=1):
            document_id = str(document.get("id", ""))
            if not document_id:
                continue
            scores[document_id] = scores.get(document_id, 0.0) + weight / (k + rank)
            documents.setdefault(document_id, dict(document))
            documents[document_id][rank_key] = rank
            documents[document_id][score_key] = document.get("score", 0.0)

    ranked_ids = sorted(scores, key=lambda document_id: scores[document_id], reverse=True)
    fused: list[dict[str, Any]] = []
    for document_id in ranked_ids:
        document = documents[document_id]
        document["rrf_score"] = round(scores[document_id], 6)
        document["score"] = document["rrf_score"]
        fused.append(document)
    return fused


__all__ = ["reciprocal_rank_fusion"]
