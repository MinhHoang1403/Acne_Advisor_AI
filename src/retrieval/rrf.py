"""Hợp nhất thứ hạng Dense, BM25 và Neo4j Graph bằng Reciprocal Rank Fusion (RRF).

Dense cosine score, BM25 score và Neo4j graph score có thang đo khác nhau, vì vậy
module này không cộng raw score. RRF dùng vị trí one-indexed trong từng channel
theo công thức ``score(d) = sum_c w_c / (k + rank_c(d))`` (Cormack et al., 2009).
Candidate xuất hiện ở nhiều channel nhận đóng góp cộng hưởng từ tất cả các channel
đó; channel không chứa candidate thì không đóng góp cho candidate đó.

Output đã sắp theo ``rrf_score`` được chuyển tiếp tới local reranker và context packer.
Module này không truy vấn datastore, không rerank theo nội dung và không đánh giá
medical truth. Muốn đổi công thức fusion bắt đầu tại ``reciprocal_rank_fusion()``;
các default runtime hiện được truyền từ ``src/retrieval/service.py``.
"""

from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    *,
    graph_results: list[dict[str, Any]] | None = None,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    graph_weight: float = 0.85,
    k: int = 60,
) -> list[dict[str, Any]]:
    """Hợp nhất danh sách xếp hạng đa kênh (2 hoặc 3 kênh) theo point ID và rank one-indexed.

    ``weight`` là ``w_c``, ``rank`` là ``rank_c(d)`` và ``k`` là hằng số làm mượt
    trong mẫu số. Tham số ``graph_results`` là tùy chọn nhằm bảo toàn tương thích
    ngược tuyệt đối với các lời gọi 2 kênh hiện có.
    """

    scores: dict[str, float] = {}
    documents: dict[str, dict[str, Any]] = {}

    channels: list[tuple[list[dict[str, Any]], float, str, str]] = [
        (dense_results, dense_weight, "dense_rank", "dense_score"),
        (sparse_results, sparse_weight, "sparse_rank", "sparse_score"),
    ]
    if graph_results:
        channels.append((graph_results, graph_weight, "graph_rank", "graph_score"))

    # Raw score vẫn được giữ trong metadata để quan sát, nhưng chỉ contribution
    # ``weight / (k + rank)`` tham gia thứ tự fused cuối cùng.
    for channel, weight, rank_key, score_key in channels:
        if weight <= 0:
            continue
        seen_ids: set[str] = set()
        rank = 0
        for document in channel:
            document_id = str(document.get("id", ""))
            if not document_id or document_id in seen_ids:
                continue
            seen_ids.add(document_id)
            rank += 1
            scores[document_id] = scores.get(document_id, 0.0) + weight / (k + rank)
            if document_id not in documents:
                documents[document_id] = dict(document)
            elif "graph_evidence" in document and "graph_evidence" not in documents[document_id]:
                documents[document_id]["graph_evidence"] = document["graph_evidence"]
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
