"""Khai báo contract BM25 dùng chung khi indexing và truy vấn Qdrant.

Project chỉ chuẩn bị cấu hình preprocessing, tham số BM25 và ``Document`` mà
Qdrant hiểu được. Qdrant mới là component tạo sparse representation, lưu vector
``bm25``, áp dụng collection IDF và thực thi BM25 search.

``reference_bm25_score()`` là phép tính Python minh bạch cho unit test đối
chiếu công thức. Hàm đó không nằm trong request path và không phải BM25 search
engine của ứng dụng.

Điểm thường cần chỉnh sửa:
- ``BM25_K1`` và ``BM25_B``: TF saturation và document-length normalization.
- ``bm25_config()``: contract preprocessing gửi cho Qdrant.
- Không chỉnh ``reference_bm25_score()`` để tune runtime retrieval.
"""

from __future__ import annotations

import math
from collections import Counter

from qdrant_client import models


BM25_CONTRACT_ID = "qdrant_native_bm25_word_language_none"
BM25_MODEL = "qdrant/bm25"
BM25_VECTOR_NAME = "bm25"
# ``k1`` điều khiển mức bão hòa của term frequency; ``b`` điều khiển mức
# normalization theo độ dài document. Đây là engineering parameters của index,
# không phải ngưỡng chất lượng y khoa hay các giá trị được tuyên bố là tối ưu.
BM25_K1 = 1.2
BM25_B = 0.75
# Giá trị avg document length được gửi cho Qdrant trong cùng contract.
BM25_AVG_LEN = 256.0
BM25_TOKENIZER = models.TokenizerType.WORD
BM25_LANGUAGE = "none"


def bm25_config() -> models.Bm25Config:
    """Tạo một cấu hình document/query thống nhất để Qdrant thực thi BM25."""

    return models.Bm25Config(
        k=BM25_K1,
        b=BM25_B,
        avg_len=BM25_AVG_LEN,
        tokenizer=BM25_TOKENIZER,
        language=BM25_LANGUAGE,
        lowercase=True,
        ascii_folding=False,
    )


def bm25_document(text: str) -> models.Document:
    """Bọc text và BM25 contract thành ``Document`` cho Qdrant inference."""

    return models.Document(text=text, model=BM25_MODEL, options=bm25_config())


def bm25_sparse_vector_config() -> models.SparseVectorParams:
    """Yêu cầu Qdrant dùng IDF được tính từ collection khi truy vấn."""

    return models.SparseVectorParams(modifier=models.Modifier.IDF)


def reference_bm25_score(
    *,
    query_terms: list[str],
    document_terms: list[str],
    document_frequencies: dict[str, int],
    document_count: int,
    average_document_length: float,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> float:
    """Tính BM25 tham chiếu để unit test đối chiếu, không chạy trong runtime.

    Với term ``t`` và document ``D``:

    ``IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))``

    contribution của term là ``IDF(t) * tf(t,D) * (k1 + 1)`` chia cho
    ``tf(t,D) + k1 * (1 - b + b * |D| / avgdl)``.

    Mapping trong code: ``document_count`` là ``N``; ``document_frequencies``
    cung cấp ``df(t)``; ``frequencies[term]`` là ``tf(t,D)``; và
    ``length_ratio`` là ``|D| / avgdl``. Qdrant thực hiện phép tính tương ứng
    trong production search; hàm này chỉ cung cấp kết quả kiểm tra độc lập.
    """

    if document_count <= 0 or average_document_length <= 0:
        raise ValueError("Corpus size and average document length must be positive")
    frequencies = Counter(document_terms)
    length_ratio = len(document_terms) / average_document_length
    score = 0.0
    for term in query_terms:
        tf = frequencies[term]
        if tf <= 0:
            continue
        df = document_frequencies.get(term, 0)
        idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
        numerator = tf * (k1 + 1.0)
        denominator = tf + k1 * (1.0 - b + b * length_ratio)
        score += idf * numerator / denominator
    return score


__all__ = [
    "BM25_AVG_LEN",
    "BM25_B",
    "BM25_CONTRACT_ID",
    "BM25_K1",
    "BM25_LANGUAGE",
    "BM25_MODEL",
    "BM25_TOKENIZER",
    "BM25_VECTOR_NAME",
    "bm25_config",
    "bm25_document",
    "bm25_sparse_vector_config",
    "reference_bm25_score",
]
