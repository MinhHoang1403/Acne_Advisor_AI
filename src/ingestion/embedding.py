"""Giữ Gemini Embedding 2 contract và cache vector theo content identity.

Project chuẩn bị request và kiểm tra shape của vector; Gemini là component thực
sự chạy embedding model. Cùng model/dimension contract được dùng cho document
khi indexing và query trong runtime. ``EmbeddingCache`` tránh gọi lại provider
khi text và toàn bộ contract không đổi; cache không thực hiện similarity search.

Muốn đổi embedding model/dimension/distance bắt đầu ở các hằng số ``EMBEDDING_*``
và phải xem đồng thời schema Qdrant trong ``src/ingestion/index.py``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.integrations.google_genai import embed_texts_sync


EMBEDDING_CONTRACT_ID = "google_gemini_embedding_2_3072_cosine"
EMBEDDING_PROVIDER = "google"
EMBEDDING_MODEL = "models/gemini-embedding-2"
EMBEDDING_DIMENSIONS = 3072
EMBEDDING_DISTANCE = "cosine"


@dataclass(frozen=True)
class EmbeddingContract:
    """Các thuộc tính phải cùng nhau xác định một vector tương thích."""

    provider: str = EMBEDDING_PROVIDER
    model: str = EMBEDDING_MODEL
    dimensions: int = EMBEDDING_DIMENSIONS
    distance: str = EMBEDDING_DISTANCE
    task_type: None = None

    def identity(self, text: str) -> str:
        """Hash canonical JSON của contract và text để định danh cache chính xác."""

        payload = {
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "distance": self.distance,
            "task_type": self.task_type,
            "text": text,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def embed_documents(texts: list[str], *, api_key: str) -> list[list[float]]:
    """Gửi knowledge texts cho Gemini và nhận Dense vectors đúng contract."""

    return embed_texts_sync(
        texts,
        model_name=EMBEDDING_MODEL,
        task_type=None,
        expected_dimensions=EMBEDDING_DIMENSIONS,
        output_dimensions=EMBEDDING_DIMENSIONS,
        api_key=api_key,
    )


class EmbeddingCache:
    """Cache một JSON cho mỗi vector, kiểm identity và dimension trước khi reuse.

    Cache chỉ thuộc knowledge preparation và có side effect trên filesystem.
    Ghi file tạm rồi replace để tránh để lại record dở dang khi process ngắt.
    """

    def __init__(self, root: Path, contract: EmbeddingContract | None = None) -> None:
        self.root = root
        self.contract = contract or EmbeddingContract()

    def get(self, text: str) -> list[float] | None:
        identity = self.contract.identity(text)
        path = self.root / f"{identity}.json"
        if not path.is_file():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            vector = record["vector"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return None
        if record.get("identity") != identity or not isinstance(vector, list):
            return None
        if len(vector) != self.contract.dimensions:
            return None
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in vector):
            return None
        return [float(value) for value in vector]

    def put(self, text: str, vector: list[float]) -> None:
        if len(vector) != self.contract.dimensions:
            raise ValueError("Embedding vector dimension does not match contract")
        identity = self.contract.identity(text)
        self.root.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {"identity": identity, "vector": vector}
        temporary = self.root / f".{identity}.tmp"
        final = self.root / f"{identity}.json"
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(final)


__all__ = [
    "EMBEDDING_CONTRACT_ID",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_DISTANCE",
    "EMBEDDING_MODEL",
    "EMBEDDING_PROVIDER",
    "EmbeddingCache",
    "EmbeddingContract",
    "embed_documents",
]
