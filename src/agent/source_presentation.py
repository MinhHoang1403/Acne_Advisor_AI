"""Tạo source label thân thiện nhưng vẫn giữ raw traceability IDs."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FILE_SOURCE_DISPLAY_NAMES = {
    "web_raw_dataset.json": "Bộ dữ liệu kiến thức mụn",
    "PIIS0190962223033893.pdf": "Tài liệu chuyên môn về điều trị mụn",
    "acne-vulgaris-management-pdf-66142088866501.pdf": "Hướng dẫn quản lý mụn trứng cá",
    "qd_4416_cut.pdf": "Tài liệu tiếng Việt về mụn trứng cá",
}

SOURCE_TYPE_ORDER = {
    "document": 1,
    "dataset": 1,
    "other": 2,
}

SOURCE_NORMALIZATION_VERSION = "source_normalization_v1"
_KNOWN_SOURCE_IDS = {
    source_id.casefold(): source_id
    for source_id in FILE_SOURCE_DISPLAY_NAMES
}
_SOURCE_FILENAME_RE = re.compile(r"(?<![\w-])([\wÀ-ỹ][\wÀ-ỹ_()\-]{0,180}\.(?:pdf|json))(?![\w-])", re.IGNORECASE)
_GENERIC_SOURCE_LABEL_RE = re.compile(r"\b(?:tài\s+liệu|tai\s+lieu|document)\s+\d+\b", re.IGNORECASE)
_UNATTRIBUTED_SOURCE_CLAIM_RE = re.compile(
    r"\b(?:the|this)\s+(?:guideline|document)\s+(?:says|states)\b|"
    r"\b(?:theo\s+)?(?:hướng\s+dẫn|tài\s+liệu)\s+(?:này|đó)\s+(?:cho\s+biết|nêu|nói)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceValidationResult:
    """Kết quả source validation theo request, dùng cho internal diagnostics."""

    answer: str
    removed_mentions: tuple[str, ...]
    allowlist_source_ids: tuple[str, ...]


def build_source_metadata(
    sources: list[Any] | None,
    contexts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Tạo display metadata ổn định cho các source vừa được retrieval.

    ``source_id`` giữ raw backend identifier cho debugging/traceability;
    ``display_name`` là field UI nên hiển thị.
    """

    context_by_id: dict[str, dict[str, Any]] = {}
    for ctx in contexts or []:
        source_id = _source_id_from_context(ctx)
        if source_id and source_id not in context_by_id:
            context_by_id[source_id] = ctx

    ordered_ids: list[str] = []
    for source in sources or []:
        source_id = _source_id_from_value(source)
        if source_id and source_id not in ordered_ids:
            ordered_ids.append(source_id)
    for source_id in context_by_id:
        if source_id not in ordered_ids:
            ordered_ids.append(source_id)

    entries = [_source_entry(source_id, context_by_id.get(source_id, {})) for source_id in ordered_ids]
    entries.sort(key=lambda item: (SOURCE_TYPE_ORDER.get(item["source_type"], 99), item["display_name"].casefold(), item["source_id"]))
    return entries


def build_source_allowlist(
    sources: list[Any] | None,
    contexts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Trả canonical source allowlist mà answer hiện tại được phép trích dẫn."""

    return build_source_metadata(sources, contexts)


def normalize_source_identifier(value: Any) -> str:
    """Chuẩn hóa source ID qua path separator, case và Unicode form."""

    if isinstance(value, dict):
        raw = _first_text(
            value.get("source_id"),
            value.get("source_file"),
            value.get("source_path"),
            value.get("display_name"),
        ) or ""
    else:
        raw = str(value or "").strip()
    if not raw:
        return ""
    normalized = unicodedata.normalize("NFC", raw.strip().strip("`'\""))
    path_like = normalized.replace("\\", "/")
    filename = path_like.rsplit("/", 1)[-1].strip()
    if not filename:
        return ""
    return _KNOWN_SOURCE_IDS.get(filename.casefold(), filename)


def is_source_request(question: str) -> bool:
    """Nhận diện khi user hỏi về tài liệu được retrieval thay vì hỏi fact."""

    folded = _fold_source_text(question)
    markers = (
        "nguon nao",
        "tai lieu nao",
        "xem nguon nao",
        "xem tai lieu nao",
        "theo kho du lieu",
        "nguon tham khao",
    )
    return any(marker in folded for marker in markers)


def build_grounded_source_answer(question: str, allowlist: list[dict[str, Any]] | None) -> str:
    """Trả lời câu hỏi về nguồn chỉ từ retrieval allowlist của request hiện tại."""

    entries = [entry for entry in (allowlist or []) if entry.get("source_id")]
    if not entries:
        return "Tài liệu hiện được truy hồi chưa cung cấp nguồn phù hợp để trả lời câu hỏi này."

    labels = [str(entry.get("display_name") or entry["source_id"]).strip() for entry in entries]
    return (
        "Các nguồn đang được truy hồi cho câu hỏi này:\n"
        + "\n".join(f"- {label}" for label in labels)
    )


def validate_answer_source_mentions(
    answer: str,
    allowlist: list[dict[str, Any]] | None,
) -> SourceValidationResult:
    """Bỏ source label không có trong retrieval allowlist của response.

    Hàm chủ ý bảo thủ và không đoán source thay thế. Nó chỉ bỏ label không được
    hỗ trợ hoặc đổi document label đánh số chung thành cách gọi trung tính về
    context đã retrieval.
    """

    allowed_ids = tuple(str(entry.get("source_id") or "") for entry in allowlist or [] if entry.get("source_id"))
    allowed = {
        _source_match_key(candidate)
        for entry in allowlist or []
        for candidate in _source_aliases(entry)
        if candidate
    }
    removed: list[str] = []
    lines: list[str] = []
    for raw_line in str(answer or "").splitlines():
        line = raw_line
        generic_matches = _GENERIC_SOURCE_LABEL_RE.findall(line)
        if generic_matches:
            removed.extend(generic_matches)
            if line.lstrip().startswith("|"):
                # Không thể attribution an toàn cho bảng source đánh số; giữ bảng
                # sẽ làm user nhìn thấy source label không có trong allowlist.
                continue
            line = _GENERIC_SOURCE_LABEL_RE.sub("tài liệu đã truy hồi", line)

        # Claim chung về guideline/document không tên vẫn tạo attribution. Giữ
        # câu clinical nhưng đổi attribution thành trung tính để không trình bày
        # source reference chưa được kiểm chứng.
        unmatched_claim = _UNATTRIBUTED_SOURCE_CLAIM_RE.search(line)
        if unmatched_claim:
            removed.append(unmatched_claim.group(0))
            line = _UNATTRIBUTED_SOURCE_CLAIM_RE.sub("Theo nội dung đang được truy hồi", line)

        def replace_filename(match: re.Match[str]) -> str:
            mention = match.group(1)
            if _source_match_key(mention) in allowed:
                return mention
            removed.append(mention)
            return "tài liệu đã truy hồi"

        line = _SOURCE_FILENAME_RE.sub(replace_filename, line)
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return SourceValidationResult(
        answer=cleaned,
        removed_mentions=tuple(dict.fromkeys(removed)),
        allowlist_source_ids=allowed_ids,
    )


def display_names_for_sources(
    sources: list[Any] | None,
    contexts: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Helper tương thích cho response cũ có ``sources: list[str]``."""

    return [entry["display_name"] for entry in build_source_metadata(sources, contexts)]


def _source_entry(source_id: str, context: dict[str, Any]) -> dict[str, Any]:
    document_title = _first_text(
        context.get("document_title"),
        context.get("title"),
        context.get("source_title"),
        _metadata_value(context, "document_title"),
        _metadata_value(context, "title"),
    )
    source_type = _source_type(source_id, context)
    source_path = _first_text(context.get("source_path"), _metadata_value(context, "source_path"))
    display_name = _display_name(source_id, document_title=document_title, source_type=source_type)
    return {
        "source_id": source_id,
        "canonical_filename": Path(source_id.replace("\\", "/")).name if source_type in {"document", "dataset"} else None,
        "source_type": source_type,
        "source_path": source_path,
        "document_title": document_title,
        "display_name": display_name,
        "chunk_id": _first_text(context.get("chunk_id"), _metadata_value(context, "chunk_id")),
        "page": context.get("page") or _metadata_value(context, "page"),
        "origin": _first_text(context.get("source_type"), _metadata_value(context, "source_type")) or source_type,
    }


def _display_name(source_id: str, *, document_title: str | None, source_type: str) -> str:
    filename = Path(source_id.replace("\\", "/")).name
    if filename in FILE_SOURCE_DISPLAY_NAMES:
        return FILE_SOURCE_DISPLAY_NAMES[filename]
    if document_title:
        return document_title
    if filename:
        stem = Path(filename).stem
        label = re.sub(r"[_-]+", " ", stem).strip()
        label = re.sub(r"\s+", " ", label)
        if label:
            return label.title()
    return "Nguồn kiến thức nội bộ"


def _source_id_from_context(context: dict[str, Any]) -> str:
    return normalize_source_identifier(_first_text(
        context.get("source_file"),
        context.get("source_id"),
        context.get("source_path"),
        _metadata_value(context, "source_file"),
        _metadata_value(context, "source_id"),
        _metadata_value(context, "source_path"),
    ))


def _source_id_from_value(source: Any) -> str:
    if isinstance(source, dict):
        raw = _first_text(source.get("source_id"), source.get("source_file"), source.get("source_path"), source.get("display_name")) or ""
    else:
        raw = str(source or "").strip()
    return normalize_source_identifier(raw) if raw else ""


def _source_type(source_id: str, context: dict[str, Any]) -> str:
    explicit = _first_text(context.get("source_type"), _metadata_value(context, "source_type"))
    if explicit:
        if explicit == "web_json":
            return "dataset"
        if explicit in {"document", "dataset", "other"}:
            return explicit
    if source_id.casefold().endswith(".json"):
        return "dataset"
    if Path(source_id.replace("\\", "/")).suffix:
        return "document"
    return "other"


def _metadata_value(context: dict[str, Any], key: str) -> Any:
    metadata = context.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _source_aliases(entry: dict[str, Any]) -> set[str]:
    values = {
        str(entry.get("source_id") or ""),
        str(entry.get("canonical_filename") or ""),
        str(entry.get("display_name") or ""),
        str(entry.get("document_title") or ""),
        str(entry.get("source_path") or ""),
    }
    return {value for value in values if value.strip()}


def _source_match_key(value: Any) -> str:
    return _fold_source_text(normalize_source_identifier(value) or str(value or ""))


def _fold_source_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("đ", "d").replace("Đ", "D").casefold()
    return re.sub(r"\s+", " ", text).strip()


__all__ = [
    "SOURCE_NORMALIZATION_VERSION",
    "SourceValidationResult",
    "build_grounded_source_answer",
    "build_source_allowlist",
    "build_source_metadata",
    "display_names_for_sources",
    "is_source_request",
    "normalize_source_identifier",
    "validate_answer_source_mentions",
]
