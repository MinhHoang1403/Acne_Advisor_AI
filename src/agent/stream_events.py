"""Kênh phát sự kiện thời gian thực (SSE) theo phạm vi từng request qua ContextVar.

Thiết kế phi xâm lấn (non-invasive):
- Khi chạy ``POST /chat`` thông thường, unit test hoặc đánh giá RAGAS, ``_STREAM_QUEUE``
  có giá trị ``None`` -> toàn bộ hàm phát sự kiện rút ngắn tức thì (0 chi phí).
- Khi chạy ``POST /chat/stream``, API mở một ``asyncio.Queue`` gắn với ngữ cảnh bất
  đồng bộ của request để đẩy tức thì:
  1. Tiến độ từng chặng xử lý (``stage``): kiểm tra an toàn, truy xuất 3 kênh, chấm điểm,
     tổng hợp câu trả lời, đối chiếu trích dẫn.
  2. Từng cụm từ sinh trực tiếp từ LLM (``delta``) giúp giảm thời gian chờ chữ đầu tiên
     (TTFT) từ ~12 giây xuống còn ~3–4 giây, trong khi chặng ``complete`` cuối cùng vẫn
     đảm bảo câu trả lời đã qua ``answer_verifier`` kiểm duyệt đầy đủ.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token
from typing import Any

_STREAM_QUEUE: ContextVar[asyncio.Queue[dict[str, Any]] | None] = ContextVar(
    "acne_chat_stream_queue",
    default=None,
)


def begin_chat_stream_scope() -> tuple[
    asyncio.Queue[dict[str, Any]],
    Token[asyncio.Queue[dict[str, Any]] | None],
]:
    """Khởi tạo hàng đợi SSE riêng biệt cho request hiện hành."""

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=512)
    token = _STREAM_QUEUE.set(queue)
    return queue, token


def bind_chat_stream_queue(
    queue: asyncio.Queue[dict[str, Any]],
) -> Token[asyncio.Queue[dict[str, Any]] | None]:
    """Gắn hàng đợi SSE đã khởi tạo vào ContextVar của coroutine worker (không vắt qua yield)."""

    return _STREAM_QUEUE.set(queue)


def reset_chat_stream_scope(token: Token[asyncio.Queue[dict[str, Any]] | None]) -> None:
    """Khôi phục ContextVar khi kết thúc stream request."""

    _STREAM_QUEUE.reset(token)


def is_chat_stream_active() -> bool:
    """Kiểm tra request hiện tại có đang bật chế độ SSE streaming hay không."""

    return _STREAM_QUEUE.get() is not None


def emit_stream_stage(stage: str, label: str, **extra: Any) -> None:
    """Đẩy thông báo tiến độ chặng xử lý (best-effort, không bao giờ làm gián đoạn luồng chính)."""

    queue = _STREAM_QUEUE.get()
    if queue is None:
        return
    payload: dict[str, Any] = {"stage": stage, "label": label, **extra}
    try:
        queue.put_nowait({"event": "stage", "data": payload})
    except asyncio.QueueFull:
        return


def emit_stream_delta(delta: str) -> None:
    """Đẩy mảnh văn bản đang sinh trực tiếp từ LLM tới trình duyệt."""

    if not delta:
        return
    queue = _STREAM_QUEUE.get()
    if queue is None:
        return
    try:
        queue.put_nowait({"event": "delta", "data": {"delta": delta}})
    except asyncio.QueueFull:
        return


def emit_stream_reset() -> None:
    """Thông báo trình duyệt xóa bản nháp dở dang khi provider retry hoặc chuyển model dự phòng."""

    queue = _STREAM_QUEUE.get()
    if queue is None:
        return
    try:
        queue.put_nowait({"event": "reset", "data": {}})
    except asyncio.QueueFull:
        return


__all__ = [
    "begin_chat_stream_scope",
    "bind_chat_stream_queue",
    "emit_stream_delta",
    "emit_stream_reset",
    "emit_stream_stage",
    "is_chat_stream_active",
    "reset_chat_stream_scope",
]
