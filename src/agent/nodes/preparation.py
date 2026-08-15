"""Chuẩn bị request không làm mất nghĩa và conversation có giới hạn."""

from __future__ import annotations

import os

from src.agent.state import ClinicalState


async def prepare_request_node(state: ClinicalState) -> dict[str, object]:
    """Chuẩn hóa whitespace và tạo một biểu diễn history có giới hạn."""

    question = " ".join(str(state.get("user_question") or "").split())
    max_messages = _bounded_int("MAX_CONVERSATION_HISTORY_MESSAGES", 10, 0, 20)
    max_chars = _bounded_int("MAX_HISTORY_MESSAGE_CHARS", 1000, 1, 4000)
    history: list[dict[str, str]] = []
    for item in list(state.get("conversation_history") or [])[-max_messages:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user")
        content = " ".join(str(item.get("content") or "").split())[:max_chars]
        if content:
            history.append({"role": role, "content": content})
    return {
        "normalized_question": question,
        "standalone_question": question,
        "conversation_context": {"messages": history, "message_count": len(history)},
    }


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return min(maximum, max(minimum, value))


__all__ = ["prepare_request_node"]
