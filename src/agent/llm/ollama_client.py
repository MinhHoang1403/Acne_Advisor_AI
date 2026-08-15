"""HTTP client cho Ollama local với payload và retry do truncation có giới hạn.

Ollama thực thi model inference. Module chỉ đóng gói messages/options, đặt timeout
và thử lại tối đa một lần khi provider báo output bị cắt. Truncation retry này
khác provider retry trong resilience layer: nó dùng instruction rút gọn để lấy
một câu hoàn chỉnh, không retry lỗi network/status.
"""

import logging
import os
import httpx
from typing import Any

from src.quality.safe_fallback import sanitize_fallback_reason

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_COMPACT_RETRY_INSTRUCTION = (
    "Câu trả lời trước đã bị cắt vì quá dài. Hãy trả lời lại từ đầu, ngắn gọn và hoàn chỉnh "
    "bằng tiếng Việt trong tối đa 160 từ. Giữ nguyên mọi yêu cầu về an toàn, thực thể, "
    "định dạng và số lượng mục của câu hỏi gốc; không mô tả quá trình suy luận."
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def build_ollama_chat_payload(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Tạo payload Ollama hữu hạn mà không log nội dung prompt."""

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": _env_bool("OLLAMA_THINK", False),
        "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "30m").strip() or "30m",
        "options": {
            "num_predict": _env_int("OLLAMA_NUM_PREDICT", 640),
            "num_ctx": _env_int("OLLAMA_NUM_CTX", 4096),
            "temperature": _env_float("OLLAMA_TEMPERATURE", temperature),
            "top_k": _env_int("OLLAMA_TOP_K", 20),
            "top_p": _env_float("OLLAMA_TOP_P", 0.9),
        },
    }
    return payload

async def list_ollama_models(timeout_seconds: float | None = None) -> list[str]:
    """Lấy tên model local; lỗi kết nối trả list rỗng để fallback bỏ qua Ollama."""
    try:
        timeout = timeout_seconds if timeout_seconds and timeout_seconds > 0 else 5.0
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            data = response.json()
            models = [model.get("name") for model in data.get("models", [])]
            return models
    except Exception as e:
        logger.warning("Could not connect to Ollama to list models: %s", sanitize_fallback_reason(e))
        return []

async def generate_ollama_response(
    model: str,
    system_prompt: str | None,
    prompt: str,
    temperature: float = 0.2,
    request_timeout: float | None = None,
) -> str:
    """Yêu cầu Ollama sinh text và thử rút gọn một lần nếu output bị cắt."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        timeout = request_timeout or float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "90"))
        # Chặn ở một compact retry để output dài không tạo vòng request vô hạn.
        retry_attempts = max(0, min(_env_int("OLLAMA_TRUNCATION_RETRY_ATTEMPTS", 1), 1))
        attempt_messages = messages
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(retry_attempts + 1):
                payload = build_ollama_chat_payload(
                    model=model,
                    messages=attempt_messages,
                    temperature=temperature,
                )
                response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                done_reason = data.get("done_reason")
                eval_count = data.get("eval_count")
                num_predict = payload.get("options", {}).get("num_predict")
                truncated = str(done_reason or "").lower() in {"length", "num_predict", "context_length"}
                logger.info(
                    "Ollama generation completed: model=%s attempt=%s done_reason=%s eval_count=%s num_predict=%s truncated=%s",
                    model,
                    attempt + 1,
                    done_reason,
                    eval_count,
                    num_predict,
                    truncated,
                )
                content = data.get("message", {}).get("content", "")
                if not truncated:
                    return content
                if attempt < retry_attempts:
                    logger.warning(
                        "Ollama generation reached output limit; retrying once with a compact-answer instruction: model=%s",
                        model,
                    )
                    attempt_messages = [*messages, {"role": "user", "content": _COMPACT_RETRY_INSTRUCTION}]
                    continue
                return f"{content}\n...[truncated_generation]"
    except httpx.ConnectError:
        raise ConnectionError("Model local hiện chưa khả dụng. Hãy mở Ollama rồi thử lại.")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise ValueError(
                f"Model Ollama '{model}' chưa có trong runtime local. "
                "Hãy provision model theo hướng dẫn cấu hình của dự án rồi thử lại."
            )
        raise
