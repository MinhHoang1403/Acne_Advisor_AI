import pytest

from src.agent.llm import ollama_client
from src.agent.llm.ollama_client import build_ollama_chat_payload, generate_ollama_response


def test_ollama_payload_uses_bounded_generation_options(monkeypatch):
    monkeypatch.setenv("OLLAMA_THINK", "false")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "30m")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "192")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.2")
    monkeypatch.setenv("OLLAMA_TOP_K", "20")
    monkeypatch.setenv("OLLAMA_TOP_P", "0.9")

    payload = build_ollama_chat_payload(
        model="qwen3:8b",
        messages=[{"role": "user", "content": "mụn là gì?"}],
        temperature=0.7,
    )

    assert payload["model"] == "qwen3:8b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["keep_alive"] == "30m"
    assert payload["options"] == {
        "num_predict": 192,
        "num_ctx": 4096,
        "temperature": 0.2,
        "top_k": 20,
        "top_p": 0.9,
    }
    assert "num_predict" not in {key for key in payload if key != "options"}


def test_ollama_payload_default_reserves_room_for_complete_structured_answers(monkeypatch):
    monkeypatch.delenv("OLLAMA_NUM_PREDICT", raising=False)

    payload = build_ollama_chat_payload(
        model="qwen3:8b",
        messages=[{"role": "user", "content": "So sánh adapalene và benzoyl peroxide."}],
    )

    assert payload["options"]["num_predict"] == 640


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    responses = []
    posted_payloads = []

    def __init__(self, *, timeout):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, _url, *, json):
        self.posted_payloads.append(json)
        return _FakeResponse(self.responses.pop(0))


@pytest.mark.asyncio
async def test_ollama_retries_one_compact_generation_after_truncation(monkeypatch):
    _FakeAsyncClient.responses = [
        {"done_reason": "length", "eval_count": 640, "message": {"content": "Dở dang"}},
        {"done_reason": "stop", "eval_count": 98, "message": {"content": "Câu trả lời hoàn chỉnh."}},
    ]
    _FakeAsyncClient.posted_payloads = []
    monkeypatch.setenv("OLLAMA_TRUNCATION_RETRY_ATTEMPTS", "1")
    monkeypatch.setattr(ollama_client.httpx, "AsyncClient", _FakeAsyncClient)

    answer = await generate_ollama_response("qwen3:8b", "System", "Câu hỏi gốc")

    assert answer == "Câu trả lời hoàn chỉnh."
    assert len(_FakeAsyncClient.posted_payloads) == 2
    assert _FakeAsyncClient.posted_payloads[1]["messages"][-1]["content"] == ollama_client._COMPACT_RETRY_INSTRUCTION


@pytest.mark.asyncio
async def test_ollama_marks_output_truncated_after_the_single_retry(monkeypatch):
    _FakeAsyncClient.responses = [
        {"done_reason": "length", "eval_count": 640, "message": {"content": "Lần một"}},
        {"done_reason": "num_predict", "eval_count": 640, "message": {"content": "Lần hai"}},
    ]
    _FakeAsyncClient.posted_payloads = []
    monkeypatch.setenv("OLLAMA_TRUNCATION_RETRY_ATTEMPTS", "1")
    monkeypatch.setattr(ollama_client.httpx, "AsyncClient", _FakeAsyncClient)

    answer = await generate_ollama_response("qwen3:8b", None, "Câu hỏi gốc")

    assert answer == "Lần hai\n...[truncated_generation]"
    assert len(_FakeAsyncClient.posted_payloads) == 2
