from __future__ import annotations

import pytest

from src.ingestion.embedding import EMBEDDING_DIMENSIONS, EmbeddingCache
from src.ingestion import index as ingestion_index
from src.ingestion import pipeline
from src.ingestion.source_manifest import CanonicalSource
from src.resilience.exceptions import ProviderUnavailableError


@pytest.mark.asyncio
async def test_cache_inspection_reports_parsed_miss_without_repairing_it(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = CanonicalSource(
        source_id="missing-cache-source",
        title="Missing cache source",
        authority="Test authority",
        source_type="document",
        version_date="2026-08-17",
        original_url="https://example.test/source",
        local_filename="source.pdf",
        media_type="application/pdf",
        sha256="a" * 64,
    )

    async def forbidden_parser(*args, **kwargs):
        raise AssertionError("cache inspection must not invoke parsing")

    monkeypatch.setattr(pipeline, "load_source_manifest", lambda path: (source,))
    monkeypatch.setattr(pipeline, "verify_source_files", lambda sources, path: None)
    monkeypatch.setattr(pipeline, "verify_manifest_support_files", lambda sources, path: None)
    monkeypatch.setattr(pipeline, "load_or_parse_source", forbidden_parser)

    result = await pipeline.inspect_embedding_cache_reuse(
        source_dir=tmp_path,
        source_manifest_path=tmp_path / "manifest.yaml",
        taxonomy_path=tmp_path / "taxonomy.yaml",
        parsed_cache=tmp_path / "parsed",
        embedding_cache=tmp_path / "embeddings",
    )

    assert result["passed"] is False
    assert result["parsed"] == {
        "hits": 0,
        "misses": 1,
        "total": 1,
        "missing_or_invalid_source_ids": ["missing-cache-source"],
    }
    assert result["knowledge_embeddings"]["inspected"] is False
    assert result["entity_embeddings"]["inspected"] is False
    assert result["provider_calls"] == 0
    assert not (tmp_path / "parsed").exists()
    assert not (tmp_path / "embeddings").exists()


@pytest.mark.asyncio
async def test_cache_inspection_reports_corrupt_parsed_cache_without_rewriting_it(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = CanonicalSource(
        source_id="corrupt-cache-source",
        title="Corrupt cache source",
        authority="Test authority",
        source_type="document",
        version_date="2026-08-17",
        original_url="https://example.test/source",
        local_filename="source.pdf",
        media_type="application/pdf",
        sha256="b" * 64,
    )
    cache_path = pipeline.artifact_path(tmp_path / "parsed", source)
    cache_path.parent.mkdir(parents=True)
    corrupt_bytes = b"\xff\xfe\x00\x80"
    cache_path.write_bytes(corrupt_bytes)

    async def forbidden_parser(*args, **kwargs):
        raise AssertionError("cache inspection must not repair corrupt parser cache")

    monkeypatch.setattr(pipeline, "load_source_manifest", lambda path: (source,))
    monkeypatch.setattr(pipeline, "verify_source_files", lambda sources, path: None)
    monkeypatch.setattr(pipeline, "verify_manifest_support_files", lambda sources, path: None)
    monkeypatch.setattr(pipeline, "load_or_parse_source", forbidden_parser)

    result = await pipeline.inspect_embedding_cache_reuse(
        source_dir=tmp_path,
        source_manifest_path=tmp_path / "manifest.yaml",
        taxonomy_path=tmp_path / "taxonomy.yaml",
        parsed_cache=tmp_path / "parsed",
        embedding_cache=tmp_path / "embeddings",
    )

    assert result["passed"] is False
    assert result["parsed"]["missing_or_invalid_source_ids"] == [source.source_id]
    assert cache_path.read_bytes() == corrupt_bytes
    assert not (tmp_path / "embeddings").exists()


@pytest.mark.asyncio
async def test_embedding_resolution_retries_finitely_and_checkpoints_success(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_embed(texts: list[str], *, api_key: str):
        nonlocal calls
        calls += 1
        assert api_key == "redacted-test-key"
        if calls == 1:
            raise ProviderUnavailableError("rate limited")
        return [[float(index)] * EMBEDDING_DIMENSIONS for index, _ in enumerate(texts)]

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(ingestion_index, "embed_documents", fake_embed)
    monkeypatch.setattr(ingestion_index.asyncio, "sleep", fake_sleep)
    cache = EmbeddingCache(tmp_path)

    vectors, stats = await ingestion_index.resolve_embeddings(
        ["one", "two"],
        cache=cache,
        api_key="redacted-test-key",
        batch_size=2,
        batch_delay_seconds=0,
        max_retries=2,
    )

    assert len(vectors) == 2
    assert stats == {
        "cache_hits": 0,
        "cache_misses": 2,
        "provider_calls": 1,
        "retry_count": 1,
    }
    assert sleeps == [30.0]
    assert cache.get("one") == vectors[0]
    assert cache.get("two") == vectors[1]


@pytest.mark.asyncio
async def test_embedding_resolution_stops_after_retry_limit(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_rate_limited(texts: list[str], *, api_key: str):
        raise ProviderUnavailableError("rate limited")

    async def no_wait(seconds: float) -> None:
        return None

    monkeypatch.setattr(ingestion_index, "embed_documents", always_rate_limited)
    monkeypatch.setattr(ingestion_index.asyncio, "sleep", no_wait)

    with pytest.raises(ProviderUnavailableError):
        await ingestion_index.resolve_embeddings(
            ["one"],
            cache=EmbeddingCache(tmp_path),
            api_key="redacted-test-key",
            batch_size=1,
            batch_delay_seconds=0,
            max_retries=2,
        )
