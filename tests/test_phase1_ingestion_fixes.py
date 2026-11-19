from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.knowledge_build import parse_args
from src.ingestion import pipeline
from src.ingestion.chunking import structural_chunks
from src.ingestion.domain_metadata import enrich_domain_metadata
from src.ingestion.json_loader import load_web_json_documents_with_stats
from src.ingestion.normalization import normalize_parsed_text
from src.ingestion.parser import (
    _extract_page_number,
    _needs_next_page_continuation,
    _parse_markdown,
)


def test_normalize_parsed_text_cleans_pdf_ligatures_invisible_and_control_chars():
    raw = "\ufeffﬁrst-line\u00a0in\ufb02ammation\u00adtherapy\u200b\x07\n> © NICE 2026. All rights reserved.\n> Page 3 of 12\nĐoa\u0300n chuẩn."
    normalized = normalize_parsed_text(raw)
    assert normalized == "first-line inflammationtherapy\nĐoàn chuẩn."


def test_parser_handles_bom_markdown_and_safe_pdf_page_labels(tmp_path: Path):
    md_path = tmp_path / "bom.md"
    md_path.write_bytes("\ufeff# Điều trị mụn\n\nNội dung chuẩn.".encode("utf-8"))
    units = _parse_markdown(md_path)
    assert len(units) == 1
    assert units[0].text.startswith("# Điều trị mụn")

    assert _extract_page_number({"page_label": "iv"}) == 0
    assert _extract_page_number({"page": 5, "page_label": "iv"}) == 5
    assert _extract_page_number({"page_label": "12"}) == 12
    assert _needs_next_page_continuation("Sử dụng adapalene **rất hiệu quả.**") is False
    assert _needs_next_page_continuation("Xem chi tiết (bảng 1).") is False
    assert _needs_next_page_continuation("Tiếp tục duy trì thuốc") is True


def test_json_loader_enforces_min_raw_text_chars_and_handles_bom(tmp_path: Path):
    json_path = tmp_path / "bom.json"
    payload = [
        {"source_url": "https://example.test/short", "raw_text": "ngắn"},
        {"source_url": "https://example.test/long", "raw_text": "Nội dung đủ dài để nạp vào hệ thống."},
    ]
    json_path.write_bytes(("\ufeff" + json.dumps(payload, ensure_ascii=False)).encode("utf-8"))

    docs, stats = load_web_json_documents_with_stats(json_path, min_raw_text_chars=10)
    assert stats == {"total_records": 2, "skipped_records": 1}
    assert len(docs) == 1
    assert docs[0]["metadata"]["source_url"] == "https://example.test/long"


def test_chunking_repeats_table_headers_on_split_and_handles_adjacent_headings():
    table_text = (
        "| Hoạt chất | Liều dùng | Ghi chú |\n"
        "|---|---|---|\n"
        "| Adapalene | Bôi tối 1 lần | Tránh thai kỳ |\n"
        "| Benzoyl peroxide | Bôi 1-2 lần/ngày | Có thể tẩy màu vải |\n"
        "| Clindamycin | Phối hợp BP | Tránh đơn trị kháng sinh |\n"
    )
    chunks = structural_chunks(table_text, max_chars=140)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.text.startswith("| Hoạt chất | Liều dùng | Ghi chú |\n|---|---|---|")

    adjacent = "# Nhóm thuốc bôi\n## Retinoid\nAdapalene dùng buổi tối."
    adjacent_chunks = structural_chunks(adjacent)
    assert adjacent_chunks[0].section_path == ("Nhóm thuốc bôi", "Retinoid")
    assert adjacent_chunks[0].text == "Adapalene dùng buổi tối."


def test_domain_metadata_avoids_false_side_effect_for_khong():
    clean_meta = enrich_domain_metadata("Không nên phối hợp kháng sinh bôi đơn trị kéo dài.")
    assert "side_effect" not in clean_meta["query_intent_hint"]

    dry_meta = enrich_domain_metadata("Bệnh nhân có thể gặp tình trạng da khô nhẹ khi mới bôi.")
    assert "side_effect" in dry_meta["query_intent_hint"]


def test_knowledge_build_cli_supports_custom_source_and_empty_bootstrap():
    args = parse_args(
        ["build", "--source", "custom_data", "--activate", "--allow-empty-bootstrap"]
    )
    assert args.source == Path("custom_data")
    assert args.activate is True
    assert args.allow_empty_bootstrap is True

    val_args = parse_args(["validate", "--source", "custom_data", "--offline"])
    assert val_args.source == Path("custom_data")
    assert val_args.offline is True


@pytest.mark.asyncio
async def test_verify_empty_bootstrap_refuses_non_empty_qdrant(monkeypatch):
    class NonEmptyQdrant:
        async def get_collections(self):
            return SimpleNamespace(collections=[SimpleNamespace(name="acne_knowledge")])

        async def get_aliases(self):
            return SimpleNamespace(aliases=[])

        async def close(self):
            return None

    monkeypatch.setattr(pipeline, "AsyncQdrantClient", lambda **_kwargs: NonEmptyQdrant())
    with pytest.raises(RuntimeError, match="Empty bootstrap refused"):
        await pipeline._verify_empty_bootstrap_state()


@pytest.mark.asyncio
async def test_structured_curated_json_preserves_per_record_provenance_and_metadata(
    tmp_path: Path,
):
    from src.ingestion.build import BuildIdentity, compile_knowledge
    from src.ingestion.parser import load_or_parse_source
    from src.ingestion.provenance import validate_provenance
    from src.ingestion.source_manifest import CanonicalSource, sha256_file

    dataset_path = tmp_path / "Acne_data_source.json"
    records = [
        {
            "id": "CHUNK_demo_001",
            "text": "Adapalene là retinoid bôi ngoài da được khuyến cáo cho mụn trứng cá.",
            "embedding_text": "EuroGuiDerm Guideline | Treatment | Adapalene là retinoid bôi ngoài da.",
            "metadata": {
                "source_id": "SRC_euroguiderm_2025",
                "document_id": "DOC_euroguiderm_2025",
                "chunk_id": "CHUNK_demo_001",
                "title": "EuroGuiDerm Guideline for the Treatment of Acne - Update 2025",
                "publisher": "European Dermatology Forum",
                "source_type": "pdf",
                "source_url": "https://example.test/euroguiderm.pdf#page=12",
                "published_year": 2025,
                "accessed_at": "2026-10-08",
                "document_hash": "a" * 64,
                "page": 12,
                "page_end": 12,
                "section_title": "Topical retinoids",
                "content_kind": "text",
                "language": "vi",
                "scope": "acne_core",
                "acne_type": ["Comedonal acne"],
                "category": ["Treatment"],
            },
        }
    ]
    dataset_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    source = CanonicalSource(
        source_id="acne_data_source",
        title="Curated Dataset",
        authority="Multi-publisher",
        source_type="curated_structured_acne_corpus",
        version_date="2026-11-19",
        original_url="https://example.test/dataset",
        local_filename="Acne_data_source.json",
        media_type="application/json",
        sha256=sha256_file(dataset_path),
    )

    artifact, cache_hit = await load_or_parse_source(
        source,
        source_dir=tmp_path,
        cache_root=tmp_path / "parsed",
    )
    assert cache_hit is False
    compiled = compile_knowledge(
        (source,),
        {source.source_id: artifact},
        BuildIdentity("1234567890abcdef1234", "s", "t", "c"),
    )
    assert len(compiled.records) == 1
    rec = compiled.records[0]
    assert validate_provenance(rec) == []
    assert rec["source_id"] == "SRC_euroguiderm_2025"
    assert rec["source_title"] == "EuroGuiDerm Guideline for the Treatment of Acne - Update 2025"
    assert rec["source_authority"] == "European Dermatology Forum"
    assert rec["source_url"] == "https://example.test/euroguiderm.pdf#page=12"
    assert rec["page_start"] == 12
    assert rec["section_path"] == ["Topical retinoids"]
    assert rec["language"] == "vi"
    assert rec["acne_type"] == ["Comedonal acne"]
    assert "adapalene" in rec["active_ingredient"]
