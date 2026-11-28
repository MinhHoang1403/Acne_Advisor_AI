"""Thí nghiệm ablation: đo đóng góp của reranker và GraphRAG trên cùng bộ 100 ca RAGAS.

Ablation = lần lượt tắt từng thành phần, chạy lại cùng bộ câu hỏi và cùng judge, rồi so sánh
ghép cặp. Ba cấu hình lồng nhau (mỗi cấu hình chỉ khác cấu hình trước đúng một thành phần):

- ``dense_bm25``: Dense + BM25 → RRF → đóng gói (tắt reranker, tắt GraphRAG).
- ``dense_bm25_rerank``: thêm BGE Cross-Encoder reranker + lọc đuôi.
- ``full_graphrag``: cấu hình sản phẩm hiện hành (thêm kênh Neo4j, NEXT_CHUNK, anchor coverage).

Mọi cấu hình dùng **một judge cố định, tắt chuỗi dự phòng**, để chênh lệch điểm phản ánh hệ
thống chứ không phải việc đổi giám khảo. Cấu hình chỉ được áp dụng qua biến môi trường mà
runtime đã đọc sẵn (``GRAPH_RAG_ENABLED``, ``RERANKER_ENABLED``...), không sửa mã sản phẩm.
Pipeline fingerprint của từng cấu hình được kiểm tra và ghi vào kết quả để chứng minh cấu hình
thực sự đã được áp dụng.

Chạy CLI (mỗi cấu hình một tiến trình, có checkpoint riêng để chạy tiếp khi hết quota ngày):
    python -m evaluation.ablation run --config dense_bm25
    python -m evaluation.ablation run --config dense_bm25_rerank
    python -m evaluation.ablation run --config full_graphrag
So sánh:
    python -m evaluation.paired_comparison \
        --baseline evaluation/results/ablation/dense_bm25_rerank/ragas_evaluation_raw.json \
        --candidate evaluation/results/ablation/full_graphrag/ragas_evaluation_raw.json \
        --baseline-label "Dense+BM25+Reranker" --candidate-label "+GraphRAG" \
        --title "Ablation: đóng góp của GraphRAG" \
        --output evaluation/results/ablation/compare_graphrag.md
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, MutableMapping, Sequence
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ragas_runner import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_PROVIDER,
    DEFAULT_RESULTS_DIR,
    export_evaluation_artifacts,
    load_evaluation_dataset,
    run_system_on_dataset,
    score_records_with_ragas,
    summarize_evaluation,
)

ABLATION_RESULTS_DIR = DEFAULT_RESULTS_DIR / "ablation"

# Mỗi cấu hình chỉ đặt các biến môi trường mà runtime hiện hành đã hỗ trợ.
ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "dense_bm25": {
        "description": "Dense + BM25 → RRF → packer (tắt reranker, tắt GraphRAG)",
        "env": {"RERANKER_ENABLED": "false", "GRAPH_RAG_ENABLED": "false"},
        "expect": {"reranker": False, "graph_rag": False},
    },
    "dense_bm25_rerank": {
        "description": "Dense + BM25 → RRF → BGE reranker → packer (tắt GraphRAG)",
        "env": {"RERANKER_ENABLED": "true", "GRAPH_RAG_ENABLED": "false"},
        "expect": {"reranker": True, "graph_rag": False},
    },
    "full_graphrag": {
        "description": "Cấu hình sản phẩm: Dense + BM25 + Neo4j Graph → RRF → BGE reranker + NEXT_CHUNK → packer",
        "env": {
            "RERANKER_ENABLED": "true",
            "GRAPH_RAG_ENABLED": "true",
            "GRAPH_RAG_NEIGHBOR_EXPANSION_ENABLED": "true",
        },
        "expect": {"reranker": True, "graph_rag": True},
    },
}


def apply_ablation_config(
    name: str,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Đặt biến môi trường của cấu hình ``name``; trả về các giá trị đã đặt."""

    if name not in ABLATION_CONFIGS:
        raise ValueError(f"Cấu hình ablation không tồn tại: {name}. Chọn: {sorted(ABLATION_CONFIGS)}")
    target = os.environ if environ is None else environ
    overrides = dict(ABLATION_CONFIGS[name]["env"])
    for key, value in overrides.items():
        target[key] = value
    return overrides


def verify_applied_config(name: str, manifest: Mapping[str, Any]) -> None:
    """Fail-closed: dừng nếu pipeline manifest không phản ánh đúng cấu hình ablation đã chọn.

    Tránh trường hợp chạy cả ngày quota rồi mới phát hiện biến môi trường bị ghi đè.
    """

    expect = ABLATION_CONFIGS[name]["expect"]
    actual_reranker = bool((manifest.get("reranker") or {}).get("enabled"))
    actual_graph = bool((manifest.get("graph_rag") or {}).get("enabled"))
    if actual_reranker != expect["reranker"] or actual_graph != expect["graph_rag"]:
        raise RuntimeError(
            f"Cấu hình '{name}' chưa được áp dụng: reranker={actual_reranker} "
            f"(cần {expect['reranker']}), graph_rag={actual_graph} (cần {expect['graph_rag']})."
        )


async def run_ablation_config(
    name: str,
    *,
    judge_model: str,
    judge_provider: str,
    embedding_model: str,
    max_cases: int | None = None,
    verbose: bool = True,
) -> Path:
    """Chạy hệ thống + chấm RAGAS cho một cấu hình ablation và xuất báo cáo riêng."""

    from src.knowledge.versioning import resolve_active_knowledge_build_id
    from src.observability.versioning import (
        build_pipeline_version_manifest,
        compute_pipeline_fingerprint,
    )

    overrides = apply_ablation_config(name)
    manifest = build_pipeline_version_manifest()
    verify_applied_config(name, manifest)
    fingerprint = compute_pipeline_fingerprint(manifest)

    out_dir: Path = ABLATION_RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_meta, cases = load_evaluation_dataset(PROJECT_ROOT / DEFAULT_DATASET_PATH)
    if max_cases:
        cases = cases[:max_cases]

    system_model = os.getenv("GOOGLE_MODEL", "gemini-3.5-flash-lite")
    system_records = await run_system_on_dataset(
        cases,
        llm_provider="gemini",
        llm_model=system_model,
        allow_model_fallback=False,
        case_delay_seconds=1.5,
        checkpoint_path=out_dir / "system_checkpoint.json",
        verbose=verbose,
    )
    scored = await score_records_with_ragas(
        system_records,
        judge_model=judge_model,
        judge_provider=judge_provider,
        fallback_judge_models=(),
        embedding_model=embedding_model,
        checkpoint_path=out_dir / "scored_checkpoint.json",
        verbose=verbose,
    )
    summary = summarize_evaluation(
        scored,
        dataset_meta=dataset_meta,
        judge_model=judge_model,
        embedding_model=embedding_model,
        kb_build_id=resolve_active_knowledge_build_id(),
    )
    summary["ablation"] = {
        "config": name,
        "description": ABLATION_CONFIGS[name]["description"],
        "env_overrides": overrides,
        "pipeline_fingerprint": fingerprint,
        "system_model": system_model,
    }
    _, raw_path = export_evaluation_artifacts(summary, output_dir=out_dir)
    return raw_path


def main(argv: Sequence[str] | None = None) -> int:
    from dotenv import load_dotenv

    # Nạp .env trước, sau đó cấu hình ablation mới ghi đè đúng các biến cần thay đổi.
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    parser = argparse.ArgumentParser(description="Chạy ablation reranker / GraphRAG trên bộ RAGAS.")
    sub = parser.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("--config", required=True, choices=sorted(ABLATION_CONFIGS))
    run_cmd.add_argument("--judge", default=None, help="Judge model (mặc định RAGAS_JUDGE_MODEL).")
    run_cmd.add_argument("--judge-provider", default=None)
    run_cmd.add_argument(
        "--embedding-model",
        default=os.getenv("RAGAS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
    run_cmd.add_argument("--max-cases", type=int, default=None, help="Chỉ chạy N ca đầu (thử nhanh).")
    args = parser.parse_args(argv)

    judge_model = (args.judge or os.getenv("RAGAS_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL).strip()
    judge_provider = (
        args.judge_provider or os.getenv("RAGAS_JUDGE_PROVIDER") or DEFAULT_JUDGE_PROVIDER
    ).strip()
    raw_path = asyncio.run(
        run_ablation_config(
            args.config,
            judge_model=judge_model,
            judge_provider=judge_provider,
            embedding_model=args.embedding_model,
            max_cases=args.max_cases,
        )
    )
    print(f"Đã ghi kết quả ablation: {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
