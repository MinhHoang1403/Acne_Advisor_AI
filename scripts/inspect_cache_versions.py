#!/usr/bin/env python3
"""Inspect the runtime answer-cache version and pipeline fingerprint."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

from src.cache.redis_cache import close_redis, get_redis  # noqa: E402
from src.observability.versioning import (  # noqa: E402
    build_pipeline_version_manifest,
    compute_pipeline_fingerprint,
    get_answer_cache_version,
    pipeline_manifest_summary,
)


async def inspect_cache_versions(max_scan: int = 1000) -> dict[str, Any]:
    manifest = build_pipeline_version_manifest()
    fingerprint = compute_pipeline_fingerprint(manifest)
    answer_version = get_answer_cache_version()
    warnings: list[str] = []
    report: dict[str, Any] = {
        "passed": True,
        "cache_backend_detected": False,
        "current_answer_cache_version": answer_version,
        "current_pipeline_fingerprint": fingerprint,
        "pipeline_manifest": pipeline_manifest_summary(manifest),
        "legacy_entries_detected": None,
        "warnings": warnings,
    }
    configured_version = os.getenv("CACHE_ANSWER_VERSION")
    if configured_version and configured_version != answer_version:
        warnings.append(
            f"CACHE_ANSWER_VERSION={configured_version!r} is obsolete; effective answer cache version is {answer_version!r}."
        )

    redis = await get_redis()
    if redis is None:
        warnings.append("Redis cache backend not reachable or disabled; inspection skipped.")
        return report

    report["cache_backend_detected"] = True
    legacy_entries = 0
    scanned = 0
    cursor = 0
    try:
        while True:
            cursor, keys = await redis.scan(cursor=cursor, match="cache:answer:*", count=200)
            for key in keys:
                scanned += 1
                if scanned > max_scan:
                    warnings.append(f"Stopped after scanning {max_scan} cache entries.")
                    break
                raw = await redis.get(key)
                try:
                    parsed = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    legacy_entries += 1
                    continue
                metadata = parsed.get("metadata", {}) if isinstance(parsed, dict) else {}
                if not isinstance(metadata, dict) or not (
                    metadata.get("pipeline_fingerprint") or parsed.get("pipeline_fingerprint")
                ):
                    legacy_entries += 1
            if cursor == 0 or scanned > max_scan:
                break
        report["legacy_entries_detected"] = legacy_entries
        report["scanned_entries"] = min(scanned, max_scan)
        return report
    except Exception as exc:
        return {
            **report,
            "passed": False,
            "error": str(exc),
        }
    finally:
        await close_redis()


def main() -> int:
    try:
        report = asyncio.run(inspect_cache_versions())
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0 if report.get("passed") else 1
    except Exception as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
