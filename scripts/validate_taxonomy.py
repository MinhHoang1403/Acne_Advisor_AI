#!/usr/bin/env python3
"""Validate the legacy versioned taxonomy catalog without mutating runtime data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.taxonomy_models import (  # noqa: E402
    DEFAULT_TAXONOMY_V2_PATH,
    TaxonomyCatalog,
    load_taxonomy_catalog,
    validate_taxonomy_catalog,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate taxonomy schema/provenance/integrity.")
    parser.add_argument(
        "--taxonomy-path",
        default=str(DEFAULT_TAXONOMY_V2_PATH),
        help="Path to the legacy taxonomy v2 migration catalog; it is not the runtime default.",
    )
    parser.add_argument(
        "--json-schema",
        action="store_true",
        help="Print the Pydantic JSON Schema for the taxonomy catalog instead of validating data.",
    )
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help="Allow draft/rejected entities in the validated set. Production validation keeps this false.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.json_schema:
        print(json.dumps(TaxonomyCatalog.model_json_schema(), ensure_ascii=False, indent=2))
        return 0

    try:
        catalog = load_taxonomy_catalog(args.taxonomy_path)
        report = validate_taxonomy_catalog(
            catalog,
            production_verified_only=not args.allow_draft,
        )
    except Exception as exc:
        payload = {
            "passed": False,
            "taxonomy_version": None,
            "entity_counts": {},
            "verified_count": 0,
            "draft_count": 0,
            "rejected_count": 0,
            "alias_count": 0,
            "checks": [],
            "warnings": [],
            "failures": [str(exc)],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 1

    print(report.model_dump_json(indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
