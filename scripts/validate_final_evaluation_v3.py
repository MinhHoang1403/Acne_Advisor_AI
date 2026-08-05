"""Validate the canonical Evaluation V3 dataset without runtime or model calls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.validators import load_cases, sha256_file, validate_cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "data" / "acne_system_eval_v3.jsonl")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    rows = load_cases(dataset)
    counts = validate_cases(rows, ROOT)
    print("FINAL_EVALUATION_V3_DATASET: PASS")
    print(f"Dataset: {dataset}")
    print(f"SHA-256: {sha256_file(dataset)}")
    print(f"Rows: {len(rows)}")
    print(f"Categories: {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
