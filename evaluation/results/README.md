# Official Formal Evaluation Result

The canonical completed result is [`formal_run_2d5f0124`](formal_run_2d5f0124/).
It evaluates production commit `2d5f0124dc532a4970a4b08dcd6cf846389a03ff`
with pipeline fingerprint `f93ad3e8dfb2c39f403b0794` and knowledge build
`94d613bc9b33628de3ef`.

The run completed all 100 benchmark cases with zero infrastructure failures.
RAGChecker `0.1.9` used the fixed evaluator snapshot
`gpt-5.4-2026-03-05`. The benchmark SHA-256 is
`f61d6807c0ce39f902936844d810562c486f1bcadaa57a8a2da0e460ad7e534b`.

| Metric | Cases | Score |
|---|---:|---:|
| Claim Recall | 70 | 72.3% |
| Context Precision | 70 | 38.9% |
| Faithfulness | 70 | 87.7% |
| Claim F1 | 70 | 33.8% |
| Negative Rejection Rate | 30 | 80.0% (24/30) |

These values are the observed research result. They are not a composite score,
a clinical validation, or evidence that the assistant is safe for autonomous
medical use.

## Preserved Artifacts

| Artifact | SHA-256 |
|---|---|
| [`raw_results.json`](formal_run_2d5f0124/raw_results.json) | `0b49d1927f91ac98c342c4051bb52a522722ebf19818bd2e2baaa11ad5efdca1` |
| [`case_metrics.csv`](formal_run_2d5f0124/case_metrics.csv) | `430830e24cd784deb16e873f958377a061caad74186572c579747c0ceb216eaf` |
| [`metrics_summary.csv`](formal_run_2d5f0124/metrics_summary.csv) | `dffc12e40f52b48c40eb5ef2dca3046ae37d100560b1c06572236c5e0daa77c9` |
| [`ragchecker_checkpoint.json`](formal_run_2d5f0124/ragchecker_checkpoint.json) | `80b871759b7a8f9f3bd7eb9bfefd9599146cf4abfb568af98c47df9a4aadb984` |
| [`evaluator_calibration_results.json`](formal_run_2d5f0124/evaluator_calibration_results.json) | `8b9a37fe770ae89d618fd4b881458633b9d8ef491bcf46fe5d42d440956ce3cf` |
| [`calibration_adjudication.json`](formal_run_2d5f0124/calibration_adjudication.json) | `ded7052769d3bf8b1df58f50526039ae051468986cd7c65666ebac3a8640268d` |

The executed notebook is preserved at
[`evaluation/formal_evaluation.ipynb`](../formal_evaluation.ipynb). Earlier and
aborted development runs are retained separately under
[`research_archive/development_evaluation`](../../research_archive/development_evaluation/)
and must not be substituted for this official result.
