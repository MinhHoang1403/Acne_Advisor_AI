# P4 P3 Regression

- Frozen fixture remained unchanged: `tests/golden/p3_evidence_sufficiency_cases.json`.
- `eval_p3_evidence_sufficiency.py`: PASS.
- Sufficient cases remained sufficient.
- Retrieval attempts remained capped at two total with one retry maximum.
- Unnecessary retry remained zero in the frozen cases.
- Insufficient and critical-missing cases still abstained.
- Unsafe answer after insufficiency remained zero.
- P4 runs only after valid draft generation; P3 abstention continues directly to the existing safe fallback/finalizer path.
