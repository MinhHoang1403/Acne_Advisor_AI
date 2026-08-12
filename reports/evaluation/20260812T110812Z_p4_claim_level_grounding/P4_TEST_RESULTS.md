# P4 Test Results

- P4 targeted: 27 PASS.
- P4 plus cache/versioning focused suite after enforcement interlock: 45 PASS before the final two extraction regressions were added; the final full suite includes both.
- Full backend: 845 PASS.
- Coverage: 84.28% (gate 70%, baseline 83.81%).
- `src/quality/claim_grounding.py`: 95% coverage.
- Locked R8: 18/18 PASS.
- P3 evaluator: PASS.
- Answer-quality: 55/55 PASS.
- Medical/safety focused regression: 178 PASS.
- Safe fallback: 13/13 PASS.
- Aggregate Phase 2: PASS.
- Runtime resilience: PASS.
- Reproducible environment: PASS.
- Offline release readiness: 17/17 PASS.
- Frontend: 42 PASS; lint PASS; build PASS; npm audit 0 vulnerabilities.
- `pip check`: PASS.
- `compileall`: PASS.

No live LLM/embedding provider was called by P4 calibration.
