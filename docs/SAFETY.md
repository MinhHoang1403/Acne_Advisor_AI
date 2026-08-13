# Safety Contracts

Safety behavior is implemented under `src/quality/` and invoked by LangGraph
nodes under `src/agent/nodes/`. Regression tests cover emergency escalation,
pregnancy constraints, medication-risk wording, safe fallback, severity-aware
answers and presentation consistency.

Key invariants:

- emergency guidance is prioritized over routine acne advice;
- the system does not diagnose or prescribe;
- insufficient evidence can produce deterministic abstention;
- provider failure can produce a safe fallback;
- cached answers remain versioned and pass eligibility checks;
- P4 shadow output does not rewrite the user-visible answer;
- EntitySignal and GraphSignal are not treated as source-backed medical evidence.

Files under `tests/fixtures/` are regression data unless explicitly documented
otherwise. They are not clinical gold and do not establish external validity.
