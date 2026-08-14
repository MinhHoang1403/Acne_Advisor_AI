# Safety Contracts

Authoritative deterministic safety behavior lives under `src/quality/` and is
invoked from the LangGraph workflow. It is not delegated to an optional LLM
tool.

Key invariants:

- emergency guidance precedes routine acne advice;
- the system does not diagnose, prescribe, or replace a clinician;
- pregnancy and medication-risk wording is preserved by final guards;
- evidence must include medical text and source provenance;
- insufficient evidence after two attempts produces explicit abstention;
- provider failure produces a safe, non-fabricated fallback;
- cache entries are versioned, fingerprinted, and quality-gated;
- EntityCards and graph structure are not medical evidence;
- observability redacts raw queries and secret-like values.

Regression tests cover emergency escalation, pregnancy constraints,
medication wording, severity-aware answers, fallback, source validation,
Markdown presentation, and cache eligibility. Regression fixtures protect
software behavior; they are not clinical gold and do not establish answer
quality or external validity.
