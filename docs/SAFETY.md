# Safety Contracts

Authoritative deterministic safety behavior lives under `src/quality/` and is
invoked from the LangGraph workflow. It is not delegated to an optional LLM
tool.

Key invariants:

- emergency guidance precedes routine acne advice;
- the system does not diagnose, prescribe, or replace a clinician;
- ordinary medication content is never appended by the severity guard;
- evidence must include medical text and source provenance;
- insufficient evidence after two attempts produces explicit abstention;
- provider failure produces a safe, non-fabricated fallback;
- cache entries are versioned, fingerprinted, and quality-gated;
- EntityCards and graph structure are not medical evidence;
- observability redacts raw queries and secret-like values.

Regression tests cover emergency escalation, narrow high-risk pregnancy policy,
severity-aware answers, fallback, source validation,
Markdown presentation, and cache eligibility. Regression fixtures protect
software behavior; they are not clinical gold and do not establish answer
quality or external validity.

## Deterministic Override Provenance

| Override | Trigger boundary | Authoritative policy source |
|---|---|---|
| anaphylaxis/severe emergency | breathing difficulty with swelling/rash, or severe drug-reaction alarms | [NHS Anaphylaxis](https://www.nhs.uk/conditions/anaphylaxis/) and the [DailyMed isotretinoin medication guide](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=1cf11710-f966-4529-8e08-02175f588bca) |
| immediate self-harm safety | explicit self-harm/suicide intent | [WHO Suicide Q&A](https://www.who.int/news-room/questions-and-answers/item/suicide) |
| suspected acne fulminans | abrupt ulcerative/hemorrhagic nodules plus systemic symptoms | [NICE NG198 recommendation 1.4.1](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations) |
| isotretinoin and pregnancy | pregnancy context plus isotretinoin/oral retinoid | [NICE NG198 recommendation 1.5.22](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations) and [MHRA isotretinoin precautions](https://www.gov.uk/drug-safety-update/isotretinoin-roaccutane-reminder-of-important-risks-and-precautions) |

These rules are action-oriented safety boundaries, not a general medical answer
engine. A full deterministic replacement is attributed to provider `system`,
clears retrieved-source display attribution, and is never answer-cache eligible.
