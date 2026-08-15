# Safety Contracts

Authoritative deterministic safety behavior lives in
`src/agent/safety_policy.py` and is invoked before cache lookup or optional
retrieval. It is not delegated to an LLM tool. Structural/provenance verification
and safe fallback contracts remain under `src/quality/`.

Key invariants:

- emergency guidance precedes routine acne advice;
- the system does not diagnose, prescribe, or replace a clinician;
- ordinary medication content is never replaced by a broad severity classifier;
- evidence must include medical text and source provenance;
- insufficient evidence after two attempts produces explicit abstention;
- provider failure produces a safe, non-fabricated fallback;
- cache entries are versioned, fingerprinted, and subject to quality checks;
- EntityCards and graph structure are not medical evidence;
- observability redacts raw queries and secret-like values.

Regression tests cover all seven narrow rules, fallback, source validation,
Markdown presentation, and cache eligibility. Regression fixtures protect
software behavior; they are not clinical gold and do not establish answer
quality or external validity.

## Deterministic Override Provenance

| Override | Trigger boundary | Authoritative policy source |
|---|---|---|
| anaphylaxis-like emergency | breathing difficulty with swelling of the lips/mouth/tongue/throat or rapidly spreading hives | [NHS Anaphylaxis](https://www.nhs.uk/conditions/anaphylaxis/) |
| chest pain with breathlessness | unnegated chest pain/tightness plus breathlessness | [NHS Chest pain](https://www.nhs.uk/symptoms/chest-pain/) |
| immediate self-harm safety | explicit personal/current self-harm or suicide intent; informational topic mentions do not trigger | [WHO Suicide Q&A](https://www.who.int/news-room/questions-and-answers/item/suicide) |
| suspected acne fulminans | acne plus fever/joint pain and severe nodular, cystic, ulcerative, or rapidly erupting lesions | [NICE NG198 recommendation 1.4.1](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations) |
| isotretinoin and pregnancy | pregnancy context plus isotretinoin/oral retinoid | [NICE NG198 recommendation 1.5.22](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations) and [MHRA oral-retinoid pregnancy prevention](https://www.gov.uk/drug-safety-update/oral-retinoids-pregnancy-prevention-reminder-of-measures-to-minimise-teratogenic-risk) |
| isotretinoin with severe headache and visual/GI symptoms | isotretinoin plus severe headache and blurred vision or nausea/vomiting | [DailyMed isotretinoin medication guide](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=72867c88-070f-4608-bfef-cc5225ebce6d) |
| prescription execution request | explicit request to prescribe, choose a drug, or choose a dose | project `ENGINEERING_POLICY_NO_PRESCRIPTION`; no external clinical-source claim |

These rules are action-oriented safety boundaries, not a general medical answer
engine. A full deterministic replacement is attributed to provider `system`,
clears retrieved-source display attribution, and is never answer-cache eligible.
The exact self-harm phrase detector is a narrow engineering policy mapped to a
clinical safety action source; it is not a clinically validated suicide-risk
classifier. Ambiguous or informational mentions continue through the grounded
agent path rather than being classified in Python.

The two NICE-mapped rules use the official recommendation URL as a current safety
cross-check. The NICE-derived retrieval snapshot has the provenance discrepancy
documented in [Data Pipeline](DATA_PIPELINE.md); passing this safety contract
does not independently reconcile that source version.
