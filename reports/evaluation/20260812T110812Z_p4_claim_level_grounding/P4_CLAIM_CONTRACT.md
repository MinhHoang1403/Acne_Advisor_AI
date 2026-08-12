# P4 Claim Contract

`AnswerClaim` is a frozen Pydantic model with `extra=forbid` and schema `answer_claim_v1`.

Fields: stable hash-based `claim_id`, externally inspectable text, normalized text, sentence/claim indexes, typed claim category, `NON_CRITICAL|CRITICAL`, source requirement, candidate evidence IDs, mapped evidence IDs, and mapped source IDs. It contains no hidden reasoning.

Claim types are `DEFINITION`, `MECHANISM`, `CAUSE_OR_ASSOCIATION`, `TREATMENT`, `SAFETY`, `CONTRAINDICATION`, `DOSING_OR_USE`, `COMPARISON`, `PROGNOSIS`, `SOURCE_ATTRIBUTION`, and `OTHER_MEDICAL_FACT`.

Extraction is deterministic: Markdown headings and boilerplate are removed; bullets, sentences, abbreviations, and table rows are handled explicitly. The normal soft bound is 16 claims. Critical claims are retained ahead of non-critical claims, and truncation/critical-overflow diagnostics are explicit.

Criticality conservatively reuses existing severity classification plus existing pregnancy, emergency, contraindication, antibiotic stewardship, dangerous treatment, and severe adverse-risk markers. It does not introduce a competing clinical severity scale.
