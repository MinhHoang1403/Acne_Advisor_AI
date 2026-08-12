# P4 Metric Registry

Zero denominators are reported as `N/A`, never 100%.

| Metric | Numerator | Denominator |
| --- | --- | --- |
| Verification accuracy | Correct verdicts | All 32 claim pairs |
| Supported precision | Gold supported among predicted supported | All predicted supported |
| Supported recall | Predicted supported among gold supported | All gold supported |
| Class detection rate | Correct predictions for one gold class | All gold examples in that class |
| Critical extraction recall | Critical claims extracted as critical | All gold critical claims |
| Critical unsupported detection | Non-supported predictions for critical unsupported/no-evidence | All critical unsupported/no-evidence |
| Critical contradiction detection | Correct contradiction predictions | All critical contradiction cases |
| Critical false allow | Gold non-supported critical predicted supported | All gold non-supported critical |
| Claim with valid evidence | Claims with a valid mapped packed chunk | All claims |
| Evidence mapping accuracy | Exact expected evidence-ID mappings | All cases |
| Provenance valid rate | Valid mapped links | All mapped links |
| Verifier error rate | `VERIFIER_ERROR` claims | All claims |
| Would-block rate | `WOULD_BLOCK_CRITICAL` decisions | All cases |
| Would-abstain rate | `WOULD_ABSTAIN` decisions | All cases |
| Shadow answer change rate | Shadow requests whose production draft changed | All shadow requests |

Observed: accuracy, supported precision/recall, each class detection rate, critical extraction, critical unsupported/contradiction detection, evidence mapping, and provenance validity are 1.0; critical false allow, verifier error, and shadow answer change are 0.0.
