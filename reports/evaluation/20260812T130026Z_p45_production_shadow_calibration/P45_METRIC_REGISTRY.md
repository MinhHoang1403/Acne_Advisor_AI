# P4.5 Metric Registry

Every rate reports numerator and denominator; zero denominators are `N/A`.

| Metric | Numerator | Denominator |
| --- | --- | --- |
| Supported precision | Human-supported claims predicted supported | All adjudicated P4-supported claims |
| Supported recall | Human-supported claims predicted supported | All adjudicated human-supported claims |
| Critical extraction recall | Extracted human-critical claims | Extracted plus human-identified missing critical claims |
| Critical false allow | Non-supported human-critical claims with `WOULD_ALLOW` | All adjudicated non-supported human-critical claims |
| Critical false block | Supported human-critical claims with `WOULD_BLOCK_CRITICAL` | All adjudicated supported human-critical claims |
| Mapping accuracy | Claims whose selected evidence is human-marked appropriate | All mapping-reviewed claims |
| Provenance validity | Evidence-bearing claims with valid provenance | All adjudicated evidence-bearing claims |

Readiness also requires at least 50 reviewed questions, 150 adjudicated claims, 30 adjudicated critical claims, all locked regressions, and no production behavior change.
