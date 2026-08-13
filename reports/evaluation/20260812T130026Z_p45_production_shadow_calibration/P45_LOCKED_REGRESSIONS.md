# P4.5 Locked Regressions

All fixtures were executed unchanged after freezing the P4.5 package.

| Gate | Result |
| --- | --- |
| Retrieval V5 locked R8 | PASS, 18/18 |
| R8 source coverage | PASS, 18/18 |
| R8 critical safety | PASS, 4/4 |
| R8 answer quality | PASS, 55/55 |
| Frozen P3 | PASS, 17/17; bounded retry and safe abstention unchanged |
| Frozen P4 curated fixture | PASS, 32/32; 14 critical pairs |
| P4/P4.5 targeted tests | PASS, 33 |
| Medical/safety focused suite | PASS, 141 |
| Aggregate Phase 2 | PASS, 12/12 |
| Runtime resilience | PASS |
| Reproducible environment | PASS |
| Offline release readiness | PASS |

No locked fixture or gate definition was changed.
