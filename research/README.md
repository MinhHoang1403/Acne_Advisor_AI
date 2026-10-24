# Research evidence on the canonical main checkout

Production has one implementation under `src/`. The scripts retained here are
frozen offline analysis/replay tools, not an alternative backend, an active
development branch, or an adopted evidence selector. No research prototype is
imported by the normal answer path.

## Current evidence and pending review

- [Evidence-loss forensics](RAG_EVIDENCE_LOSS_ROOT_CAUSE_STUDY.md): 85/88 required
  evidence occurrences were acquired and 58/88 packed; 20 character-budget and
  7 item-budget losses explain the post-acquisition gap. No implementation bug
  was confirmed. Inputs are the frozen thesis outputs now retained on `main`.
- [Controlled packing study](BUDGET_AWARE_PACKING_CONTROLLED_STUDY.md) and
  [locked protocol](BUDGET_AWARE_PACKING_PROTOCOL.md): exact baseline replay,
  treatment 65/88 versus 58/88, but +12/-5 requirements, costly selection and
  no eligible independent holdout. `EXPERIMENT_FAILED`; not adopted. Keep the
  approximately 2.35 MiB ignored candidate cache at
  `data/cache/research_budget_aware_packing/candidates.json` for offline replay.
- [Local extraction contract study](LOCAL_RAGCHECKER_EXTRACTION_CONTRACT_STUDY.md):
  `INDEPENDENT_GOLD_REVIEW_REQUIRED`. Candidate C produced parser-valid outputs
  but still has unresolved semantic review items. Its protocol, A/B/C outputs,
  developer-review packet, draft holdout and overlap audit are retained under
  `results/local_extraction_contract_*`. The original local 4B/9B calibration
  outputs are retained as the review baseline. Failed model runners and alternate
  evaluator implementations are not maintained here.
- [Controlled evidence mapping review](CONTROLLED_EVIDENCE_REVIEW.md): eight
  technical mappings still need researcher decisions. Unchecked boxes remain
  unchecked; this packet is not independent clinical validation or final gold.

The local extraction holdout concerns evaluator claims, not context packing.
It must not be presented as an independent packing holdout. Historical references
to branch names or runner paths in the frozen study reports describe the original
execution; ongoing work uses `main` and the retained review artifacts.

## Closed experiments: conclusions to preserve

| Study / source commit | Supported outcome | Production disposition |
|---|---|---|
| Local RAGChecker 4B, `0a598a8` | `LOCAL_EVALUATOR_QUALITY_FAILED`; stable execution did not satisfy calibration | No migration |
| Local RAGChecker 9B, `9e52989` | Checking improved, extraction gate still failed; hybrid execution alone was insufficient | No migration |
| RAGAS shadow, `1c4748c` | `RAGAS_CALIBRATION_BLOCKER`; formal shadow was not run | RAGChecker remains the semantic evaluator |
| Focused evidence selector, `104c4fe` | No defensible final holdout; facet/BGE prototype rejected | No selector adoption |
| GPU evidence selection, `ed0b469` | Rejected/inconclusive candidates; label review needed | No selector adoption |
| Information-need decomposition, `7806b52` | Critical/stability gates failed; only 3/13 GOOD development cases | No routing change |
| Requested-information representation, `a2fd3f4` | Schema and critical-case gates failed | No runtime replacement |
| Retrieval-first boundary, `a410730` | `RETRIEVAL_FIRST_QUERY_REWRITE_BLOCKER` | Keep the existing action/query-rewrite boundary |
| Simple-path baseline, `6baa08a` | Safety/label gates not frozen | No production baseline replacement |
| Retrieval methods, `72046e2` | Research-only comparison; no production adoption established | Keep the current retrieval contract |
| Reranker studies, `a374b8e`, `7e02625`, `0d96c9a` | BF16 finding integrated; depth-24 cap rejected | PR #115 keeps full-union inference |
| Retry evidence loss, `b6e1fd6` | Retained/acquired union reranked with the targeted query | Already integrated through PR #116 |
| Evidence-path audit, `a99f259` | Packing trace and serialized-inference fixes have subsequent implementations on main | Do not reapply the older patch |

Old architecture audit/checkpoint reports are superseded by `docs/ARCHITECTURE.md`.
The old formatting stash belonged to a removed deterministic-answer architecture;
the current presentation and cache-version contracts supersede it. The other code
stash only removed the retired `src/web/index.html` frontend.

An old notebook stash changed only `RESEARCHER_REVIEW_APPROVED=False` to `True`
and captured a superseded run: Claim Recall 60.5%, Context Precision 26.3%,
Faithfulness 74.3%, Claim F1 37.3%, rejection 21/30. Those outputs did not establish
an immutable current-run identity and must not replace either retained formal run.
Its approval flag is not carried into the canonical notebook.

## Formal evaluation

See [evaluation results](../evaluation/results/README.md) and the
[thesis protocol](../evaluation/THESIS_EVALUATION_PROTOCOL.md). The 100 frozen
current-build system outputs and locked gold are preserved on `main`; semantic
scoring remains blocked by evaluator credits. Resume scoring from those exact
outputs rather than regenerating successful cases. The completed historical
formal result remains separate and immutable.
