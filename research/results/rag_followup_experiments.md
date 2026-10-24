# Follow-up experiments for RAG evidence loss

These are preregistration proposals only. They were not implemented or executed in this study.

## Priority 1 — Budget-aware whole-chunk selection

- **Target root cause:** `20` observed character-budget losses among
  `27` post-acquisition losses.
- **Hypothesis:** a query-conditioned, coverage-aware selector that preserves whole chunks can
  increase required-evidence packing without increasing the 7,000-character/9-item budgets.
- **Code owner:** a new research selector adjacent to `src/retrieval/context_packer.py`; production
  packer remains the control.
- **Baseline:** frozen `366f3ad8` rank-order whole-chunk packer.
- **Development data:** derive selector features on a development split only; do not expose gold IDs
  to runtime selection.
- **Independent validation:** frozen holdout cases or a separately adjudicated benchmark, with all
  thresholds preregistered before results are opened.
- **Primary metric:** packed required-evidence coverage on the independent set.
- **Secondary metrics:** complete reviewed claims/cases, context precision, per-case non-inferiority.
- **Safety regression:** evidence-gap rejection, qualifier/negation preservation, provenance validity.
- **Latency/resources:** report selector p50/p95 and total context bytes; no model/API dependency.
- **Failure criterion:** no significant holdout gain, any safety regression, or reliance on gold IDs.
- **Stop rule:** one fixed candidate pool and one preregistered selector; no budget/top-k sweep.
- **Production risk:** coverage optimization may displace useful unlabeled context or overfit labels.

## Priority 2 — Rerank-order diagnosis at the item boundary

- **Target root cause:** `7` item-budget losses whose observed post-rerank positions occur
  after the nine-item packed prefix.
- **Hypothesis:** a bounded second-stage ordering objective can preserve high-relevance evidence while
  reducing required evidence demotion, without changing the BGE model or candidate depth.
- **Code owner:** research-only ordering wrapper around `src/retrieval/reranker.py` output.
- **Baseline:** persisted BGE order at frozen SHA; no new inference during the forensic study.
- **Development data:** a disjoint labeled development split with claim/qualifier roles.
- **Independent validation:** locked holdout with paired per-case analysis.
- **Primary metric:** required IDs retained within the first nine admissible candidates.
- **Secondary metrics:** nDCG/recall at nine, packed coverage, complete cases.
- **Safety regression:** qualifier loss, evidence-gap rejection, structural provenance.
- **Latency/resources:** same candidate count and model; measure only ordering overhead.
- **Failure criterion:** aggregate gain with any preregistered per-case regression, or no holdout gain.
- **Stop rule:** one ordering hypothesis; do not swap model, top-k, weights, or context limits.
- **Production risk:** labeled relevance may not represent usefulness of unlabeled evidence.

## Priority 3 — Evidence-sufficiency decision calibration

- **Target root cause:** `6` confirmed-absence cases persisted as
  `generate/evidence_sufficient`.
- **Hypothesis:** an explicit qualifier/scope contract and text-free decision telemetry can reduce
  false-generate actions while preserving answerable completion.
- **Code owner:** `src/agent/action_decision.py` and its policy tests; no medical-answer prompt change in
  the forensic phase.
- **Baseline:** 18/24 confirmed evidence-gap rejections on frozen outputs.
- **Development data:** confirmed negative development cases separated from six ambiguous cases.
- **Independent validation:** owner-adjudicated negative holdout plus answerable safety controls.
- **Primary metric:** confirmed evidence-gap rejection.
- **Secondary metrics:** answerable generation rate and retry rate.
- **Safety regression:** emergency/scope decisions, response-contract compliance, unsupported claims.
- **Latency/resources:** preserve the current model/provider and bounded two-attempt workflow.
- **Failure criterion:** reduced answerable completion, no negative holdout gain, or policy ambiguity
  converted into automatic failure labels.
- **Stop rule:** one preregistered decision-contract change and one independent evaluation.
- **Production risk:** over-abstention and reduced usefulness.

Increasing top-k or replacing the model is not the first experiment: acquisition already covers
`85/88` requirements, while
`27` are lost after acquisition. The saved traces prove packing
and ordering outcomes, but do not prove that a different model is the required remedy.
