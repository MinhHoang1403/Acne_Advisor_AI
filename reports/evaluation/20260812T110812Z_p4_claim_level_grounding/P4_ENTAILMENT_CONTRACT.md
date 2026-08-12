# P4 Entailment Contract

The initial verifier is provider-free `deterministic_entailment_v1`. It receives one claim, up to three mapped packed chunks, and their provenance. It does not receive the conversation, graph state, or corpus.

Typed verdicts:

- `SUPPORTED`: evidence supports the proposition as written.
- `PARTIALLY_SUPPORTED`: evidence supports a weaker or incomplete proposition.
- `UNSUPPORTED`: mapped evidence does not establish the key relation.
- `CONTRADICTED`: evidence explicitly conflicts with the proposition.
- `NO_EVIDENCE`: no valid candidate evidence exists.
- `VERIFIER_ERROR`: extraction/verifier/schema processing failed.

The verifier checks deterministic domain proposition polarity, explicit safety polarity, bounded lexical coverage, and unsupported specificity. It judges only the supplied evidence. General medical plausibility cannot turn absent evidence into support.

Output is schema-validated and records only status, bounded verifier confidence, evidence IDs, contradiction evidence IDs, and a structural reason code. Confidence is not a calibrated probability and is never mixed with retrieval scores. Any exception becomes `VERIFIER_ERROR`, never `SUPPORTED`.
