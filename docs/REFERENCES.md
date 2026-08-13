# References and Evidence Status

This repository separates implementation provenance from scientific validation.

- Source medical documents are controlled through `sample_data/` and the Phase 1
  ingestion manifest.
- Taxonomy source files and provenance fields live under `data/taxonomy/`.
- Formula locations and evidence status are listed in
  `docs/METHODS_AND_FORMULAS.md`.
- Regression fixtures protect known behavior but are not trusted clinical gold.
- The retired synthetic Evaluation V3 dataset is preserved only in Git history
  and must not be cited as decision-grade evidence.

Outstanding literature review, parameter validation and source-corpus freezing
belong to S4A. Final agent/tool ablation and architecture decisions belong to
S4B.
