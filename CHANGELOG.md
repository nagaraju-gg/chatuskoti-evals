# Changelog

## 1.3.0 - 2026-06-03

- Added coupling analysis module with sliding-window anti-coupling detection and lead-time measurement (Goodhart/Pyrrhic trajectory warnings vs snapshot pre-check).
- Added `lead-time` CLI command to run multi-step progressive-degradation trajectories and report lead steps N.
- Added axis-level ablation names (`t_only`, `t_r`, `t_v`, `t_r_v`) for the independence table in §4 of the Vec3 paper.
- Added annotation framework (`AnnotationStore`, `extract-cases` CLI) for human rater ground-truth studies.
- Updated test assertions to `1.3.0`.

## 1.2.0 - 2026-04-09

- Added a threshold calibration sweep bundle and `run-calibration` CLI command for nearby-threshold robustness checks.
- Enriched bundle manifests with release provenance, detector/backend config, and a benchmark spec id.
- Upgraded the torch release script to run tests, generate the V1.2 bundle, and verify artifacts automatically.
- Added a release demo page and refreshed release runbook docs around the V1.2 workflow.

## 1.1.0 - 2026-04-02

- Introduced the V1.1 `T/R/V` evaluator and torch-backed evidence bundle.
- Reframed the public release around the canonical `strong_v1_1_torch` benchmark artifacts.
- Added release-oriented docs for the benchmark bundle, failure benchmark, and public claims.
- Aligned release maintenance so the package version, release checklist, and bundle-generation script point to the same V1.1 artifact set.
