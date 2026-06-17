# Changelog

## 1.3.0 (representation reframe)

- Reframed project from "benchmark/comparison" to "representation sufficiency" framing
- Removed: lead-time analysis, trajectory prediction, Vec3-vs-binary comparison, challenge mode, calibration sweep
- Added: representation sufficiency criterion framing throughout docs and reports
- Added: CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, AUTHORS.md, issue/PR templates
- Added: docs/index.md documentation landing page
- Renamed: `canonical_primary_metric` → `primary_metric` across backend adapters
- Simplified: CLI to `run-loop`, `run-failure-set`, `run-ablation`, `extract-cases`
- Simplified: `ProposalEngine` no longer supports calibration/challenge modes
- Simplified: `sliding_window_coupling` no longer computes tau-based warning thresholds
- Updated: README with CI/PyPI/DOI badges, TOC, roadmap, BibTeX citation

## 1.2.0

- Added coupling analysis module with sliding-window anti-coupling detection and lead-time measurement.
- Reframed the public release around the canonical torch benchmark artifacts.
- Strengthened the narrow-scope claim language throughout docs.

## 1.1.0

- Initial public release with simulator and torch backends.
- Canonical failure benchmark with 4 matching cases.
- Ablation bundle proving all 3 axes necessary.
- Collaboration protocol with action library and wisdom store.
