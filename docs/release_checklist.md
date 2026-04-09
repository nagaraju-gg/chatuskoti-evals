# Release Checklist

Use this checklist before treating the repo as a public `Chatuskoti Eval Framework` benchmark release.

## Positioning

- Present the project as a benchmark-specific calibrated prototype.
- Lead with the adversarial calibration benchmark, not the open-loop controller runs.
- Keep claims narrow: distinguish pyrrhic, metric-gamed, broken, and incomparable cases on the controlled suite.

## Required artifacts

- Canonical benchmark report: [summary.md](../artifacts/strong_v1_2_torch/canonical_failure/failure_injection/summary.md)
- Figure: [benchmark_figure.svg](../artifacts/strong_v1_2_torch/canonical_failure/failure_injection/benchmark_figure.svg)
- Figure caption: [benchmark_figure_caption.md](../artifacts/strong_v1_2_torch/canonical_failure/failure_injection/benchmark_figure_caption.md)
- Challenge comparison: [comparison.md](../artifacts/strong_v1_2_torch/challenge_compare/comparison.md)
- Ablation summary: [summary.md](../artifacts/strong_v1_2_torch/ablations/summary.md)
- Calibration summary: `artifacts/strong_v1_2_torch/calibration/summary.md`
- Calibration sweep figure: `artifacts/strong_v1_2_torch/calibration/threshold_sweep.svg`
- Paper-ready section: [paper_figure_1.md](paper_figure_1.md)
- Claims and limitations note: [canonical_failure_benchmark.md](canonical_failure_benchmark.md)
- Demo page: [release_demo.md](release_demo.md)

## Repo hygiene

- `python3 -m unittest discover -s tests -v`
- `python3 scripts/verify_release_bundle.py artifacts/strong_v1_2_torch`
- `python3 scripts/generate_failure_figure.py artifacts/strong_v1_2_torch/canonical_failure/failure_injection/failure_results.json`
- Be explicit about whether you are referencing the checked-in curated bundle or a freshly regenerated `3`-seed bundle.
- Confirm the package version and release label agree before publishing anything as `v1.2`.
- Add an annotated git tag such as `v1.2.0` once tests and curated artifacts are final.
- Confirm the README points to the canonical benchmark artifact first.
- Confirm each generated manifest includes `package_version`, `git_commit`, `benchmark_spec_id`, and matching artifact paths.
- If you mention threshold robustness publicly, note that loosening the validity gate to `0.05` flips the incomparable eval-shift case.
- Avoid committing local environment files, `.DS_Store`, and transient artifacts unless they are intentional benchmark outputs.

## Safe public claims

Good:

- Binary evaluation would adopt three of four adversarial calibration cases that `Vec3` routes to `hold`, `reframe`, or `rollback`.
- The benchmark validates the four intended `Vec3` branches on a controlled `CIFAR-100 + ResNet-18` suite.
- When showing open-loop comparisons, prefer the benchmark-aware `challenge` mode over the plain default loop.

Avoid:

- Claims of universal superiority across automated research systems
- Claims that threshold values are generally valid across domains
- Claims that the open-loop benchmark is already stronger evidence than the calibration suite

## Nice-to-have before public release

- A short roadmap for broadening beyond the current benchmark
- One short note explaining how the calibration sweep should be interpreted when a threshold profile flips a case
