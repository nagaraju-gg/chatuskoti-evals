# Release Checklist

Use this checklist before treating the repo as a public benchmark release.

## Positioning

- Present the project as a benchmark-specific calibrated prototype.
- Lead with the adversarial calibration benchmark, not the open-loop controller runs.
- Keep claims narrow: distinguish pyrrhic, Goodhart-style, broken, and incomparable cases on the controlled suite.

## Required artifacts

- Canonical benchmark report: [summary.md](../artifacts/torch_failure_set_e10_v5/failure_injection/summary.md)
- Figure: [benchmark_figure.svg](../artifacts/torch_failure_set_e10_v5/failure_injection/benchmark_figure.svg)
- Figure caption: [benchmark_figure_caption.md](../artifacts/torch_failure_set_e10_v5/failure_injection/benchmark_figure_caption.md)
- Paper-ready section: [paper_figure_1.md](paper_figure_1.md)
- Claims and limitations note: [canonical_failure_benchmark.md](canonical_failure_benchmark.md)

## Repo hygiene

- `python3 -m unittest discover -s tests -v`
- `python3 scripts/generate_failure_figure.py artifacts/torch_failure_set_e10_v5/failure_injection/failure_results.json`
- Confirm the README points to the canonical benchmark artifact first.
- Avoid committing local environment files, `.DS_Store`, and transient artifacts unless they are intentional benchmark outputs.

## Safe public claims

Good:

- Binary evaluation would adopt three of four adversarial calibration cases that `Vec3` routes to `hold`, `reject`, or `reframe`.
- The benchmark validates the four intended `Vec3` branches on a controlled `CIFAR-100 + ResNet-18` suite.
- When showing open-loop comparisons, prefer the benchmark-aware `challenge` mode over the plain default loop.

Avoid:

- Claims of universal superiority across automated research systems
- Claims that threshold values are generally valid across domains
- Claims that the open-loop benchmark is already stronger evidence than the calibration suite

## Nice-to-have before public release

- A single top-level demo page or notebook that links the benchmark figure and result table
- A short roadmap for strengthening the open-loop controller comparison
- A license file and explicit citation metadata, if the repo is going public beyond a private draft audience
