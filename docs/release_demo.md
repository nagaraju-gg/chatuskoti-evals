# Release Demo

This is the intended V1.2 reading order for a public bundle.

## 1. Start with the canonical benchmark

- [summary.md](../artifacts/strong_v1_2_torch/canonical_failure/failure_injection/summary.md)
- [benchmark_figure.svg](../artifacts/strong_v1_2_torch/canonical_failure/failure_injection/benchmark_figure.svg)

Lead with the controlled four-case benchmark. This is the strongest checked-in evidence because it validates the intended `hold`, `reframe`, and `rollback` branches directly.

## 2. Show what happens in a loop

- [comparison.md](../artifacts/strong_v1_2_torch/challenge_compare/comparison.md)
- [challenge_cases.md](../artifacts/strong_v1_2_torch/challenge_compare/challenge_cases.md)

Use the challenge comparison as companion evidence, not as the headline. The point is that binary can post a higher metric by accepting benchmark-aware invalid merges.

## 3. Show that the extra axes matter

- [summary.md](../artifacts/strong_v1_2_torch/ablations/summary.md)

The ablation table is the compact proof that `reliability` and `validity` are doing real work rather than acting as decorative scores.

## 4. Show V1.2 calibration and provenance

When you regenerate the V1.2 release bundle with `bash scripts/run_torch_release_bundle.sh`, add:

- `artifacts/strong_v1_2_torch/calibration/summary.md`
- `artifacts/strong_v1_2_torch/calibration/threshold_sweep.svg`
- the richer `manifest.json` files under each release sub-bundle

These artifacts answer the V1.2 questions:

- do nearby thresholds preserve the intended benchmark behavior?
- can someone identify exactly which detector settings and git revision produced the bundle?
- does the release script verify the generated outputs automatically?

## Public framing

Keep the claim narrow:

- benchmark-specific
- calibrated to `CIFAR-100 + ResNet-18`
- reproducible and inspectable

Avoid presenting the project as a general automated researcher or as proof that `Vec3` universally dominates metric-only gating.
