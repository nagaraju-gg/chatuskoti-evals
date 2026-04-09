# Figure 1: Canonical Failure Benchmark

![Canonical failure benchmark figure](../artifacts/strong_v1_2_torch/canonical_failure/failure_injection/benchmark_figure.svg)

## Result

Figure 1 shows the canonical four-case adversarial calibration benchmark on the torch-backed `CIFAR-100 + ResNet-18` setup. On this benchmark, a metric-only binary controller would adopt `3/4` cases, including the pyrrhic probe, the metric-gaming probe, and the evaluation-regime shift. In contrast, `Vec3` routes those same cases to `hold`, `reframe`, and `reframe`, while escalating the damaged-failure case to `rollback`.

The checked-in figure corresponds to the current curated canonical artifact in the repo. The V1.2 release bundle script regenerates the same figure layout from the checked-in `3`-seed benchmark run and records the detector/backend provenance in the manifest.

The benchmark therefore validates the four intended `Vec3` branches on a benchmark-specific calibrated suite:

- pyrrhic improvement handling
- metric-gaming / validity failure detection
- damaged failure escalation
- incomparability detection

## Interpretation

The strongest comparative signal here is not that `Vec3` always finds a better optimization path. It is that `Vec3` preserves structural information about why a result should or should not be trusted, whereas binary evaluation collapses that structure into a single merge/no-merge decision keyed only to the primary metric.

This benchmark is the clearest current evidence that the additional axes are useful rather than decorative. The binary controller is correct only when the primary metric is already a sufficient summary of the run. The benchmark cases are designed to show the opposite regime: situations where the metric moves in a direction that would tempt a naive controller to merge a result that is unstable, metric-gamed, or incomparable.

## Limitations

This figure should be presented as a benchmark-specific adversarial calibration result, not as a general claim about all automated research systems. The probes are explicit stress tests, which makes them appropriate for validating the evaluator’s intended behavior but insufficient on their own to establish broad superiority in open-ended automated research loops. The open-loop torch experiments remain exploratory and should be framed as secondary evidence rather than the primary proof point.

## Suggested in-paper caption

`Vec3` versus binary evaluation on the canonical four-case failure benchmark. A metric-only binary controller would adopt three of four cases, including pyrrhic, metric-gamed, and incomparable outcomes. `Vec3` instead routes those cases to `hold`, `reframe`, and `reframe`, and escalates the damaged failure case to `rollback`.
