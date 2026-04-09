# Canonical Failure Benchmark

This document captures the strongest current benchmark artifact for the project:

- source run: [summary.md](../artifacts/strong_v1_2_torch/canonical_failure/failure_injection/summary.md)
- benchmark type: explicit adversarial calibration suite
- backend: torch
- current artifact budget: `10` epochs, `3` seeds, `num_workers=0`

This document refers to the strongest current torch V1.2 artifact in the repo workspace.

## Core result

The benchmark validates all four intended `Vec3` branches on the real torch backend.

| Case | Candidate Metric | Binary | Vec3 | Why Vec3 differs |
| --- | ---: | --- | --- | --- |
| Pyrrhic probe | `0.4519` | `adopt` | `hold` | Metric improves, but reliability is negative from gap and instability signals |
| Metric-gaming probe | `0.4569` | `adopt` | `reframe` | Metric improves, but validity collapses under proxy decoupling |
| Broken probe | `0.3919` | `reject` | `rollback` | Metric worsens and reliability is strongly negative, indicating internal damage |
| Eval TTA | `0.4272` | `adopt` | `reframe` | Comparison is invalid because the evaluation regime changed |

Headline numbers:

- `4/4` expected cases matched
- `binary` would adopt `3/4`
- `Vec3` would adopt `0/4`

## What this supports

This benchmark supports the narrow claim that a `Vec3` evaluator can distinguish multiple failure modes that a metric-only binary evaluator would merge or mishandle on a benchmark-specific calibrated suite.

In particular, it now demonstrates:

- pyrrhic improvement handling
- metric-gaming / validity failure detection
- broken failure escalation
- incomparability detection

## What this does not support

This benchmark does not by itself establish that `Vec3` is generally superior on open-ended automated research loops.

It should not be used to claim:

- universal threshold validity
- general superiority across domains
- superiority without explicit calibration

The V1.2 calibration sweep is consistent with that framing: nearby threshold changes preserve the intended benchmark behavior except when the validity gate is loosened enough to let the incomparable eval-shift case pass.

## Recommended paper wording

Use wording close to:

> On a benchmark-specific adversarial calibration suite for `CIFAR-100 + ResNet-18`, the `Vec3` evaluator correctly separated pyrrhic, metric-gamed, broken, and incomparable outcomes, while a binary metric-only controller would have accepted three of the four cases.

Avoid wording like:

> Vec3 solves automated research evaluation in general.

## Recommended demo structure

1. Show the four-case table.
2. Focus visually on the three binary false positives:
   - pyrrhic probe
   - metric-gaming probe
   - eval regime shift
3. Use the broken probe as the “damage escalation” case rather than the headline case.
4. Present open-loop controller runs only after this benchmark, as exploratory evidence.

## Limitations

- The benchmark uses explicit adversarial probes for calibration.
- The open-loop torch experiments are still weaker than the benchmark result.
- The current result is strongest as a controlled stress-test, not yet as a broad claim about unconstrained automated research.
