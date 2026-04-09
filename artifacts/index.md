# Strong V1.2 Torch Artifact Index

This directory collects the strongest current torch-backed evidence bundle for `Chatuskoti Evals`.

The current lead bundle is [strong_v1_2_torch](strong_v1_2_torch). It preserves the same real-backend benchmark result while adding a calibration sweep and richer manifests.

## Start here

- canonical failure benchmark: [summary.md](strong_v1_2_torch/canonical_failure/failure_injection/summary.md)
- canonical benchmark figure: [benchmark_figure.svg](strong_v1_2_torch/canonical_failure/failure_injection/benchmark_figure.svg)
- challenge comparison: [comparison.md](strong_v1_2_torch/challenge_compare/comparison.md)
- challenge case table: [challenge_cases.md](strong_v1_2_torch/challenge_compare/challenge_cases.md)
- ablation summary: [summary.md](strong_v1_2_torch/ablations/summary.md)
- calibration summary: [summary.md](strong_v1_2_torch/calibration/summary.md)

## How to read this bundle

1. Read the canonical failure benchmark first.
2. Use the challenge comparison as companion evidence, not as a plain leaderboard.
3. Use the ablation sweep to see whether `reliability` and `validity` are doing real work.

## Interpretation shortcut

- If binary is higher on the challenge metric but only because it adopted pyrrhic, metric-gamed, or incomparable cases, that is evidence for the value of structured evaluation.
- The canonical failure benchmark is the primary proof artifact.
- The challenge comparison is the “what this looks like inside a loop” artifact.
