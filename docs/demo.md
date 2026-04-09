# Demo Walkthrough

This repo is easiest to understand in three stops.

## 1. Canonical failure benchmark

Start here:

- [summary.md](../artifacts/strong_v1_2_torch/canonical_failure/failure_injection/summary.md)
- [benchmark_figure.svg](../artifacts/strong_v1_2_torch/canonical_failure/failure_injection/benchmark_figure.svg)

This is the strongest artifact in the repo. It deliberately constructs four benchmark-aware stress cases and matches all `4/4` expected outcomes on the real torch backend:

- pyrrhic gain
- metric-gamed gain
- broken/damaged result
- incomparable eval shift

The intended read is simple:

- binary eval would merge several of these cases
- `Chatuskoti Eval Framework` routes each one differently because they are different kinds of outcomes

## 2. Challenge comparison

Then read:

- [comparison.md](../artifacts/strong_v1_2_torch/challenge_compare/comparison.md)
- [challenge_cases.md](../artifacts/strong_v1_2_torch/challenge_compare/challenge_cases.md)

This is the open-loop-style companion result. It is not meant to be read as a plain leaderboard. The important question is whether a higher binary metric came from accepting cases the benchmark says should not be merged.

That is why the challenge report explicitly lists:

- which cases binary adopted
- what `Vec3` did instead
- which signals fired

In the current torch V1.2 bundle, the key divergences are exactly the ones you want:

- `pyrrhic_probe`: binary `adopt` vs `Vec3` `hold`
- `metric_gaming_probe`: binary `adopt` vs `Vec3` `reframe`
- `eval_tta`: binary `adopt` vs `Vec3` `reframe`

## 3. Ablation sweep

Finally read:

- [summary.md](../artifacts/strong_v1_2_torch/ablations/summary.md)

And for the V1.2 release-specific trust checks:

- [summary.md](../artifacts/strong_v1_2_torch/calibration/summary.md)

This answers the question: do the extra `Vec3` axes actually matter?

Expected pattern:

- remove `reliability` and pyrrhic/broken handling gets worse
- remove `validity` and metric-gaming/eval-shift handling gets worse

## Limitations

This is a strong v1, not a universal claim.

- one benchmark: `CIFAR-100 + ResNet-18`
- typed interventions, not open-ended code synthesis
- benchmark-specific calibration
- evaluation layer, not a full autonomous research system

That narrowness is deliberate. The goal is to make the core decision logic legible and reproducible before expanding breadth.
