# Canonical Failure Cases

## Purpose

Four cases demonstrate that scalar evaluation (T-only) is not action-sufficient for research-loop decisions. Each case shares T > 0 with clean gains but requires a distinct resolver action because the T/R/V decomposition reveals structurally different outcomes.

## The cases

| Case | Candidate Metric | Binary Action | Vec3 Action | Why Vec3 Differs |
| --- | ---: | --- | --- | --- |
| Pyrrhic Pass | `0.6581` | `adopt` | `hold` | Metric improves, but reliability is degraded — train/val gap widened |
| Orthogonal Pass | `0.6830` | `adopt` | `reframe` | Metric improves, but validity collapses under proxy decoupling |
| Broken Failure | `0.5912` | `reject` | `rollback` | Metric worsens and internals detect damage |
| Incomparable Pass | `0.6550` | `adopt` | `reframe` | Comparison is invalid because the evaluation regime changed |

## Headline

- `4/4` expected cases matched
- Scalar (binary) would adopt `3/4` cases that share T > 0 but require non-adopt actions
- Vec3 distinguishes all four correctly

## What this supports

The argument that a single scalar (T) is not sufficient for research-loop decisions, because states with identical T require different actions.

## What this does not support

This does not by itself establish that Vec3 is generally superior on all benchmarks. It is a representational claim, not a performance claim.

## Recommended paper wording

Use wording like:

> Standard evaluation collapses structurally distinct outcomes into a single "metric up" category. Vec3 separates these cases by decomposing evaluation along three non-redundant axes, revealing that a positive metric delta is not a uniform signal.

Avoid wording like:

> Vec3 outperforms binary evaluation.

## Recommended demo structure

1. Show the sufficiency criterion
2. Show the four-case table as evidence that identical T implies non-identical actions
3. Show the distinguishability matrix
4. Show the action-collapse figure
5. Use the ablation analysis to show that removing axes causes the predicted representational collapse

## Limitations

- The benchmark uses explicitly constructed adversarial probes
- The current result is strongest as a controlled demonstration, not yet as a broad claim
