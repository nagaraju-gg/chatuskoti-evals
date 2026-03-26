# Failure Injection Set

We include deliberately adversarial interventions to stress-test detectors.

This is a core part of the benchmark story, not an optional appendix. The failure injection set is designed to answer the reviewer question: "Does the controller only work on friendly examples?"

## Injected interventions

- `dropout_high`: pyrrhic win injection
- `stochastic_depth_high`: Goodhart-style injection
- `high_lr`: broken failure injection
- `eval_tta`: incomparable evaluation injection

The named cases are defined in [catuskoti_ar/scenarios.py](../catuskoti_ar/scenarios.py).

## Canonical demo cases

These three should anchor the paper and short demo:

1. Pyrrhic win example
2. Goodhart example
3. Recovery via Vec3 history

## Scope

Claims should stay narrow:

- benchmark-specific calibrated prototype
- evidence on one controlled benchmark
- deterministic evaluation and interpretability outputs

Avoid broad claims about all automated research systems until the real backend and wider calibration suite exist.
