# Failure Injection Set

We include deliberately adversarial interventions to stress-test detectors.

This is a core part of the benchmark story, not an optional appendix. The failure injection set is designed to answer the reviewer question: "Does the controller only work on friendly examples?"

## Injected interventions

- `pyrrhic_probe`: pyrrhic win injection
- `metric_gaming_probe`: metric-gaming / validity-failure injection
- `broken_probe`: broken failure injection
- `eval_tta`: incomparable evaluation injection

The named cases are defined in [chatuskoti_evals/scenarios.py](../chatuskoti_evals/scenarios.py).

## Canonical demo cases

These three should anchor the paper and short demo:

1. Pyrrhic win example
2. Metric-gaming example
3. Recovery via Vec3 history

## Scope

Claims should stay narrow:

- benchmark-specific calibrated prototype
- evidence on one controlled benchmark
- deterministic evaluation and interpretability outputs

Avoid broad claims about all automated research systems until the wider calibration suite and broader benchmark coverage exist.
