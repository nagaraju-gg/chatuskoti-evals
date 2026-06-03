# Failure Injection Report

- Baseline metric: `0.6379`
- Cases: `4`
- Expectation matches: `4/4`
- Binary would adopt: `3` cases
- Vec3 would adopt: `0` cases

## Cases
### `pyrrhic_win_injection` via `dropout_high`
- Narrative: Metric nudges upward while the train/val gap blows out, forcing the reliability detector to intervene.
- Candidate metric: `0.6581`
- Expected resolution: `hold`
- Binary action: `adopt`
- Binary reason: `mean metric delta 0.0202 exceeds binary threshold 0.0020`
- Actual resolution: `hold`
- Match: `True`
- Resolver reason: `reliability 0.329 is below threshold 0.350; result is too unstable to merge`
- Expected signals: `instability_gap`
- Actual signals: `instability_gap`
- TRV: `(0.762, 0.329, 0.298)`
- Reliability components: `gap_health=-0.478, gradient_health=0.648, seed_variance=0.888`
- Validity components: `comparison_validity=1.000, efficiency_validity=-1.000, proxy_alignment=0.138`
### `goodhart_injection` via `stochastic_depth_high`
- Narrative: Detector stress test where the metric improves while gradients collapse into hyper-coherence and validity degrades.
- Candidate metric: `0.6830`
- Expected resolution: `reframe`
- Binary action: `adopt`
- Binary reason: `mean metric delta 0.0451 exceeds binary threshold 0.0020`
- Actual resolution: `reframe`
- Match: `True`
- Resolver reason: `validity -0.100 is below threshold 0.150; apparent gain is not decision-ready`
- Expected signals: `hyper_coherence, proxy_decoupling`
- Actual signals: `instability_gap, hyper_coherence, proxy_decoupling`
- TRV: `(0.978, 0.426, -0.100)`
- Reliability components: `gap_health=-0.513, gradient_health=1.000, seed_variance=0.906`
- Validity components: `comparison_validity=1.000, efficiency_validity=-1.000, proxy_alignment=-1.000`
### `broken_failure_injection` via `high_lr`
- Narrative: Adversarial learning-rate jump that should trigger the damaged-failure path.
- Candidate metric: `0.5912`
- Expected resolution: `rollback`
- Binary action: `reject`
- Binary reason: `mean metric delta -0.0467 does not exceed binary threshold 0.0020`
- Actual resolution: `rollback`
- Match: `True`
- Resolver reason: `truthness -0.981 is below -0.250 and reliability 0.044 indicates internal damage`
- Expected signals: `exploding_gradients`
- Actual signals: `exploding_gradients, loss_instability`
- TRV: `(-0.981, 0.044, 0.095)`
- Reliability components: `gap_health=0.012, gradient_health=-1.000, seed_variance=0.951`
- Validity components: `comparison_validity=1.000, efficiency_validity=-1.000, proxy_alignment=-0.443`
### `incomparable_eval_injection` via `eval_tta`
- Narrative: Evaluation protocol shift that should invalidate comparison to baseline.
- Candidate metric: `0.6550`
- Expected resolution: `reframe`
- Binary action: `adopt`
- Binary reason: `mean metric delta 0.0171 exceeds binary threshold 0.0020`
- Actual resolution: `reframe`
- Match: `True`
- Resolver reason: `validity 0.039 is below threshold 0.150; apparent gain is not decision-ready`
- Expected signals: `eval_regime_changed`
- Actual signals: `eval_regime_changed`
- TRV: `(0.691, 0.966, 0.039)`
- Reliability components: `gap_health=0.994, gradient_health=1.000, seed_variance=0.908`
- Validity components: `comparison_validity=-1.000, efficiency_validity=1.000, proxy_alignment=0.825`
