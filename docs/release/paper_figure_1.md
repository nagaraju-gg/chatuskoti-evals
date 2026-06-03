# Figure 1: Canonical Failure Benchmark

![Lead-time trajectory figure](../artifacts/strong_v1_3_torch/lead_time/lead_time_analysis/lead_time_figure.svg)

## Result

Figure 1 shows the multi-step lead-time trajectory on the torch-backed `CIFAR-100 + ResNet-18` setup. The trajectory tracks how a controller's decisions degrade over `15` iterations when faced with a `stochastic_depth_high` stress action, and measures how many steps of lead time the `Vec3` evaluator provides before a bad merge would occur.

The checked-in figure corresponds to the current curated lead-time artifact in the repo. The `scripts/run_torch_bundle.sh` script regenerates the same analysis from the `3`-seed run and stages the artifacts in git automatically.

The benchmark therefore validates the four intended `Vec3` branches on a benchmark-specific calibrated suite:

- pyrrhic improvement handling
- metric-gaming / validity failure detection
- damaged failure escalation
- incomparability detection

## Interpretation

The key signal in the lead-time trajectory is not that `Vec3` always finds a better optimization path. It is that `Vec3` preserves structural information about why a result should or should not be trusted, and that this information provides early warning — lead time — before a controller would otherwise merge a degrading outcome.

## Limitations

This figure should be presented as a benchmark-specific trajectory analysis, not as a general claim about all automated research systems. The stress action is explicit, and the window/tau calibration is specific to this benchmark. Lead-time results are strongest as evidence that structured evaluation provides decision-relevant early warning, not as a broad claim about general superiority.

## Suggested in-paper caption

Multi-step lead-time trajectory for `Vec3` versus binary evaluation under `stochastic_depth_high` stress on `CIFAR-100 + ResNet-18`. The `Vec3` evaluator provides `N` steps of lead time before the controller would merge a pyrrhic or metric-gamed outcome, while binary evaluation would adopt those cases immediately.
