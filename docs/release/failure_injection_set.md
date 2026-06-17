# Canonical Failure Case Set

We include deliberately constructed cases to demonstrate representational insufficiency of scalar evaluation.

This is a core part of the representation argument, not an optional appendix. The failure case set answers the reviewer question: "Couldn't a smarter binary threshold handle these cases?"

## Cases

- `dropout_high`: metric improves, reliability degrades (Pyrrhic Pass)
- `stochastic_depth_high`: metric improves, validity collapses (Orthogonal Pass)
- `high_lr`: metric and internals both damaged (Broken Failure)
- `eval_tta`: metric improves under changed evaluation regime (Incomparable Pass)

Each case produces a positive metric delta (T > 0) but requires a distinct resolver action that scalar evaluation cannot distinguish.

## Scope

Claims should stay narrow:

- demonstration of representational insufficiency on one benchmark
- concrete three-axis representation and its implementation
- deterministic evaluation and interpretability outputs

Avoid broad claims about all automated research systems until wider validation exists.
