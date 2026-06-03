# Paper revision notes for V1

## Claims to narrow

- Replace any claim that the framework is already production-ready with "prototype framework with deterministic evaluation and reproducible benchmark evidence."
- Keep `comparability` strictly about whether candidate and baseline can be compared at all.
- Keep Goodhart detection as a separate score, not the positive pole of comparability.

## V1 framing

- The first public artifact should describe a `CIFAR-100 + ResNet-18` benchmark with a real torch-backed calibration suite and a simulator convenience path.
- The demo should emphasize that the evaluation method, controller behavior, and richer history are the contribution.
- Thresholds should be described as benchmark-calibrated defaults, not universal constants.
