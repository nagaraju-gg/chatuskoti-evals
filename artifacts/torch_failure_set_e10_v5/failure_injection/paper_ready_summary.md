# Vec3 Failure Benchmark

Source artifact: [summary.md](summary.md)

## Summary

On the torch-backed adversarial calibration suite, `Vec3` matched `4/4` expected outcomes. A binary metric-only controller would have adopted `3` of the `4` cases, while `Vec3` would adopt `0`.

## Result Table

| Case | Metric | Binary | Vec3 | Key signal |
| --- | ---: | --- | --- | --- |
| Pyrrhic probe | `0.4486` | `adopt` | `hold` | `instability_gap`, `loss_instability` |
| Goodhart probe | `0.4536` | `adopt` | `reject` | `hyper_coherence`, `proxy_decoupling` |
| Broken probe | `0.3886` | `reject` | `rollback` | `exploding_gradients`, `loss_instability` |
| Eval TTA | `0.4250` | `adopt` | `reframe` | `eval_regime_changed` |

## Clean takeaway

The benchmark now demonstrates the exact comparative story we wanted:

- binary eval merges seductive bad results into “good enough to merge”
- `Vec3` distinguishes incoherent, Goodharted, damaged, and incomparable outcomes

## Safe claim

This is a benchmark-specific calibrated result on a controlled stress suite for `CIFAR-100 + ResNet-18`, not a general claim about all automated research systems.

## Suggested caption

`Vec3` versus binary evaluation on a four-case adversarial calibration suite. Binary evaluation would adopt three cases that `Vec3` routes to `hold`, `reject`, or `reframe`, while `Vec3` additionally escalates the damaged failure case to `rollback`.
