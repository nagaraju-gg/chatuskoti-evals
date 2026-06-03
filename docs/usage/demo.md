# Demo Walkthrough

This repo is easiest to understand in three stops.

## 1. Lead-time trajectory analysis

Start here:

- [`lead_time/lead_time_analysis/summary.md`](../artifacts/strong_v1_3_torch/lead_time/lead_time_analysis/summary.md)

This is the strongest artifact. It shows a multi-step trajectory where a controller's decisions degrade over iterations, and measures how many steps of lead time the evaluator provides before a bad merge would occur.

## 2. Annotation cases

Then read:

- [`annotation_cases.csv`](../artifacts/strong_v1_3_torch/annotation_cases.csv)

Each trajectory step is extracted as a human-rater annotation case for ground-truth studies.

## 3. Ablation sweep

Finally read:

- [ablations/summary.md](../artifacts/strong_v1_3_torch/ablations/summary.md)

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
