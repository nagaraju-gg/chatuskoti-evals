# Release Demo

Reading order for the current evidence bundle.

## 1. Start with the lead-time trajectory

- [`lead_time/lead_time_analysis/summary.md`](../artifacts/strong_v1_3_torch/lead_time/lead_time_analysis/summary.md)

The multi-step trajectory analysis is the strongest evidence: it tracks how a controller's decisions degrade over multiple iterations and measures how many steps of lead time the evaluator provides before a pyrrhic or metric-gamed outcome would be merged.

## 2. Show the annotation ground-truth cases

- [`annotation_cases.csv`](../artifacts/strong_v1_3_torch/annotation_cases.csv)

Each trajectory step is extracted as a human-rater annotation case for ground-truth studies.

## 3. Show that the extra axes matter

- [ablations/summary.md](../artifacts/strong_v1_3_torch/ablations/summary.md)

The ablation table is the compact proof that `reliability` and `validity` are doing real work rather than acting as decorative scores.

## Regenerate

```bash
bash scripts/run_torch_bundle.sh
```

The script runs lead-time analysis, extracts annotation cases, runs ablations, then stages the artifacts in git (replacing any previously tracked bundle).

## Public framing

Keep the claim narrow:

- benchmark-specific
- calibrated to `CIFAR-100 + ResNet-18`
- reproducible and inspectable

Avoid presenting the project as a general automated researcher or as proof that `Vec3` universally dominates metric-only gating.
